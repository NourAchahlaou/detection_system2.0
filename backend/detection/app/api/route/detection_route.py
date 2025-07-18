import cv2
import asyncio
import io
import logging
import threading
import numpy as np
from typing import Annotated, AsyncGenerator
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session
import requests
import time
from datetime import datetime

from detection.app.service.detection_service import DetectionSystem
from detection.app.db.session import get_session
# from database.inspection.InspectionImage import InspectionImage

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("training_logs.log", mode='a')])

logger = logging.getLogger(__name__)

router = APIRouter()
db_dependency = Annotated[Session, Depends(get_session)]
stop_event = threading.Event()

class TrainRequest(BaseModel):
    piece_labels: list[str]

class ArtifactKeeperClient:
    """Client for communicating with the Artifact Keeper microservice."""
    
    def __init__(self, base_url: str = "http://artifact_keeper:8001"):  # Adjust URL as needed
        self.base_url = base_url
        logger.info(f"Initializing ArtifactKeeperClient with base URL: {base_url}")
    
    def start_camera(self, camera_id: int):
        """Start a camera via artifact keeper."""
        try:
            response = requests.post(
                f"{self.base_url}/api/artifact_keeper/camera/start",
                json={"camera_id": camera_id}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to start camera via artifact keeper: {str(e)}")
    
    def stop_camera(self):
        """Stop camera via artifact keeper."""
        try:
            response = requests.post(f"{self.base_url}/api/artifact_keeper/camera/stop")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to stop camera via artifact keeper: {str(e)}")
    
    def check_camera(self):
        """Check camera status via artifact keeper."""
        try:
            response = requests.get(f"{self.base_url}/api/artifact_keeper/camera/check_camera")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to check camera via artifact keeper: {str(e)}")
    
    def get_video_feed(self):
        """Get video feed from artifact keeper."""
        try:
            response = requests.get(
                f"{self.base_url}/api/artifact_keeper/camera/video_feed",
                stream=True
            )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to get video feed from artifact keeper: {str(e)}")

# Initialize clients
artifact_keeper_client = ArtifactKeeperClient()
detection_system = None

async def load_model_once():
    """Load the model once when the application starts."""
    global detection_system
    if detection_system is None:
        detection_system = DetectionSystem()
        detection_system.get_my_model()

@router.get("/load_model")
async def load_model_endpoint():
    """Endpoint to load the model once when the inspection page is accessed."""
    try:
        await load_model_once()
        return {"message": "Model loaded successfully."}
    except Exception as e:
        logging.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while loading the model: {e}")

def resize_frame(frame: np.ndarray) -> np.ndarray:
    """Resize frame to the nearest size divisible by 32."""
    height, width = frame.shape[:2]
    new_height = (height // 32) * 32
    new_width = (width // 32) * 32
    return cv2.resize(frame, (new_width, new_height))

async def process_frame_async(frame: np.ndarray, target_label: str):
    """Asynchronously process a single frame to perform detection and contouring."""
    try:
        detection_results = detection_system.detect_and_contour(frame, target_label)
        if isinstance(detection_results, tuple):
            processed_frame = detection_results[0]
            detected_target = detection_results[1] if len(detection_results) > 1 else False
            non_target_count = detection_results[2] if len(detection_results) > 2 else 0
        else:
            processed_frame = detection_results
            detected_target = False
            non_target_count = 0
        
        return processed_frame, detected_target, non_target_count
    except cv2.error as e:
        logging.error(f"OpenCV error: {e}")
        return frame, False, 0
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return frame, False, 0

async def generate_frames_from_artifact_keeper(camera_id: int, target_label: str) -> AsyncGenerator[bytes, None]:
    """Generate video frames from artifact keeper service and perform detection."""
    try:
        # Start camera via artifact keeper
        start_result = artifact_keeper_client.start_camera(camera_id)
        logger.info(f"Camera started: {start_result}")
        
        # Get video feed from artifact keeper
        video_response = artifact_keeper_client.get_video_feed()
        
        detection_time = time.time()
        timeout_duration = 60
        object_detected = False
        frame_counter = 0
        
        # Process the video stream
        for chunk in video_response.iter_content(chunk_size=1024):
            if stop_event.is_set():
                break
                
            if chunk:
                # Parse the multipart stream to extract individual frames
                # This is a simplified approach - you might need more robust parsing
                if b'\xff\xd8' in chunk and b'\xff\xd9' in chunk:  # JPEG markers
                    try:
                        # Extract JPEG data
                        start_idx = chunk.find(b'\xff\xd8')
                        end_idx = chunk.find(b'\xff\xd9') + 2
                        
                        if start_idx != -1 and end_idx != 1:
                            jpeg_data = chunk[start_idx:end_idx]
                            
                            # Decode JPEG to OpenCV format
                            nparr = np.frombuffer(jpeg_data, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if frame is not None and frame_counter % 1 == 0:
                                # Resize frame
                                frame = cv2.resize(frame, (640, 480))
                                
                                # Process frame for detection
                                processed_frame, detected_target, non_target_count = await process_frame_async(frame, target_label)
                                
                                if non_target_count > 0:
                                    logging.error(f"Detected {non_target_count} pieces that do not belong.")
                                
                                if detected_target:
                                    object_detected = True
                                    detection_time = time.time()
                                
                                # Encode and yield the processed frame
                                if processed_frame.shape[2] == 3:
                                    _, buffer = cv2.imencode('.jpg', processed_frame)
                                    if _:
                                        frame_bytes = buffer.tobytes()
                                        yield (b'--frame\r\n'
                                              b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                            
                            frame_counter += 1
                            
                            # Timeout logic
                            if time.time() - detection_time > timeout_duration and not object_detected:
                                logger.debug("Timeout reached without object detection, stopping.")
                                break
                                
                    except Exception as e:
                        logger.error(f"Error processing frame: {e}")
                        continue
                        
                await asyncio.sleep(0.001)  # Small delay to prevent blocking
                
    except Exception as e:
        logger.error(f"Error in video stream: {e}")
        raise HTTPException(status_code=500, detail=f"Video stream error: {e}")
    finally:
        try:
            artifact_keeper_client.stop_camera()
        except:
            pass
        stop_event.clear()

@router.get("/video_feed")
async def video_feed(camera_id: int, target_label: str, db: Session = Depends(get_session)):
    """Endpoint to start the video feed with detection overlay."""
    await load_model_once()
    return StreamingResponse(
        generate_frames_from_artifact_keeper(camera_id, target_label),
        media_type='multipart/x-mixed-replace; boundary=frame'
    )

@router.post("/stop_camera_feed")
async def stop_camera_feed():
    """Stop camera feed."""
    try:
        stop_event.set()
        result = artifact_keeper_client.stop_camera()
        logger.info("Camera feed stopped successfully.")
        return {"message": "Camera feed stopped successfully.", "artifact_keeper_response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@router.post("/start_camera")
async def start_camera(camera_id: int):
    """Start camera via artifact keeper."""
    try:
        result = artifact_keeper_client.start_camera(camera_id)
        return {"message": "Camera started successfully.", "artifact_keeper_response": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start camera: {e}")

@router.get("/check_camera")
async def check_camera():
    """Check camera status via artifact keeper."""
    try:
        result = artifact_keeper_client.check_camera()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to check camera: {e}")

# # Directory to save captured images
# SAVE_DIR = "captured_images"
# import os
# if not os.path.exists(SAVE_DIR):
#     os.makedirs(SAVE_DIR)

# async def capture_frame_from_artifact_keeper(camera_id: int, of: str, target_label: str, user_id: str, db: Session):
#     """Capture a frame via artifact keeper and perform detection."""
#     await load_model_once()
    
#     try:
#         # Start camera via artifact keeper
#         start_result = artifact_keeper_client.start_camera(camera_id)
#         logger.info(f"Camera started for capture: {start_result}")
        
#         # Get a single frame from the video feed
#         video_response = artifact_keeper_client.get_video_feed()
        
#         # Extract first frame from the stream
#         frame = None
#         for chunk in video_response.iter_content(chunk_size=8192):
#             if b'\xff\xd8' in chunk and b'\xff\xd9' in chunk:
#                 start_idx = chunk.find(b'\xff\xd8')
#                 end_idx = chunk.find(b'\xff\xd9') + 2
                
#                 if start_idx != -1 and end_idx != 1:
#                     jpeg_data = chunk[start_idx:end_idx]
#                     nparr = np.frombuffer(jpeg_data, np.uint8)
#                     frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
#                     break
        
#         if frame is None:
#             raise HTTPException(status_code=500, detail="No frame captured from the camera.")
        
#         # Resize frame
#         frame = cv2.resize(frame, (640, 480))
        
#         # Detect object in the frame
#         processed_frame, detected_target, _ = await process_frame_async(frame, target_label)
        
#         if detected_target:
#             # Save the frame with the detected object
#             timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#             image_name = f"captured_{target_label}_{timestamp}_{user_id}.jpg"
#             image_path = os.path.join(SAVE_DIR, image_name)
            
#             if processed_frame.shape[2] == 3:
#                 # Encode and save the image
#                 success, buffer = cv2.imencode('.jpg', processed_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
#                 if not success:
#                     raise HTTPException(status_code=500, detail="Failed to encode frame.")
                
#                 with open(image_path, 'wb') as f:
#                     f.write(buffer.tobytes())
                
#                 logger.info(f"Captured image saved at {image_path}")
                
#                 # Save to database
#                 try:
#                     inspection_image = InspectionImage(
#                         image_path=image_path,
#                         image_name=image_name,
#                         order_of_fabrication=of,
#                         target_label=target_label,
#                         created_at=datetime.now(),
#                         type="inspection",
#                         user_id=user_id,
#                     )
                    
#                     db.add(inspection_image)
#                     db.commit()
                    
#                     logger.info(f"Image details saved successfully with ID: {inspection_image.id}")
                    
#                     return {
#                         "message": "Image captured and saved.",
#                         "file_path": image_path,
#                         "db_entry": inspection_image.id
#                     }
                    
#                 except Exception as e:
#                     logger.error(f"Error saving image details to database: {str(e)}")
#                     raise HTTPException(status_code=500, detail=f"Error saving image details: {str(e)}")
#             else:
#                 raise HTTPException(status_code=500, detail="Processed frame is not in correct format.")
#         else:
#             return {"message": "No target object detected in the frame."}
    
#     except Exception as e:
#         logger.error(f"Error capturing frame: {e}")
#         raise HTTPException(status_code=500, detail=f"Error capturing frame: {e}")
#     finally:
#         try:
#             artifact_keeper_client.stop_camera()
#         except:
#             pass

# @router.get("/capture_image")
# async def capture_image(camera_id: int, of: str, target_label: str, user_id: str, db: Session = Depends(get_session)):
#     """Capture image endpoint."""
#     return await capture_frame_from_artifact_keeper(camera_id, of, target_label, user_id, db)