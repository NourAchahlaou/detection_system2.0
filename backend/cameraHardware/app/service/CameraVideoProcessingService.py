import cv2
import asyncio
import aiohttp
import logging
import threading
import numpy as np
import time
import os
import queue
from datetime import datetime
from typing import Optional, AsyncGenerator, Tuple
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

# Import your existing camera services
from app.service.frameSource import FrameSource
from app.service.camera_capture import ImageCapture

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("camera_video_logs.log", mode='a')])

logger = logging.getLogger(__name__)

# Configuration for detection service
DETECTION_SERVICE_URL = "http://detection:8000"  # Adjust to your detection service URL

# Directory to save captured images
SAVE_DIR = "captured_images"
if not os.path.exists(SAVE_DIR):
    os.makedirs(SAVE_DIR)

class CameraVideoProcessingService:
    """
    Service to handle camera video feed processing and detection integration.
    This runs on the external camera hardware service (not dockerized).
    """
    
    def __init__(self):
        self.frame_source = FrameSource()
        self.image_capture = ImageCapture()
        self.detection_client = DetectionServiceClient()
        self.stop_event = threading.Event()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.frame_queue = queue.Queue(maxsize=3)
        
    def resize_frame_optimized(self, frame: np.ndarray, target_size=(640, 480)) -> np.ndarray:
        """Optimized frame resizing with better interpolation."""
        if frame.shape[:2] != target_size[::-1]:
            return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
        return frame
    
    async def process_frame_with_detection(self, frame: np.ndarray, target_label: str) -> Tuple[np.ndarray, bool, int]:
        """
        Send frame to detection service for processing and return results.
        """
        try:
            # Resize frame for consistent processing
            resized_frame = self.resize_frame_optimized(frame)
            
            # Send frame to detection service
            processed_frame, detected_target, non_target_count = await self.detection_client.process_frame(
                resized_frame, target_label
            )
            
            if processed_frame is not None:
                return processed_frame, detected_target, non_target_count
            else:
                # Fallback to original frame if detection service fails
                logger.warning("Detection service failed, returning original frame")
                return resized_frame, False, 0
                
        except Exception as e:
            logger.error(f"Error processing frame with detection: {e}")
            return self.resize_frame_optimized(frame), False, 0
    
    async def generate_video_frames(self, camera_type: str, camera_identifier: str, target_label: str) -> AsyncGenerator[bytes, None]:
        """
        Generate video frames with detection processing.
        
        Args:
            camera_type: 'opencv' or 'basler'
            camera_identifier: camera index for OpenCV or serial number for Basler
            target_label: target object label for detection
        """
        
        # Start the appropriate camera
        try:
            if camera_type.lower() == 'opencv':
                self.frame_source.start_opencv_camera(int(camera_identifier))
            elif camera_type.lower() == 'basler':
                self.frame_source.start_basler_camera(camera_identifier)
            else:
                raise ValueError(f"Unsupported camera type: {camera_type}")
                
            logger.info(f"Started {camera_type} camera with identifier {camera_identifier}")
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to start camera: {e}")
        
        # Wait for camera to be ready
        await self._wait_for_camera_ready()
        
        frame_counter = 0
        consecutive_failures = 0
        max_consecutive_failures = 3
        last_successful_frame = None
        
        # Performance settings
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]
        target_fps = 25
        frame_time = 1.0 / target_fps
        detection_skip = 2  # Process detection every 2nd frame
        
        try:
            while not self.stop_event.is_set():
                frame_start_time = time.time()
                
                try:
                    # Capture frame from camera
                    frame = await self._capture_frame()
                    
                    if frame is None:
                        consecutive_failures += 1
                        logger.warning(f"Frame capture failed {consecutive_failures} times")
                        
                        if consecutive_failures >= max_consecutive_failures:
                            logger.error("Too many consecutive frame failures, stopping")
                            break
                            
                        # Use cached frame if available
                        if last_successful_frame is not None:
                            frame = last_successful_frame
                        else:
                            await asyncio.sleep(0.05)
                            continue
                    else:
                        consecutive_failures = 0
                        # Cache successful frame (resize to save memory)
                        last_successful_frame = cv2.resize(frame, (320, 240))
                    
                    # Process detection (skip frames for better performance)
                    if frame_counter % detection_skip == 0:
                        try:
                            processed_frame, detected_target, non_target_count = await self.process_frame_with_detection(
                                frame, target_label
                            )
                            
                            if detected_target:
                                logger.info(f"Target object '{target_label}' detected!")
                            if non_target_count > 0:
                                logger.debug(f"Detected {non_target_count} non-target objects")
                                
                            frame = processed_frame
                            
                        except Exception as e:
                            logger.error(f"Error processing frame with detection: {e}")
                    
                    # Encode and yield frame
                    if frame is not None and len(frame.shape) == 3:
                        try:
                            _, buffer = cv2.imencode('.jpg', frame, encode_params)
                            if buffer is not None:
                                frame_bytes = buffer.tobytes()
                                yield (b'--frame\r\n'
                                    b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        except Exception as e:
                            logger.error(f"Error encoding frame: {e}")

                    frame_counter += 1
                    
                    # Maintain target FPS
                    elapsed = time.time() - frame_start_time
                    if elapsed < frame_time:
                        await asyncio.sleep(frame_time - elapsed)
                        
                except Exception as e:
                    logger.error(f"Error in frame generation loop: {e}")
                    consecutive_failures += 1
                    if consecutive_failures >= max_consecutive_failures:
                        break
                    await asyncio.sleep(0.05)

        except Exception as e:
            logger.error(f"Error in generate_video_frames: {e}")
        finally:
            self._stop_camera()
            self.stop_event.clear()
            logger.info("Video frame generation stopped and camera resources released")
    
    async def _capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame from the active camera."""
        try:
            if not self.frame_source.camera_is_running:
                return None
                
            if self.frame_source.type == "regular":
                success, frame = self.frame_source.capture.read()
                if success and frame is not None and frame.size > 0:
                    return frame
                    
            elif self.frame_source.type == "basler":
                if self.frame_source.basler_camera and self.frame_source.basler_camera.IsGrabbing():
                    from pypylon import pylon
                    grab_result = self.frame_source.basler_camera.RetrieveResult(
                        1000, pylon.TimeoutHandling_ThrowException
                    )
                    if grab_result.GrabSucceeded():
                        frame = self.frame_source.converter.Convert(grab_result).GetArray()
                        grab_result.Release()
                        if frame is not None and frame.size > 0:
                            return frame
                    grab_result.Release()
                    
            return None
            
        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            return None
    
    async def _wait_for_camera_ready(self, max_wait_time: int = 10) -> bool:
        """Wait for camera to be ready."""
        logger.info("Waiting for camera to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < max_wait_time:
            try:
                if self.frame_source.camera_is_running:
                    # Test frame capture
                    test_frame = await self._capture_frame()
                    if test_frame is not None:
                        logger.info("Camera verified with successful frame capture")
                        return True
                        
                await asyncio.sleep(1.0)
                
            except Exception as e:
                logger.warning(f"Error during camera readiness check: {e}")
                await asyncio.sleep(1.0)
        
        logger.error(f"Camera not ready after {max_wait_time} seconds")
        return False
    
    def _stop_camera(self):
        """Stop the camera and clean up resources."""
        try:
            self.frame_source.stop()
            logger.info("Camera stopped successfully")
        except Exception as e:
            logger.error(f"Error stopping camera: {e}")
    
    async def capture_single_image(self, camera_type: str, camera_identifier: str, 
                                 target_label: str, user_id: str) -> dict:
        """
        Capture a single image with detection processing.
        """
        try:
            # Start camera
            if camera_type.lower() == 'opencv':
                self.frame_source.start_opencv_camera(int(camera_identifier))
            elif camera_type.lower() == 'basler':
                self.frame_source.start_basler_camera(camera_identifier)
            else:
                raise ValueError(f"Unsupported camera type: {camera_type}")
            
            # Wait for camera readiness
            camera_ready = await self._wait_for_camera_ready()
            if not camera_ready:
                raise HTTPException(status_code=500, detail="Camera did not become ready in time")
            
            # Capture frame
            frame = await self._capture_frame()
            if frame is None:
                # Single retry
                await asyncio.sleep(0.5)
                frame = await self._capture_frame()
                
            if frame is None:
                raise HTTPException(status_code=500, detail="No frame captured from camera")
            
            # Process with detection
            processed_frame, detected_target, non_target_count = await self.process_frame_with_detection(
                frame, target_label
            )
            
            if detected_target:
                # Save detected image
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_name = f"captured_{target_label}_{timestamp}_{user_id}.jpg"
                image_path = os.path.join(SAVE_DIR, image_name)
                
                success, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                if not success:
                    raise HTTPException(status_code=500, detail="Failed to encode frame")
                
                with open(image_path, 'wb') as f:
                    f.write(buffer.tobytes())
                
                logger.info(f"Captured image saved at {image_path}")
                return {
                    "message": "Target object detected and image captured.",
                    "file_path": image_path,
                    "non_target_count": non_target_count
                }
            else:
                return {
                    "message": "No target object detected in the frame.",
                    "non_target_count": non_target_count
                }
                
        except Exception as e:
            logger.error(f"Error capturing single image: {e}")
            raise HTTPException(status_code=500, detail=f"Error capturing image: {e}")
        finally:
            self._stop_camera()
    
    def stop_video_processing(self):
        """Stop video processing."""
        self.stop_event.set()
        logger.info("Stop video signal sent")


class DetectionServiceClient:
    """
    Client to communicate with the dockerized detection service.
    """
    
    def __init__(self, detection_service_url: str = DETECTION_SERVICE_URL):
        self.detection_service_url = detection_service_url
        self.session = None
    
    async def _get_session(self):
        """Get or create aiohttp session."""
        if self.session is None or self.session.closed:
            timeout = aiohttp.ClientTimeout(total=10.0)
            self.session = aiohttp.ClientSession(timeout=timeout)
        return self.session
    
    async def load_model(self) -> bool:
        """Load the detection model on the detection service."""
        try:
            session = await self._get_session()
            async with session.get(f"{self.detection_service_url}/detection/load_model") as response:
                if response.status == 200:
                    logger.info("Detection model loaded successfully on detection service")
                    return True
                else:
                    logger.error(f"Failed to load model: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    async def process_frame(self, frame: np.ndarray, target_label: str) -> Tuple[Optional[np.ndarray], bool, int]:
        """
        Send frame to detection service for processing.
        
        Returns:
            Tuple of (processed_frame, detected_target, non_target_count)
        """
        try:
            # Encode frame as JPEG
            success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not success:
                logger.error("Failed to encode frame for detection service")
                return None, False, 0
            
            session = await self._get_session()
            
            # Prepare multipart form data
            data = aiohttp.FormData()
            data.add_field('frame', buffer.tobytes(), content_type='image/jpeg')
            data.add_field('target_label', target_label)
            
            async with session.post(
                f"{self.detection_service_url}/detection/process_frame", 
                data=data
            ) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Decode processed frame
                    import base64
                    frame_data = base64.b64decode(result.get('processed_frame', ''))
                    nparr = np.frombuffer(frame_data, np.uint8)
                    processed_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    detected_target = result.get('detected_target', False)
                    non_target_count = result.get('non_target_count', 0)
                    
                    return processed_frame, detected_target, non_target_count
                else:
                    logger.warning(f"Detection service returned status {response.status}")
                    return None, False, 0
                    
        except Exception as e:
            logger.error(f"Error communicating with detection service: {e}")
            return None, False, 0
    
    async def close(self):
        """Close the aiohttp session."""
        if self.session and not self.session.closed:
            await self.session.close()


# Create router for video processing endpoints
video_router = APIRouter(
    prefix="/video",
    tags=["Video Processing"],
    responses={404: {"description": "Not found"}},
)

# Global service instance
video_service = CameraVideoProcessingService()

@video_router.get("/feed/{camera_type}/{camera_identifier}")
async def video_feed_with_detection(camera_type: str, camera_identifier: str, target_label: str):
    """
    Stream video feed with detection processing.
    
    Args:
        camera_type: 'opencv' or 'basler'
        camera_identifier: camera index for OpenCV or serial number for Basler
        target_label: target object label for detection
    """
    try:
        # Load detection model first
        model_loaded = await video_service.detection_client.load_model()
        if not model_loaded:
            raise HTTPException(status_code=503, detail="Failed to load detection model")
        
        video_service.stop_event.clear()
        
        return StreamingResponse(
            video_service.generate_video_frames(camera_type, camera_identifier, target_label),
            media_type='multipart/x-mixed-replace; boundary=frame'
        )
        
    except Exception as e:
        logger.error(f"Error starting video feed: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start video feed: {e}")

@video_router.post("/stop")
async def stop_video_feed():
    """Stop the video feed."""
    try:
        video_service.stop_video_processing()
        return {"message": "Video feed stopped successfully"}
    except Exception as e:
        logger.error(f"Error stopping video feed: {e}")
        raise HTTPException(status_code=500, detail=f"Error stopping video feed: {e}")

@video_router.post("/capture/{camera_type}/{camera_identifier}")
async def capture_image_with_detection(
    camera_type: str, 
    camera_identifier: str, 
    target_label: str, 
    user_id: str
):
    """
    Capture a single image with detection processing.
    
    Args:
        camera_type: 'opencv' or 'basler'
        camera_identifier: camera index for OpenCV or serial number for Basler
        target_label: target object label for detection
        user_id: user identifier for image naming
    """
    try:
        # Load detection model first
        model_loaded = await video_service.detection_client.load_model()
        if not model_loaded:
            raise HTTPException(status_code=503, detail="Failed to load detection model")
        
        return await video_service.capture_single_image(
            camera_type, camera_identifier, target_label, user_id
        )
        
    except Exception as e:
        logger.error(f"Error capturing image: {e}")
        raise HTTPException(status_code=500, detail=f"Error capturing image: {e}")

@video_router.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check detection service
        detection_available = await video_service.detection_client.load_model()
        
        return {
            "status": "healthy" if detection_available else "degraded",
            "detection_service": "available" if detection_available else "unavailable",
            "camera_service": "ready",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Cleanup function to be called on application shutdown
async def cleanup_video_service():
    """Cleanup function to be called on application shutdown."""
    try:
        video_service.stop_video_processing()
        await video_service.detection_client.close()
        logger.info("Video service cleanup completed")
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")