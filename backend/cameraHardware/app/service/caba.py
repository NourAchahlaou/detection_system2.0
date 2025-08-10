
import os
import threading
from pypylon import pylon
from sqlalchemy import func
from typing import Dict, Generator, List, Optional, Tuple
import cv2
from fastapi import HTTPException
import numpy as np
from sqlalchemy.orm import Session
from api.camera.models.camera_settings import UpdateCameraSettings

from database.piece.piece import Piece
from database.piece.piece_image import PieceImage
from hardware.camera.external_camera import get_available_cameras
from database.camera.camera_settings import CameraSettings
from database.camera.camera import Camera
from datetime import datetime
import asyncio
stop_event = asyncio.Event()

class FrameSource:
    """
    Allows capturing images from a camera frame-by-frame.
    """

    def __init__(self, cam_id=None,):
        self.camera_is_running = False
        self.cam_id = cam_id
        self.capture = None
        self.detection_service = None
        self.temp_photos = []  # To keep track of temporary photos
        self.basler_camera = None
        self.type= None
        self.confidence_threshold = 0.5  # Set the confidence threshold
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    # Reset virtual_storage whenever needed


    def _check_camera(self):
        return self.capture.isOpened()

    def detect_and_save_cameras(self, db: Session):
        available_cameras = get_available_cameras()  # Use the updated function
        print("Available Cameras:", available_cameras)

        for camera in available_cameras:
            if camera['type'] == 'basler':
                # Handle Basler camera
                print(f"Detected Basler Camera: {camera['caption']}")
                
                # Assume `device` contains necessary Basler information
                basler_info = {
                    "type": "basler",
                    "caption": camera['caption'],
                    "device_details": str(camera['device'])  # Convert to string for saving if needed
                }
                # Pass the correct camera type and other details
                camera_info = self.get_camera_info(camera_index=None,
                                                   serial_number=None,  # Basler cameras might not use index
                                                    model_name=camera['caption'], 
                                                    camera_type='basler', 
                                                    device=camera['device'])
                if camera_info:
                    print(camera_info)
                    self.save_camera_info(db, camera_info)

            elif camera['type'] == 'opencv':
                # Handle OpenCV-compatible camera
                index = camera.get('index')
                capture = cv2.VideoCapture(index)

                if capture.isOpened():
                    print(f"Detected OpenCV Camera: {camera['caption']}")

                    # Pass the correct camera type
                    camera_info = self.get_camera_info(camera_index=index, 
                                                       serial_number=None,
                                                    model_name=camera['caption'], 
                                                    camera_type='regular')
                    if camera_info:
                        print(camera_info)
                        self.save_camera_info(db, camera_info)

                    capture.release()
                else:
                    print(f"Failed to open OpenCV Camera at index {index}")
                    continue

        return available_cameras


    @staticmethod
    def get_camera_info(camera_index: Optional[int],serial_number:Optional[str], model_name: str, camera_type: str, device: Optional[pylon.DeviceInfo] = None) -> Optional[Dict]:
        """
        Retrieve or apply default camera settings based on the camera type (regular or Basler).
        """
        try:
            if camera_type == "regular":
                # For OpenCV-compatible cameras
                capture = cv2.VideoCapture(camera_index)
                if not capture.isOpened():
                    raise ValueError(f"Failed to open camera with index {camera_index}")

                # Getting camera settings
                settings = {
                    "exposure": capture.get(cv2.CAP_PROP_EXPOSURE),
                    "contrast": capture.get(cv2.CAP_PROP_CONTRAST),
                    "brightness": capture.get(cv2.CAP_PROP_BRIGHTNESS),
                    "focus": capture.get(cv2.CAP_PROP_FOCUS),
                    "aperture": capture.get(cv2.CAP_PROP_APERTURE),
                    "gain": capture.get(cv2.CAP_PROP_GAIN),
                    "white_balance": capture.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U),
                }
                return {
                    "camera_type": "regular",
                    "camera_index": camera_index,
                    "model": model_name,
                    "settings": settings,
                }

            elif camera_type == "basler" and device:
                # For Basler cameras
                camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device))
                camera.Open()

                try:
                    # Get the camera's node map and apply settings
                    node_map = camera.GetNodeMap()
                    print("Applying default settings for Basler camera...")
                    exposure_node = node_map.GetNode("ExposureTime")
                    exposure_node.SetValue(40000)
                    gain_node = node_map.GetNode("Gain")
                    gain_node.SetValue(0.8)
                    acquisition_mode_node = node_map.GetNode("AcquisitionMode")
                    acquisition_mode_node.SetValue("Continuous")

                    # Retrieve camera settings
                    settings = {
                        "exposure": 40000,
                        "contrast": None,
                        "brightness": None,
                        "focus": None,
                        "aperture": None,
                        "gain": 0.8,
                        "white_balance": None,
                    }

                    # Retrieve camera information
                    camera_info = camera.GetDeviceInfo()
                    serial_number = camera_info.GetSerialNumber() if camera_info.GetSerialNumber() else "unknown"

                    return {
                        "camera_type": "basler",
                        "serial_number": serial_number,
                        "model": model_name,
                        "settings": settings,
                    }

                except Exception as e:
                    print(f"Error retrieving camera info for Basler camera: {e}")
                    return None

                finally:
                    camera.Close()
            else:
                raise ValueError(f"Unsupported camera type: {camera_type}")

        except Exception as e:
            print(f"Error retrieving camera info for {camera_type} camera: {e}")
            return None

        finally:
            if 'capture' in locals() and capture.isOpened():
                capture.release()


    @staticmethod
    def save_camera_info(db: Session, camera_info: Dict):
        # Check if serial_number exists in camera_info
        serial_number = camera_info.get('serial_number', 'Unknown')

        existing_camera = db.query(Camera).filter(Camera.serial_number == serial_number).first()
        if existing_camera:
            print(f"Camera with serial number {serial_number} already registered.")
            return existing_camera

        # Create CameraSettings object
        settings = CameraSettings(
            exposure=camera_info['settings'].get('exposure'),
            contrast=camera_info['settings'].get('contrast'),
            brightness=camera_info['settings'].get('brightness'),
            focus=camera_info['settings'].get('focus'),
            aperture=camera_info['settings'].get('aperture'),
            gain=camera_info['settings'].get('gain'),
            white_balance=camera_info['settings'].get('white_balance'),
        )
        db.add(settings)
        db.commit()
        db.refresh(settings)

        # Create Camera object
        camera = Camera(
            camera_type=camera_info['camera_type'],
            camera_index=camera_info.get('camera_index'),
            serial_number=serial_number,
            model=camera_info['model'],
            status=False,
            settings_id=settings.id,
        )
        db.add(camera)
        db.commit()
        db.refresh(camera)

        print(f"Camera with serial number {serial_number} registered successfully!")
        return camera

    def get_camera_by_index(self, camera_id, db: Session):
       
        return db.query(Camera).filter(Camera.id == camera_id).first()

    def get_camera_model_and_ids(self, db: Session) -> List[Tuple[int, str]]:

        return db.query(Camera.id, Camera.model).all()
    


    def start(self, camera_id, db: Session):
        if self.camera_is_running:
            print("Camera is already running.")
            return

        if camera_id is None:
            raise ValueError("Please provide a camera ID to start the camera.")

        # Fetch the camera from the database using the provided ID
        camera = self.get_camera_by_index(camera_id, db)
        if camera is None:
            raise ValueError(f"No camera found with id {camera_id} in the database.")

        # Check the camera type (regular or Basler)
        if camera.camera_type == "regular":
            # For regular cameras (OpenCV compatible)
            self.capture = cv2.VideoCapture(camera.camera_index)
            if not self._check_camera():
                raise SystemError(f"Camera with index {camera.camera_index} not working.")
            self.type = "regular"
            print(f"Camera with index {camera.camera_index} started successfully.")

        elif camera.camera_type == "basler":
            # For Basler cameras
            print(f"Attempting to start Basler camera with serial number: {camera.serial_number}")

            try:
                device_info = pylon.DeviceInfo()
                serial_number = str(camera.serial_number)  # Convert serial_number to string
                device_info.SetSerialNumber(serial_number)  # Set serial number

                # Create and open the Basler camera
                self.basler_camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device_info))
                self.basler_camera.Open()
                self.converter = pylon.ImageFormatConverter()
                self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
                self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
                self.type = "basler"
                print("Basler camera opened successfully.")

                # Check if the camera is grabbing frames
                if not self.basler_camera.IsGrabbing():
                    print("Starting camera grabbing...")
                    self.basler_camera.StartGrabbing()
                    print("Camera grabbing started.")
                else:
                    print("Camera is already grabbing.")

            except Exception as e:
                raise SystemError(f"Failed to start Basler camera: {e}")

        else:
            raise ValueError(f"Unsupported camera type: {camera.camera_type}")

        # Mark the camera as running
        self.camera_is_running = True
        print(f"Camera with ID {camera_id} started successfully.")


   
    def stopInspection(self):
        if not self.camera_is_running:
            print("Camera is not running.")
            return

        # Stop regular OpenCV camera (if applicable)
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            print("Regular OpenCV camera stopped.")

        # Stop Basler camera (if applicable)
        if self.basler_camera is not None:
            try:
                self.basler_camera.StopGrabbing()
                self.basler_camera.Close()  # Close the Basler camera to release resources
                self.basler_camera = None
                print("Basler camera stopped and resources released.")
            except Exception as e:
                print(f"Error stopping Basler camera: {e}")

        self.camera_is_running = False
        self.type =None
        print("Camera stopped and resources released.")

        # If using stop_event for threading purposes
        if 'stop_event' in globals():
            stop_event.set()

    def stop(self):
        if not self.camera_is_running:
            print("Camera is not running.")
            return

        # Stop regular OpenCV camera (if applicable)
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            print("Regular OpenCV camera stopped.")

        # Stop Basler camera (if applicable)
        if self.basler_camera is not None:
            try:
                self.basler_camera.StopGrabbing()
                self.basler_camera.Close()  # Close the Basler camera to release resources
                self.basler_camera = None
                print("Basler camera stopped and resources released.")
            except Exception as e:
                print(f"Error stopping Basler camera: {e}")

        self.camera_is_running = False
        self.type = None
        print("Camera stopped and resources released.")

        # Set the event to signal stop
        stop_event = threading.Event() 
        stop_event.set()
        print("Stop event triggered.")



    def frame(self):
        assert self.camera_is_running, "Start the camera first by calling the start() method"

        success, frame = self.capture.read()
        if not success:
            raise SystemError("Failed to capture a frame. Ensure the camera is functioning properly.")
        
        if frame is None or frame.size == 0:
            raise ValueError("Captured frame is empty or invalid.")

        # Enhance the frame: adjust brightness and contrast
        enhanced_frame = cv2.convertScaleAbs(frame,  alpha=2, beta=-50)


        return enhanced_frame


    # def _gpu_encode_frame(self, frame: np.ndarray) -> bytes:
    #     """
    #     GPU-accelerated frame encoding using OpenCV's CUDA functions.
    #     """
    #     # Convert the frame to GPU memory (if it's not already in GPU memory)
    #     frame_gpu = cv2.cuda_GpuMat()
    #     frame_gpu.upload(frame)

    #     # Perform any GPU-based processing, like resizing or converting formats
    #     frame_resized = cv2.cuda.resize(frame_gpu, (640, 480))  # Example resizing operation

    #     # Download the processed frame back to CPU memory
    #     frame_resized_cpu = frame_resized.download()

    #     # Encode the frame
    #     _, buffer = cv2.imencode('.jpg', frame_resized_cpu, [cv2.IMWRITE_JPEG_QUALITY, 90])
    #     return buffer.tobytes()

    def generate_frames(self) -> Generator[bytes, None, None]:
        assert self.camera_is_running, "Start the camera first by calling the start() method"
        if self.type == "basler" and not hasattr(self, 'converter'):
            raise AttributeError("Converter is not initialized for Basler camera")

        while self.camera_is_running:
            try:
                if self.type == "regular":
                    success, frame = self.capture.read()
                    if not success:
                        break
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                elif self.type == "basler":
                    if self.basler_camera and self.basler_camera.IsGrabbing():
                        grab_result = self.basler_camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
                        if grab_result.GrabSucceeded():
                            frame = self.converter.Convert(grab_result).GetArray()
                            
                            # Optional: Add threading/multiprocessing for encoding
                            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            frame_bytes = buffer.tobytes()
                            
                            yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        grab_result.Release()
                    else:
                        break
            except Exception as e:
                print(f"Error during frame generation: {e}")
                break
    
# !TODO : change the timestamps into a counter 
#didn't use this 
    def next_frame(self, db: Session, save_folder: str, url: str, piece_label: str):
        assert self.camera_is_running, "Start the camera first by calling the start() method"

        # Check if the piece already exists in the database
        piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()

        if piece is None:
            # If the piece doesn't exist, create a new one
            piece = Piece(piece_label=piece_label, nbre_img=0)
            db.add(piece)
            db.commit()
            db.refresh(piece)

        
        

        success, frame = self.capture.read()
        if not success:
            raise SystemError("Failed to capture a frame")

        # Resize the frame if needed
        frame = cv2.resize(frame, (1000, 1000))

        # Generate a filename based on the current time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{piece_label}_{timestamp}.jpg"
        file_path = os.path.join(save_folder, filename)
        photo_url = os.path.join(url, filename)  # Renamed from `url` to `photo_url` to avoid shadowing the input parameter

        # Save the frame as an image file
        cv2.imwrite(file_path, frame)
        if piece.nbre_img == 10 :
            raise SystemError("number of images has already reached 10")
        self.temp_photos.append({
            'piece_id': piece.id,
            'file_path': file_path,
            'timestamp': datetime.now()
        })
        print(len(self.temp_photos))

        # Check if the number of photos captured is equal to max_photos
        if len(self.temp_photos) == 10 and piece.nbre_img <= 10:
            print (piece.nbre_img)
            for photo in self.temp_photos:
                piece.nbre_img += 1
                # Save the file path to the database associated with the piece
                new_photo = PieceImage(piece_id=photo['piece_id'], piece_path=photo['file_path'], timestamp=photo['timestamp'], url=photo_url)
                db.add(new_photo)
                db.commit()
                db.refresh(piece)
            self.temp_photos = [] 
        else:
            print("Number of photos is not compatible with the requested number")

        print("Photo of the piece is registered successfully!")
        return frame
    
    #instead of next_frame i used this 
    def capture_images(self, save_folder: str, url: str, piece_label: str):
        """
        Captures an image from the camera, resizes it to 1920x1152, and saves it with the specified naming format.
        """
        assert self.camera_is_running, "Start the camera first by calling the start() method"

        # Check if the camera is regular or Basler
        if self.type == "regular":
            success, frame = self.capture.read()
            if not success:
                raise SystemError("Failed to capture a frame")

        elif self.type == "basler":
            if not self.basler_camera.IsGrabbing():
                raise SystemError("Basler camera is not grabbing frames.")
            
            grab_result = self.basler_camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
            if not grab_result.GrabSucceeded():
                raise SystemError("Failed to grab a frame from Basler camera.")
            
            # Convert the Basler grab result to a frame
            frame = self.converter.Convert(grab_result).GetArray()
            grab_result.Release()

        else:
            raise ValueError("Unsupported camera type for image capture")

        # Resize the frame to 1920x1152
        frame = cv2.resize(frame, (1920, 1152))

        # Generate a filename based on the current time
        timestamp = datetime.now()
        image_count = len(self.temp_photos) + 1  # Get the current image count
        image_name = f"{piece_label}_{image_count}.jpg"  # Use the required naming format
        file_path = os.path.join(save_folder, image_name)
        photo_url = os.path.join(url, image_name)

        # Save the frame as an image file
        cv2.imwrite(file_path, frame)

        # Store the captured photo in a temporary list
        self.temp_photos.append({
            'piece_label': piece_label,
            'file_path': file_path,
            'timestamp': timestamp,
            'url': photo_url,
            'image_name': image_name
        })

        # Limit the number of captured photos to 10
        if len(self.temp_photos) > 10:
            raise SystemError("Already 10 pictures captured.")
        
        print(f"Captured {len(self.temp_photos)} photo(s) so far.")
        return frame

    def cleanup_temp_photos(self):
        for photo in self.temp_photos:
            try:
                os.remove(photo['file_path'])
            except FileNotFoundError:
                print(f"File {photo['file_path']} not found for deletion.")
        self.temp_photos = []  # Clear the temp list
        print("Temporary photos have been cleaned up.")
    

    def save_images_to_database(self, db: Session, piece_label: str):
        # Extract the group prefix from the piece label (e.g., "D123.12345")
        group_prefix = '.'.join(piece_label.split('.')[:2])
        # Ensure the model directory exists
  
        # piece_labels = get_piece_labels_by_group(group_prefix, db)
        # print("Retrieved piece labels:", piece_labels)

        # # Rotate and save images and annotations for each piece label
        # for piece_label in piece_labels:
        #     print(piece_label)
        #     rotate_and_save_images_and_annotations(piece_label, rotation_angles=[90, 180, 270])


        # Find the maximum class_data_id 
        max_class_data_id = db.query(func.max(Piece.class_data_id)).scalar()
        
        

        if len(self.temp_photos) == 0:
            raise SystemError("There are no photos captured in the temp list.")

        # Check if the piece already exists in the database
        piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()

        if piece is None:
            # If no piece exists, set class_data_id to 0; otherwise, increment max_class_data_id by 1
            next_class_data_id = (max_class_data_id + 1) if max_class_data_id is not None else 0
            
            # Create a new piece with the determined class_data_id
            piece = Piece(
                piece_label=piece_label,
                nbre_img=0,
                class_data_id=next_class_data_id
            )
            db.add(piece) 
            db.commit()
            db.refresh(piece)


        # Iterate over the list of captured photos and save them to the database
        if len(self.temp_photos) == 10 and piece.nbre_img <= 10:
            for index, photo in enumerate(self.temp_photos, start=1):
                piece.nbre_img += 1

                # Save the file path to the database associated with the piece
                new_photo = PieceImage(
                    piece_id=piece.id,
                    image_name=photo['image_name'],  # Use the formatted image name
                    piece_path=photo['file_path'],
                    timestamp=photo['timestamp'],
                    url=photo['url']
                )
                db.add(new_photo)
                db.commit()
                db.refresh(piece)

            self.temp_photos = []  # Clear the temporary list after saving

            print("All captured photos have been saved to the database.")
        else:
            print(len(self.temp_photos))
            self.temp_photos = [] 
        
            raise SystemError("Number of images has already reached 10")


 
    def get_camera(camera_id: int, db: Session):
        # Fetch the camera details
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        # Fetch the related settings
        settings = db.query(CameraSettings).filter(CameraSettings.id == camera.settings_id).all()        
        # Include settings in the camera details
        camera_data = camera.__dict__
        camera_data["settings"] = [setting.__dict__ for setting in settings]

        return camera_data
        
    @staticmethod
    def change_camera_settings(camera_id: int, camera_settings_update: UpdateCameraSettings, db: Session):
        # Fetch the camera by its ID
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")

        # Fetch the associated settings
        camera_settings = db.query(CameraSettings).filter(CameraSettings.id == camera.settings_id).first()
        if not camera_settings:
            raise HTTPException(status_code=404, detail="Camera settings not found")

        # Open the camera using OpenCV
        capture = cv2.VideoCapture(camera_settings.cameraIndex)
        if not capture.isOpened():
            raise HTTPException(status_code=500, detail="Failed to open camera")


        # Update the settings
        try:
            for key, value in camera_settings_update.dict(exclude_unset=True).items():
                setattr(camera_settings, key, value)

                if key == "brightness":
                    capture.set(cv2.CAP_PROP_BRIGHTNESS, value)
                elif key == "contrast":
                    capture.set(cv2.CAP_PROP_CONTRAST, value)
                elif key == "exposure":
                    capture.set(cv2.CAP_PROP_EXPOSURE, value)
                elif key == "white_balance":
                    capture.set(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, value)
                elif key == "focus":
                    capture.set(cv2.CAP_PROP_FOCUS, value)
                elif key == "aperture":
                    capture.set(cv2.CAP_PROP_APERTURE, value)
                elif key == "gain":
                    capture.set(cv2.CAP_PROP_GAIN, value)

            db.commit()
            db.refresh(camera_settings)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update camera settings: {e}")
        finally:
            capture.release()
        
        return camera   
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       