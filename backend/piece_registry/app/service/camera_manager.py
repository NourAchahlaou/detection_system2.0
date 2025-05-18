from typing import Dict, List, Optional, Tuple
from sqlalchemy.orm import Session
from fastapi import HTTPException
import cv2
from pypylon import pylon

from piece_registry.app.db.models.camera import Camera
from piece_registry.app.service.external_camera import get_available_cameras
from piece_registry.app.db.models.camera_settings import CameraSettings
from piece_registry.app.db.schemas.camera_settings import UpdateCameraSettings

class CameraManager:
    """
    Manages camera detection, registration, and settings.
    """
    
    @staticmethod
    def detect_and_save_cameras(db: Session):
        """
        Detects available cameras and registers them in the database.
        """
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
                camera_info = CameraManager.get_camera_info(camera_index=None,
                                                serial_number=None,  # Basler cameras might not use index
                                                model_name=camera['caption'], 
                                                camera_type='basler', 
                                                device=camera['device'])
                if camera_info:
                    print(camera_info)
                    CameraManager.save_camera_info(db, camera_info)

            elif camera['type'] == 'opencv':
                # Handle OpenCV-compatible camera
                index = camera.get('index')
                capture = cv2.VideoCapture(index)

                if capture.isOpened():
                    print(f"Detected OpenCV Camera: {camera['caption']}")

                    # Pass the correct camera type
                    camera_info = CameraManager.get_camera_info(camera_index=index, 
                                                    serial_number=None,
                                                    model_name=camera['caption'], 
                                                    camera_type='regular')
                    if camera_info:
                        print(camera_info)
                        CameraManager.save_camera_info(db, camera_info)

                    capture.release()
                else:
                    print(f"Failed to open OpenCV Camera at index {index}")
                    continue

        return available_cameras

    @staticmethod
    def get_camera_info(camera_index: Optional[int], serial_number: Optional[str], model_name: str, camera_type: str, device: Optional[pylon.DeviceInfo] = None) -> Optional[Dict]:
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
                # For Basler cameras - Fixed to use GetInstance() correctly
                tl_factory = pylon.TlFactory.GetInstance()
                camera = pylon.InstantCamera(tl_factory.CreateDevice(device))
                camera.Open()

                try:
                    # Get the camera's node map and apply settings
                    node_map = camera.GetNodeMap()
                    print("Applying default settings for Basler camera...")
                    
                    # Get nodes with more error checking
                    try:
                        exposure_node = node_map.GetNode("ExposureTime")
                        if exposure_node and exposure_node.IsWritable():
                            exposure_node.SetValue(40000)
                    except Exception as e:
                        print(f"Could not set exposure: {e}")
                    
                    try:
                        gain_node = node_map.GetNode("Gain")
                        if gain_node and gain_node.IsWritable():
                            gain_node.SetValue(0.8)
                    except Exception as e:
                        print(f"Could not set gain: {e}")
                    
                    try:
                        acquisition_mode_node = node_map.GetNode("AcquisitionMode")
                        if acquisition_mode_node and acquisition_mode_node.IsWritable():
                            acquisition_mode_node.SetValue("Continuous")
                    except Exception as e:
                        print(f"Could not set acquisition mode: {e}")

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
        """
        Saves camera information to the database.
        """
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

    @staticmethod
    def get_camera(camera_id: int, db: Session):
        """
        Gets camera details with its settings.
        """
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
        """
        Updates camera settings.
        """
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