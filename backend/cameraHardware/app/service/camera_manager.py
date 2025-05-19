from typing import Dict , Optional
import cv2
from pypylon import pylon

from app.service.external_camera import get_available_cameras

class CameraManager:
    """
    Manages camera detection, registration, and settings.
    """
    
    @staticmethod
    def detect_cameras():
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

