from typing import Dict, Optional, List, Any
import cv2
import logging
from pypylon import pylon

from app.service.external_camera import get_available_cameras

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CameraManager:
    """
    Manages camera detection, registration, and settings.
    """
    
    @staticmethod
    def detect_cameras() -> List[Dict[str, Any]]:
        """
        Detects available cameras and registers them in the database.
        Returns a list of camera information in the format expected by the API.
        """
        logger.info("Starting camera detection process")
        available_cameras = get_available_cameras()
        logger.info(f"Available Cameras found: {len(available_cameras)}")
        for idx, cam in enumerate(available_cameras):
            logger.info(f"Camera {idx+1} details: {cam}")
        
        detected_cameras = []

        for camera in available_cameras:
            logger.info(f"Processing camera: {camera.get('caption', 'Unknown')}, Type: {camera.get('type', 'Unknown')}")
            
            if camera['type'] == 'basler':
                # Handle Basler camera
                logger.info(f"Processing Basler Camera: {camera['caption']}")
                
                # Pass the correct camera type and other details
                camera_info = CameraManager.get_camera_info(
                    camera_index=None,
                    serial_number=None,  
                    model_name=camera['caption'], 
                    camera_type='basler', 
                    device=camera['device']
                )
                
                if camera_info:
                    logger.info(f"Successfully retrieved Basler Camera Info: {camera_info}")
                    detected_cameras.append(camera_info)
                else:
                    logger.error(f"Failed to get information for Basler camera: {camera['caption']}")

            elif camera['type'] == 'regular' or camera['type'] == 'opencv':
                # Handle OpenCV-compatible camera
                index = camera.get('index')
                logger.info(f"Processing OpenCV Camera with index: {index}")
                
                try:
                    # First verify the camera can be opened
                    capture = cv2.VideoCapture(index)
                    if capture.isOpened():
                        logger.info(f"Successfully opened OpenCV Camera: {camera['caption']}")
                        
                        # Try reading a frame to confirm camera is working
                        ret, frame = capture.read()
                        if ret:
                            logger.info(f"Successfully read a frame from camera {index}")
                        else:
                            logger.warning(f"Could not read frame from camera {index}, but camera opened successfully")
                        
                        # Pass the correct camera type
                        camera_info = CameraManager.get_camera_info(
                            camera_index=index, 
                            serial_number=None,
                            model_name=camera['caption'], 
                            camera_type='regular'
                        )
                        
                        if camera_info:
                            logger.info(f"Successfully retrieved OpenCV Camera Info: {camera_info}")
                            detected_cameras.append(camera_info)
                        else:
                            logger.error(f"Failed to get information for OpenCV camera at index {index}")
                        
                    else:
                        logger.error(f"Failed to open OpenCV Camera at index {index}")
                    
                except Exception as e:
                    logger.exception(f"Exception while processing OpenCV camera: {e}")
                
                finally:
                    if 'capture' in locals() and capture is not None:
                        if capture.isOpened():
                            capture.release()
                            logger.info(f"Released camera at index {index}")

        logger.info(f"Total cameras with retrieved information: {len(detected_cameras)}")
        return detected_cameras

    @staticmethod
    def get_camera_info(camera_index: Optional[int], serial_number: Optional[str], model_name: str, camera_type: str, device: Optional[pylon.DeviceInfo] = None) -> Optional[Dict]:
        """
        Retrieve or apply default camera settings based on the camera type (regular or Basler).
        Returns camera information in the format expected by the API.
        """
        logger.info(f"Getting info for camera: {model_name}, Type: {camera_type}")
        
        try:
            if camera_type == "regular":
                # For OpenCV-compatible cameras
                logger.info(f"Attempting to get info for OpenCV camera index {camera_index}")
                
                try:
                    capture = cv2.VideoCapture(camera_index)
                    logger.info(f"Created VideoCapture object for index {camera_index}")
                    
                    if not capture.isOpened():
                        logger.error(f"Failed to open camera with index {camera_index} for info retrieval")
                        return None

                    # Getting camera settings with debug information
                    settings = {}
                    properties = [
                        ("exposure", cv2.CAP_PROP_EXPOSURE),
                        ("contrast", cv2.CAP_PROP_CONTRAST),
                        ("brightness", cv2.CAP_PROP_BRIGHTNESS),
                        ("focus", cv2.CAP_PROP_FOCUS),
                        ("aperture", cv2.CAP_PROP_APERTURE),
                        ("gain", cv2.CAP_PROP_GAIN),
                        ("white_balance", cv2.CAP_PROP_WHITE_BALANCE_BLUE_U),
                        ("width", cv2.CAP_PROP_FRAME_WIDTH),
                        ("height", cv2.CAP_PROP_FRAME_HEIGHT),
                        ("fps", cv2.CAP_PROP_FPS)
                    ]
                    
                    for name, prop in properties:
                        try:
                            value = capture.get(prop)
                            settings[name] = value
                            logger.info(f"Camera property {name}: {value}")
                        except Exception as e:
                            logger.warning(f"Could not get camera property {name}: {e}")
                            settings[name] = None
                    
                    # Try reading a frame just to validate camera is working
                    ret, frame = capture.read()
                    if ret:
                        logger.info(f"Successfully read test frame from camera {camera_index}")
                    else:
                        logger.warning(f"Could not read test frame from camera {camera_index}")
                    
                    # Return in the format expected by the API - with 'type' and 'caption' fields
                    result = {
                        "type": "regular",  # Changed from 'camera_type' to 'type'
                        "caption": model_name,  # Changed from 'model' to 'caption'
                        "index": camera_index,  # Added 'index' field
                        "settings": settings,
                        "can_capture_frame": ret
                    }
                    
                    logger.info(f"Successfully retrieved OpenCV camera info: {model_name}")
                    return result
                    
                except Exception as e:
                    logger.exception(f"Error getting OpenCV camera info: {e}")
                    return None
                
                finally:
                    if 'capture' in locals() and capture is not None:
                        if capture.isOpened():
                            capture.release()
                            logger.info(f"Released camera at index {camera_index}")

            elif camera_type == "basler" and device:
                logger.info(f"Attempting to get info for Basler camera {model_name}")
                # For Basler cameras
                try:
                    tl_factory = pylon.TlFactory.GetInstance()
                    camera = pylon.InstantCamera(tl_factory.CreateDevice(device))
                    logger.info("Created Basler InstantCamera object")
                    
                    camera.Open()
                    logger.info("Successfully opened Basler camera")

                    # Get the camera's node map and apply settings
                    node_map = camera.GetNodeMap()
                    logger.info("Retrieved node map from Basler camera")
                    
                    # Get nodes with more error checking
                    exposure_value = None
                    try:
                        exposure_node = node_map.GetNode("ExposureTime")
                        if exposure_node and exposure_node.IsWritable():
                            exposure_node.SetValue(40000)
                            exposure_value = 40000
                            logger.info("Successfully set exposure to 40000")
                        else:
                            logger.warning("Exposure node not available or not writable")
                    except Exception as e:
                        logger.warning(f"Could not set exposure: {e}")
                    
                    gain_value = None
                    try:
                        gain_node = node_map.GetNode("Gain")
                        if gain_node and gain_node.IsWritable():
                            gain_node.SetValue(0.8)
                            gain_value = 0.8
                            logger.info("Successfully set gain to 0.8")
                        else:
                            logger.warning("Gain node not available or not writable")
                    except Exception as e:
                        logger.warning(f"Could not set gain: {e}")
                    
                    try:
                        acquisition_mode_node = node_map.GetNode("AcquisitionMode")
                        if acquisition_mode_node and acquisition_mode_node.IsWritable():
                            acquisition_mode_node.SetValue("Continuous")
                            logger.info("Successfully set acquisition mode to Continuous")
                        else:
                            logger.warning("Acquisition mode node not available or not writable")
                    except Exception as e:
                        logger.warning(f"Could not set acquisition mode: {e}")

                    # Retrieve camera settings with values we successfully set or None if we failed
                    settings = {
                        "exposure": exposure_value,
                        "contrast": None,
                        "brightness": None,
                        "focus": None,
                        "aperture": None,
                        "gain": gain_value,
                        "white_balance": None,
                    }

                    # Retrieve camera information
                    camera_info = camera.GetDeviceInfo()
                    serial_number = camera_info.GetSerialNumber() if camera_info.GetSerialNumber() else "unknown"
                    logger.info(f"Retrieved Basler serial number: {serial_number}")

                    # Return in the format expected by the API - with 'type' and 'caption' fields
                    result = {
                        "type": "basler",  # Changed from 'camera_type' to 'type'
                        "caption": model_name,  # Changed from 'model' to 'caption'
                        "serial_number": serial_number,
                        "settings": settings,
                        "device": str(device)  # Include device info in serializable form
                    }
                    
                    logger.info(f"Successfully retrieved Basler camera info")
                    return result

                except Exception as e:
                    logger.exception(f"Error retrieving camera info for Basler camera: {e}")
                    return None

                finally:
                    if 'camera' in locals():
                        try:
                            if camera.IsOpen():
                                logger.info("Closing Basler camera connection")
                                camera.Close()
                        except Exception as e:
                            logger.warning(f"Error closing Basler camera: {e}")
            else:
                error_msg = f"Unsupported camera type: {camera_type}"
                logger.error(error_msg)
                if not device and camera_type == "basler":
                    logger.error("Device information missing for Basler camera")
                return None

        except Exception as e:
            logger.exception(f"Error retrieving camera info for {camera_type} camera: {e}")
            return None