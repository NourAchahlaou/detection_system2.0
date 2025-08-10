import json
import logging
import requests
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class CameraClient:
    """Client for communicating with the Hardware Service API."""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        logger.info(f"Initializing CameraClient with base URL: {base_url}")
        
    def detect_cameras(self) -> List[Dict[str, Any]]:
        """
        Detect available cameras and normalize the response format to match
        what the CameraService expects.
        
        Returns:
            List of camera dictionaries with standardized keys.
        """
        logger.info("Calling hardware service to detect cameras")
        try:
            response = requests.get(f"{self.base_url}/camera/detect")
            response.raise_for_status()
            
            # Log the raw response for debugging
            raw_data = response.json()
            logger.info(f"Raw hardware service response: {raw_data}")
            
            cameras = []
            for idx, camera_data in enumerate(raw_data):

                # CRITICAL FIX: The hardware service returns different field names
                # Hardware service returns: "caption", "index", "serial_number"  
                # We need to map to: "model", "camera_index", "serial_number"
                
                camera_info = {
                    "type": camera_data.get("type"),
                    "model": camera_data.get("caption"),  # Map "caption" -> "model"
                }
                
                # Handle camera type-specific fields
                if camera_info["type"] == "regular":
                    camera_info["camera_index"] = camera_data.get("index")  # Map "index" -> "camera_index"
                elif camera_info["type"] == "basler":
                    # Try top-level first
                    serial_number = camera_data.get("serial_number")

                    # If not found, try inside "device"
                    if not serial_number and isinstance(camera_data.get("device"), dict):
                        serial_number = camera_data["device"].get("serial_number")

                    if serial_number:
                        camera_info["serial_number"] = str(serial_number)
                        logger.info(f"Mapped serial_number for Basler camera: {serial_number}")
                    else:
                        logger.warning(f"No serial_number found for Basler camera {camera_info.get('model', 'unknown')}")
                        camera_info["serial_number"] = None
                
                # CRITICAL FIX: Process settings from hardware service
                hardware_settings = camera_data.get("settings", {})
                
                if hardware_settings and isinstance(hardware_settings, dict):
                    # Filter out non-database settings (width, height, fps) 
                    # and keep only camera control settings
                    filtered_settings = {}
                    db_setting_keys = ['exposure', 'contrast', 'brightness', 'focus', 'aperture', 'gain', 'white_balance']
                    
                    for key in db_setting_keys:
                        if key in hardware_settings and hardware_settings[key] is not None:
                            filtered_settings[key] = hardware_settings[key]
                            logger.info(f"  - Mapped setting {key}: {hardware_settings[key]}")
                    
                    camera_info["settings"] = filtered_settings
                else:
                    logger.warning(f"  - No valid settings found for camera {camera_info.get('model', 'unknown')}")
                    camera_info["settings"] = {}
                
                logger.info(f"Final camera_info for {camera_info.get('model', 'unknown')}: {camera_info}")
                cameras.append(camera_info)
                
            return cameras
            
        except requests.RequestException as e:
            logger.error(f"Error detecting cameras from hardware service: {str(e)}")
            raise ConnectionError(f"Failed to connect to hardware service: {str(e)}")
    # Other methods remain the same
    def start_opencv_camera(self, camera_index: int):
        """Start OpenCV camera."""
        try:
            # Correct endpoint without camera_index in URL
            response = requests.post(
                f"{self.base_url}/camera/opencv/start",
                json={"camera_index": camera_index}  # Send camera_index in request body
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to start OpenCV camera: {str(e)}")
        
    def start_basler_camera(self, serial_number: str):
        """Start Basler camera."""
        try:
            response = requests.post(f"{self.base_url}/camera/basler/start",
            json={"serial_number": serial_number}                     )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to start Basler camera: {str(e)}")
            
    def stop_camera(self):
        """Stop the current camera."""
        try:
            response = requests.post(f"{self.base_url}/camera/stop")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to stop camera: {str(e)}")
            
    def check_camera(self):
        """Check if a camera is running."""
        try:
            response = requests.get(f"{self.base_url}/camera/check_camera")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to check camera: {str(e)}")
            
    def capture_images(self, piece_label: str):
        """Capture images for a piece."""
        try:
            response = requests.get(
                f"{self.base_url}/camera/capture_images/{piece_label}",
                timeout=30  # Increased timeout for image capture
            )
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to capture images: {str(e)}")
            
    def cleanup_temp_photos(self):
        """Clean up temporary photos."""
        try:
            response = requests.post(f"{self.base_url}/camera/cleanup-temp-photos")
            response.raise_for_status()
            return response.json() 
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to clean up temporary photos: {str(e)}")
            
    def get_circuit_breaker_status(self):
        """Get circuit breaker status."""
        try:
            response = requests.get(f"{self.base_url}/circuit-breaker/status")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to get circuit breaker status: {str(e)}")
            
    def reset_circuit_breaker(self, breaker_name: str):
        """Reset a circuit breaker."""
        try:
            response = requests.post(f"{self.base_url}/circuit-breaker/reset/{breaker_name}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to reset circuit breaker: {str(e)}")