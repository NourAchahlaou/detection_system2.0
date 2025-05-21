# This should be added to your hardwareServiceClient.py file or wherever your CameraClient class is defined
import json
import logging
import requests
from typing import Dict, List, Any

logger = logging.getLogger(__name__)

class CameraClient:
    """Client for communicating with the Hardware Service API."""
    
    
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
            logger.info(f"Raw camera detection response: {json.dumps(raw_data, indent=2)}")
            
            cameras = []
            for camera_data in raw_data:
                # Map field names to match what CameraService expects
                camera_info = {
                    "type": camera_data.get("type"),  # Keep type (regular/basler)
                    "model": camera_data.get("caption"),  # Map caption -> model
                }
                
                # Handle camera type-specific fields
                if camera_info["type"] == "regular":
                    camera_info["camera_index"] = camera_data.get("index")
                elif camera_info["type"] == "basler":
                    camera_info["serial_number"] = camera_data.get("serial_number")
                
                # Process settings - this is the critical fix
                raw_settings = camera_data.get("settings", {})
                if raw_settings:
                    logger.info(f"Found settings for camera {camera_info['model']}: {json.dumps(raw_settings, indent=2)}")
                    
                    # Filter out non-database settings (width, height, fps) and keep only camera control settings
                    filtered_settings = {}
                    db_setting_keys = ['exposure', 'contrast', 'brightness', 'focus', 'aperture', 'gain', 'white_balance']
                    
                    for key in db_setting_keys:
                        if key in raw_settings and raw_settings[key] is not None:
                            filtered_settings[key] = raw_settings[key]
                    
                    camera_info["settings"] = filtered_settings
                    logger.info(f"Filtered settings for camera {camera_info['model']}: {json.dumps(filtered_settings, indent=2)}")
                else:
                    logger.warning(f"No valid settings found for camera {camera_info['model']}")
                    camera_info["settings"] = {}
                
                cameras.append(camera_info)
                logger.info(f"Processed camera data: {json.dumps(camera_info, indent=2)}")
                
            return cameras
            
        except requests.RequestException as e:
            logger.error(f"Error detecting cameras from hardware service: {str(e)}")
            raise ConnectionError(f"Failed to connect to hardware service: {str(e)}")
   
    # Other methods remain the same
    def start_opencv_camera(self, camera_index: int):
        """Start OpenCV camera."""
        try:
            response = requests.post(f"{self.base_url}/camera/opencv/start/{camera_index}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to start OpenCV camera: {str(e)}")
            
    def start_basler_camera(self, serial_number: str):
        """Start Basler camera."""
        try:
            response = requests.post(f"{self.base_url}/camera/basler/start/{serial_number}")
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
            response = requests.get(f"{self.base_url}/camera/check")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to check camera: {str(e)}")
            
    def capture_images(self, piece_label: str):
        """Capture images for a piece."""
        try:
            response = requests.post(
                f"{self.base_url}/camera/capture/{piece_label}",
                timeout=30  # Increased timeout for image capture
            )
            response.raise_for_status()
            return response.content
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to capture images: {str(e)}")
            
    def cleanup_temp_photos(self):
        """Clean up temporary photos."""
        try:
            response = requests.post(f"{self.base_url}/camera/cleanup")
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