# This should be added to your hardwareServiceClient.py file or wherever your CameraClient class is defined
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
        
    def detect_cameras(self) -> List[Any]:
        """Detect available cameras."""
        logger.info("Calling hardware service to detect cameras")
        try:
            response = requests.get(f"{self.base_url}/camera/detect")
            response.raise_for_status()
            
            # Log the raw response for debugging
            raw_data = response.json()
            logger.info(f"Raw camera detection response: {json.dumps(raw_data, indent=2)}")
            
            cameras = []
            for camera_data in raw_data:
                # Ensure we have a consistent format for the settings
                # Make sure 'settings' key exists and handle potential None values
                if 'settings' not in camera_data or camera_data['settings'] is None:
                    logger.warning(f"No valid settings found for camera {camera_data.get('caption', 'Unknown')}")
                    camera_data['settings'] = {}
                
                # Log the settings we found
                logger.info(f"Settings for camera {camera_data.get('caption', 'Unknown')}: {camera_data.get('settings', {})}")
                
                # Keep the original response structure intact
                cameras.append(camera_data)
                
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