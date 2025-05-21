from typing import Dict, List, Any
import requests
import logging

from artifact_keeper.app.response.camera import CameraClientResponse, CameraStatusResponse


logger = logging.getLogger(__name__)



class CameraClient:
    """
    Client for communicating with the hardware service running on the host machine.
    """
    def __init__(self, base_url: str = "http://host.docker.internal:8003"):
        self.base_url = base_url
        logger.info(f"Hardware service client initialized with base URL: {base_url}")

    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to hardware service with error handling."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.request(method, url, **kwargs)
            response.raise_for_status()
            return response
        except requests.exceptions.ConnectionError:
            logger.error(f"Connection error to hardware service at {url}")
            raise ConnectionError(f"Failed to connect to hardware service at {url}")
        except requests.exceptions.RequestException as e:
            logger.error(f"Request to hardware service failed: {str(e)}")
            raise e

    def detect_cameras(self) -> List[CameraClientResponse]:
        """Detect all available cameras."""
        response = self._make_request("GET", "/camera/detect")
        return [CameraClientResponse(**camera) for camera in response.json()]

    def start_opencv_camera(self, camera_index: int) -> Dict[str, str]:
        """Start an OpenCV camera using the provided index."""
        response = self._make_request(
            "POST", 
            "/camera/opencv/start", 
            json={"camera_index": camera_index}
        )
        return response.json()

    def start_basler_camera(self, serial_number: str) -> Dict[str, str]:
        """Start a Basler camera using the provided serial number."""
        response = self._make_request(
            "POST", 
            "/camera/basler/start", 
            json={"serial_number": serial_number}
        )
        return response.json()

    def stop_camera(self) -> Dict[str, str]:
        """Stop the currently running camera."""
        response = self._make_request("POST", "/camera/stop")
        return response.json()

    def check_camera(self) -> CameraStatusResponse:
        """Check if the camera is running."""
        response = self._make_request("GET", "/camera/check_camera")
        return CameraStatusResponse(**response.json())

    def capture_images(self, piece_label: str) -> bytes:
        """Capture images for a specific piece."""
        response = self._make_request("GET", f"/camera/capture_images/{piece_label}")
        return response.content

    def cleanup_temp_photos(self) -> Dict[str, str]:
        """Clean up temporary photos."""
        response = self._make_request("POST", "/camera/cleanup-temp-photos")
        return response.json()

    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get the status of all circuit breakers."""
        response = self._make_request("GET", "/camera/circuit-breaker-status")
        return response.json()

    def reset_circuit_breaker(self, breaker_name: str) -> Dict[str, str]:
        """Reset a specific circuit breaker."""
        response = self._make_request("POST", f"/camera/reset-circuit-breaker/{breaker_name}")
        return response.json()