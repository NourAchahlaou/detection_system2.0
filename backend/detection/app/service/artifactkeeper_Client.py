import logging
import requests

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("training_logs.log", mode='a')])

logger = logging.getLogger(__name__)

class ArtifactKeeperClient:
    """Client for communicating with the Artifact Keeper microservice."""
    
    def __init__(self, base_url: str = "http://artifact_keeper:8000"):  # Adjust URL as needed
        self.base_url = base_url
        logger.info(f"Initializing ArtifactKeeperClient with base URL: {base_url}")
    
    def start_camera(self, camera_id: int):
        """Start a camera via artifact keeper."""
        try:
            response = requests.post(
                f"{self.base_url}/camera/start",
                json={"camera_id": camera_id}
            )
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to start camera via artifact keeper: {str(e)}")
    
    def stop_camera(self):
        """Stop camera via artifact keeper."""
        try:
            response = requests.post(f"{self.base_url}/camera/stop")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to stop camera via artifact keeper: {str(e)}")
    
    def check_camera(self):
        """Check camera status via artifact keeper."""
        try:
            response = requests.get(f"{self.base_url}/camera/check_camera")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to check camera via artifact keeper: {str(e)}")
    
    def get_video_feed(self):
        """Get video feed from artifact keeper."""
        try:
            response = requests.get(
                f"{self.base_url}/camera/video_feed",
                stream=True
            )
            response.raise_for_status()
            return response
        except requests.RequestException as e:
            raise ConnectionError(f"Failed to get video feed from artifact keeper: {str(e)}")