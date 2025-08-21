from typing import Dict
from video_streaming.app.db.models.camera import Camera
from video_streaming.app.db.models.camera_settings import CameraSettings
from sqlalchemy.orm import Session
from fastapi import HTTPException
import logging
from video_streaming.app.service.hardwareServiceClient import CameraClient



logger = logging.getLogger(__name__)

class CameraService:
    """
    Service class for camera-related operations that interacts with both
    the database and the hardware service.
    """   
        
    def __init__(self, hardware_service_url: str = "http://host.docker.internal:8003"):
        self.hardware_client = CameraClient(base_url=hardware_service_url)
        

     
                    
  
    
    def get_camera_by_id(self, db: Session, camera_id: int) -> Camera:
        """Get a camera by its ID."""
        camera = db.query(Camera).filter(Camera.id == camera_id).first()
        if not camera:
            raise HTTPException(status_code=404, detail=f"Camera with ID {camera_id} not found")
        return camera
    
    def start_camera(self, db: Session, camera_id: int) -> Dict[str, str]:
        """Start a camera by its ID."""
        camera = self.get_camera_by_id(db, camera_id)
        
        try:
            if camera.camera_type == "regular":
                response = self.hardware_client.start_opencv_camera(camera.camera_index)
            elif camera.camera_type == "basler":
                response = self.hardware_client.start_basler_camera(camera.serial_number)
            else:
                raise HTTPException(status_code=400, detail=f"Unsupported camera type: {camera.camera_type}")
            
            # Update camera status in database
            camera.status = True
            db.commit()
            
            return response
        except ConnectionError:
            raise HTTPException(status_code=503, detail="Hardware service unavailable")
        except Exception as e:
            logger.error(f"Error starting camera: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to start camera: {str(e)}")
    
    def stop_camera(self, db: Session) -> Dict[str, str]:
        """Stop the currently running camera."""
        try:
            response = self.hardware_client.stop_camera()
            
            # Update all cameras to inactive
            db.query(Camera).update({"status": False})
            db.commit()
            
            return response
        except ConnectionError:
            raise HTTPException(status_code=503, detail="Hardware service unavailable")
        except Exception as e:
            logger.error(f"Error stopping camera: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to stop camera: {str(e)}")
    
    def check_camera(self) -> Dict[str, bool]:
        """Check if the camera is running."""
        try:
            return self.hardware_client.check_camera()
        except ConnectionError:
            raise HTTPException(status_code=503, detail="Hardware service unavailable")
        except Exception as e:
            logger.error(f"Error checking camera: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to check camera: {str(e)}")
    


    
    
   
    
    