import os

from typing import List, Dict, Any
from sqlalchemy.orm import Session
from fastapi import HTTPException
import logging
import json
from detection.app.service.hardwareServiceClient import CameraClient
from detection.app.db.models.camera_settings import CameraSettings

from detection.app.db.models.camera import Camera

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
    
    
    
   
    
    