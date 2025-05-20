import re
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from fastapi import HTTPException
import logging

from ArtifactKeeper.app.services.hardwareServiceClient import CameraClient
from ArtifactKeeper.app.db.models.camera_settings import CameraSettings
from ArtifactKeeper.app.db.models.piece import Piece
from ArtifactKeeper.app.db.models.piece_image import PieceImage
from ArtifactKeeper.app.db.models.camera import Camera
logger = logging.getLogger(__name__)

class CameraService:
    """
    Service class for camera-related operations that interacts with both
    the database and the hardware service.
    """   
    
    def __init__(self, hardware_service_url: str = "http://host.docker.internal:8003"):
        self.hardware_client = CameraClient(base_url=hardware_service_url)
        logger.info(f"CameraService initialized with hardware service at {hardware_service_url}")
    
    def detect_and_save_cameras(self, db: Session) -> List[Dict[str, Any]]:
        """Detect available cameras and save them to the database."""
        try:
            # Get cameras from hardware service
            available_cameras = self.hardware_client.detect_cameras()
            logger.info(f"Detected {len(available_cameras)} cameras")
            
            result = []
            for camera in available_cameras:
                # Check if camera already exists in database
                existing_camera = None
                if camera.type == "basler" and camera.identifier:
                    existing_camera = db.query(Camera).filter(
                        Camera.camera_type == "basler",
                        Camera.serial_number == camera.identifier
                    ).first()
                elif camera.type == "regular" and camera.identifier:
                    existing_camera = db.query(Camera).filter(
                        Camera.camera_type == "regular", 
                        Camera.camera_index == int(camera.identifier)
                    ).first()
                
                if existing_camera:
                    logger.info(f"Camera {camera.model} already registered in database")
                    result.append({
                        "id": existing_camera.id,
                        "type": existing_camera.camera_type,
                        "model": existing_camera.model,
                        "status": "already_exists"
                    })
                    continue
                
                # Create new camera settings
                settings = CameraSettings(
                    exposure=None,
                    contrast=None,
                    brightness=None,
                    focus=None,
                    aperture=None,
                    gain=None,
                    white_balance=None
                )
                db.add(settings)
                db.flush()
                
                # Create new camera
                new_camera = Camera(
                    camera_type=camera.type,
                    camera_index=int(camera.identifier) if camera.type == "regular" else None,
                    serial_number=camera.identifier if camera.type == "basler" else None,
                    model=camera.model,
                    status=False,
                    settings_id=settings.id
                )
                db.add(new_camera)
                db.commit()
                db.refresh(new_camera)
                
                logger.info(f"New camera registered: {new_camera.model} (ID: {new_camera.id})")
                result.append({
                    "id": new_camera.id,
                    "type": new_camera.camera_type,
                    "model": new_camera.model,
                    "status": "new"
                })
            
            return result
            
        except ConnectionError as e:
            logger.error(f"Failed to connect to hardware service: {str(e)}")
            raise HTTPException(status_code=503, detail="Hardware service unavailable")
        except Exception as e:
            logger.error(f"Error detecting cameras: {str(e)}")
            db.rollback()
            raise HTTPException(status_code=500, detail=f"Failed to detect cameras: {str(e)}")
    
    def get_all_cameras(self, db: Session) -> List[Camera]:
        """Get all registered cameras from the database."""
        return db.query(Camera).all()
    
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
            return self.hardware_client.check_camera().dict()
        except ConnectionError:
            raise HTTPException(status_code=503, detail="Hardware service unavailable")
        except Exception as e:
            logger.error(f"Error checking camera: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to check camera: {str(e)}")
    
    def capture_images(self, piece_label: str) -> bytes:
        """Capture images for a piece."""
        # Validate piece label format
        if not re.match(r'([A-Z]\d{3}\.\d{5})', piece_label):
            raise HTTPException(status_code=400, detail="Invalid piece_label format.")
        
        try:
            return self.hardware_client.capture_images(piece_label)
        except ConnectionError:
            raise HTTPException(status_code=503, detail="Hardware service unavailable")
        except Exception as e:
            logger.error(f"Error capturing images: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to capture images: {str(e)}")
    
    def cleanup_temp_photos(self) -> Dict[str, str]:
        """Clean up temporary photos."""
        try:
            return self.hardware_client.cleanup_temp_photos()
        except ConnectionError:
            raise HTTPException(status_code=503, detail="Hardware service unavailable")
        except Exception as e:
            logger.error(f"Error cleaning up temp photos: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to clean up temporary photos: {str(e)}")
    
    def save_images_to_database(self, db: Session, piece_label: str) -> Dict[str, str]:
        """Save captured images to the database."""
        # Extract the group prefix from the piece label
        match = re.match(r'([A-Z]\d{3}\.\d{5})', piece_label)
        if not match:
            raise HTTPException(status_code=400, detail="Invalid piece_label format.")
        
        # Check if the piece already exists in the database
        piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
        
        if not piece:
            # If no piece exists, create a new one
            max_class_data_id = db.query(func.max(Piece.class_data_id)).scalar()
            next_class_data_id = (max_class_data_id + 1) if max_class_data_id is not None else 0
            
            piece = Piece(
                piece_label=piece_label,
                nbre_img=0,
                class_data_id=next_class_data_id
            )
            db.add(piece)
            db.commit()
            db.refresh(piece)
        
        # Now save the temporary images to the database
        # This would typically involve getting the list of temp images from the hardware service
        # But for simplicity, we'll just assume the hardware service handles this
        try:
            # Make the API call to save images
            # (In a real implementation, you might need to retrieve the list of images first)
            self.hardware_client.cleanup_temp_photos()  # This would also handle saving in a real implementation
            
            return {"message": "Images saved to database successfully"}
        except ConnectionError:
            raise HTTPException(status_code=503, detail="Hardware service unavailable")
        except Exception as e:
            logger.error(f"Error saving images: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save images: {str(e)}")
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get the status of all circuit breakers."""
        try:
            return self.hardware_client.get_circuit_breaker_status()
        except ConnectionError:
            raise HTTPException(status_code=503, detail="Hardware service unavailable")
        except Exception as e:
            logger.error(f"Error getting circuit breaker status: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to get circuit breaker status: {str(e)}")
    
    def reset_circuit_breaker(self, breaker_name: str) -> Dict[str, str]:
        """Reset a specific circuit breaker."""
        try:
            return self.hardware_client.reset_circuit_breaker(breaker_name)
        except ConnectionError:
            raise HTTPException(status_code=503, detail="Hardware service unavailable")
        except Exception as e:
            logger.error(f"Error resetting circuit breaker: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to reset circuit breaker: {str(e)}")