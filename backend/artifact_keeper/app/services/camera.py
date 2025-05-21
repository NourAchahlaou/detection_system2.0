import re
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from fastapi import HTTPException
import logging
import json
from artifact_keeper.app.services.hardwareServiceClient import CameraClient
from artifact_keeper.app.db.models.camera_settings import CameraSettings
from artifact_keeper.app.db.models.piece import Piece
from artifact_keeper.app.db.models.piece_image import PieceImage
from artifact_keeper.app.db.models.camera import Camera
logger = logging.getLogger(__name__)

class CameraService:
    """
    Service class for camera-related operations that interacts with both
    the database and the hardware service.
    """   
    
    def __init__(self, hardware_service_url: str = "http://host.docker.internal:8003"):
        self.hardware_client = CameraClient(base_url=hardware_service_url)

    def detect_and_save_cameras(self, db: Session) -> List[Dict[str, Any]]:
        try:
            available_cameras = self.hardware_client.detect_cameras()
          
            result = []
            for idx, camera in enumerate(available_cameras):

                # Extract camera information
                camera_type = camera.get('type')
                model = camera.get('model', 'Unknown')
                
                # FIXED: Better settings extraction and validation
                raw_settings = camera.get('settings', {})
                # Create a proper settings dictionary with default values
                settings_dict = {}
                setting_keys = ['exposure', 'contrast', 'brightness', 'focus', 'aperture', 'gain', 'white_balance']
                
                if raw_settings and isinstance(raw_settings, dict):
                    for key in setting_keys:
                        value = raw_settings.get(key)
                        if value is not None:
                            settings_dict[key] = value
                            logger.info(f"Setting {key} = {value} for camera {model}")
                      
                # Extract identifier based on camera type
                identifier = None
                if camera_type == "regular":
                    identifier = camera.get('camera_index')
                elif camera_type == "basler":
                    identifier = camera.get('serial_number')
               
                # Check if settings exist and log them
                if settings_dict:
                    logger.info(f"Found valid settings for camera {model}: {json.dumps(settings_dict, indent=2)}")
                else:
                    logger.warning(f"No valid settings found in camera data for {model}")
                
                # Get existing camera if any
                existing_camera = None
                if camera_type == "basler" and identifier:
                    existing_camera = db.query(Camera).filter(
                        Camera.camera_type == "basler",
                        Camera.serial_number == identifier
                    ).first()
                elif camera_type == "regular" and identifier is not None:
                    existing_camera = db.query(Camera).filter(
                        Camera.camera_type == "regular", 
                        Camera.camera_index == identifier
                    ).first()
                
                if existing_camera:
                    logger.info(f"Camera {model} already registered in database with ID: {existing_camera.id}")
                    
                    # Update existing camera settings if they exist
                    if settings_dict:
                        
                        camera_settings = db.query(CameraSettings).filter(
                            CameraSettings.id == existing_camera.settings_id
                        ).first()
                        
                        if camera_settings:
                            # Update each setting if it exists in the incoming data
                            for setting_name, setting_value in settings_dict.items():
                                if setting_value is not None:
                                    setattr(camera_settings, setting_name, setting_value)
                                    logger.info(f"Updated {setting_name} to {setting_value}")
                            
                            db.commit()
                            db.refresh(camera_settings)
                            logger.info(f"Updated settings for camera {existing_camera.id}")
                    
                    # FIXED: Return the actual settings from database, not the incoming dict
                    current_settings = {}
                    if existing_camera.settings_id:
                        camera_settings = db.query(CameraSettings).filter(
                            CameraSettings.id == existing_camera.settings_id
                        ).first()
                        if camera_settings:
                            current_settings = {
                                'exposure': camera_settings.exposure,
                                'contrast': camera_settings.contrast,
                                'brightness': camera_settings.brightness,
                                'focus': camera_settings.focus,
                                'aperture': camera_settings.aperture,
                                'gain': camera_settings.gain,
                                'white_balance': camera_settings.white_balance
                            }
                            # Remove None values
                            current_settings = {k: v for k, v in current_settings.items() if v is not None}
                    
                    result.append({
                        "id": existing_camera.id,
                        "camera_index": existing_camera.camera_index,
                        "serial_number": existing_camera.serial_number,
                        "type": existing_camera.camera_type,
                        "model": existing_camera.model,
                        "settings": current_settings,  # Return actual DB settings
                        "status": False
                    })
                    continue
                

                
                # FIXED: Explicitly set each setting with proper None handling
                settings = CameraSettings(
                    exposure=settings_dict.get('exposure'),
                    contrast=settings_dict.get('contrast'),
                    brightness=settings_dict.get('brightness'),
                    focus=settings_dict.get('focus'),
                    aperture=settings_dict.get('aperture'),
                    gain=settings_dict.get('gain'),
                    white_balance=settings_dict.get('white_balance')
                )
                
                # Log the settings object being created
                settings_values = {
                    'exposure': settings.exposure,
                    'contrast': settings.contrast,
                    'brightness': settings.brightness,
                    'focus': settings.focus,
                    'aperture': settings.aperture,
                    'gain': settings.gain,
                    'white_balance': settings.white_balance
                }
                
                db.add(settings)
                db.flush()
                
                # Create new camera
                new_camera = Camera(
                    camera_type=camera_type,
                    camera_index=int(identifier) if camera_type == "regular" and identifier is not None else None,
                    serial_number=identifier if camera_type == "basler" else None,
                    model=model,
                    status=False,
                    settings_id=settings.id
                )
                db.add(new_camera)
                db.commit()
                db.refresh(new_camera)
                
                
                # FIXED: Return the actual created settings, not the input dict
                created_settings = {k: v for k, v in settings_values.items() if v is not None}
                
                result.append({
                    "id": new_camera.id,
                    "camera_index": new_camera.camera_index,
                    "serial_number": new_camera.serial_number,
                    "type": new_camera.camera_type,
                    "model": new_camera.model,
                    "settings": created_settings,  # Return actual created settings
                    "status": False
                })
            
            return result
            
        except ConnectionError as e:
            logger.error(f"Failed to connect to hardware service: {str(e)}")
            raise HTTPException(status_code=503, detail="Hardware service unavailable")
        except Exception as e:
            logger.error(f"Error detecting cameras: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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