from datetime import datetime
import os
import re
import tempfile
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
        self.temp_photos = []
                # Create a temp directory for storing images
        self.temp_dir = tempfile.mkdtemp(prefix="camera_images_")
        logger.info(f"Initialized temp directory: {self.temp_dir}")

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
            return self.hardware_client.check_camera()
        except ConnectionError:
            raise HTTPException(status_code=503, detail="Hardware service unavailable")
        except Exception as e:
            logger.error(f"Error checking camera: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to check camera: {str(e)}")
    
    def capture_images(self, piece_label: str) -> bytes:
        """Capture images for a piece and store them locally."""
        # Validate piece label format
        if not re.match(r'([A-Z]\d{3}\.\d{5})', piece_label):
            raise HTTPException(status_code=400, detail="Invalid piece_label format.")
        
        try:
            # Get the image from hardware service
            image_data = self.hardware_client.capture_images(piece_label)
            
            # Store the image locally in temp directory
            timestamp = datetime.now()
            image_count = len([p for p in self.temp_photos if p['piece_label'] == piece_label]) + 1
            image_name = f"{piece_label}_{image_count}.jpg"
            file_path = os.path.join(self.temp_dir, image_name)
            
            # Save image to local temp directory
            with open(file_path, 'wb') as f:
                f.write(image_data)
            
            # Store metadata in temp_photos list
            photo_metadata = {
                'piece_label': piece_label,
                'file_path': file_path,
                'timestamp': timestamp,
                'image_name': image_name
            }
            self.temp_photos.append(photo_metadata)
            
            # Limit to 10 photos per piece
            piece_photos = [p for p in self.temp_photos if p['piece_label'] == piece_label]
            if len(piece_photos) > 10:
                # Remove the oldest photo for this piece
                oldest_photo = min(piece_photos, key=lambda x: x['timestamp'])
                self.temp_photos.remove(oldest_photo)
                if os.path.exists(oldest_photo['file_path']):
                    os.remove(oldest_photo['file_path'])
                raise HTTPException(status_code=400, detail="Maximum 10 photos per piece. Oldest photo removed.")
            
            logger.info(f"Captured image {image_count} for piece {piece_label}")
            return image_data
            
        except ConnectionError:
            raise HTTPException(status_code=503, detail="Hardware service unavailable")
        except Exception as e:
            logger.error(f"Error capturing images: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to capture images: {str(e)}")
    
    def get_temp_photos(self, piece_label: str = None) -> List[Dict[str, Any]]:
        """Get temporary photos, optionally filtered by piece_label."""
        if piece_label:
            return [photo for photo in self.temp_photos if photo['piece_label'] == piece_label]
        return self.temp_photos
    
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

        if piece.nbre_img >= 10:
            raise HTTPException(status_code=400, detail="Maximum number of images reached for this piece.")
        
        try:
            # Get temporary photos for this piece
            piece_photos = self.get_temp_photos(piece_label)
            
            if not piece_photos:
                raise HTTPException(status_code=400, detail=f"No temporary photos found for piece {piece_label}.")
            
            # Check if adding these photos would exceed the limit
            if piece.nbre_img + len(piece_photos) > 10:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Adding {len(piece_photos)} photos would exceed the maximum of 10 images for this piece."
                )
            
            saved_images = []
            
            # Save each photo to the database
            for photo_data in piece_photos:
                try:
                    # Create PieceImage record (corrected field names)
                    piece_image = PieceImage(
                        piece_id=piece.id,
                        file_name=photo_data['image_name'],      # Use file_name (not image_name)
                        image_path=photo_data['file_path'],      # Use image_path (not image_data)
                        upload_date=photo_data['timestamp'],     # Use upload_date (not created_at)
                        is_deleted=False                         # Set default value
                    )
                    
                    db.add(piece_image)
                    saved_images.append(photo_data['image_name'])
                    
                except FileNotFoundError:
                    logger.warning(f"Image file not found: {photo_data['file_path']}")
                    continue

            
            # Update the piece's image count
            piece.nbre_img += len(saved_images)
            
            # Commit all changes
            db.commit()
            db.refresh(piece)
            
            # Clean up temporary photos for this piece
            cleanup_response = self.cleanup_temp_photos()
            logger.info(f"Successfully saved {len(saved_images)} images for piece {piece_label}")
            
            return {
                "message": f"Successfully saved {len(saved_images)} images to database",
                "piece_label": piece_label,
                "images_saved": saved_images,
                "total_images": piece.nbre_img,
                "cleanup_status": cleanup_response
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving images: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save images: {str(e)}")

    def __del__(self):
        """Cleanup temp directory on service destruction."""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                import shutil
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {str(e)}")