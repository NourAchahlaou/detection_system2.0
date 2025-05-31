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
import shutil
from pathlib import Path
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
    
    def __init__(self, hardware_service_url: str = "http://host.docker.internal:8003", 
                 dataset_base_path: str = "/app/shared/dataset"):
        self.hardware_client = CameraClient(base_url=hardware_service_url)
        self.temp_photos = []
        
        # Dataset structure: dataset/piece/piece/{piece_name}/images/
        self.dataset_base_path = dataset_base_path
        self.dataset_piece_path = os.path.join(self.dataset_base_path, "piece", "piece")
        
        # Create the base dataset structure if it doesn't exist
        os.makedirs(self.dataset_piece_path, exist_ok=True)
        
        # Create a temp directory for storing images before moving to dataset
        self.temp_dir = tempfile.mkdtemp(prefix="camera_images_")
        logger.info(f"Initialized temp directory: {self.temp_dir}")
        logger.info(f"Dataset base path: {self.dataset_base_path}")
        logger.info(f"Dataset piece path: {self.dataset_piece_path}")

    def _create_piece_directory(self, piece_label: str) -> str:
        """Create directory structure for a piece if it doesn't exist."""
        piece_dir = os.path.join(self.dataset_piece_path, piece_label)
        images_dir = os.path.join(piece_dir, "images")
        
        # Create the directory structure: dataset/piece/piece/{piece_name}/images/
        os.makedirs(images_dir, exist_ok=True)
        
        logger.info(f"Created/verified directory structure: {images_dir}")
        return images_dir

    def _get_piece_images_path(self, piece_label: str) -> str:
        """Get the images directory path for a piece."""
        return os.path.join(self.dataset_piece_path, piece_label, "images")

    def _count_existing_images(self, piece_label: str) -> int:
        """Count existing images in the piece's dataset directory."""
        images_dir = self._get_piece_images_path(piece_label)
        if not os.path.exists(images_dir):
            return 0
        
        # Count .jpg files in the directory
        jpg_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
        return len(jpg_files)

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
        """Capture images for a piece and store them temporarily."""
        # Validate piece label format
        if not re.match(r'([A-Z]\d{3}\.\d{5})', piece_label):
            logger.error(f"Invalid piece_label format: {piece_label}")
            raise HTTPException(status_code=400, detail="Invalid piece_label format.")
        
        try:
            logger.info(f"Starting image capture for piece: {piece_label}")
            
            # Get the image from hardware service
            try:
                image_data = self.hardware_client.capture_images(piece_label)
                logger.info(f"Successfully received image data from hardware service. Size: {len(image_data)} bytes")
            except Exception as hw_error:
                logger.error(f"Hardware client error: {str(hw_error)}")
                raise HTTPException(status_code=503, detail=f"Hardware service error: {str(hw_error)}")
            
            # Count existing images in temp and dataset
            temp_count = len([p for p in self.temp_photos if p['piece_label'] == piece_label])
            logger.info(f"Temporary images count for {piece_label}: {temp_count}")
            
            try:
                dataset_count = self._count_existing_images(piece_label)
                logger.info(f"Dataset images count for {piece_label}: {dataset_count}")
            except Exception as count_error:
                logger.error(f"Error counting existing images: {str(count_error)}")
                dataset_count = 0  # Default to 0 if we can't count
            
            total_count = temp_count + dataset_count
            logger.info(f"Total images count for {piece_label}: {total_count}")
            
            # Check if we've reached the limit
            if total_count >= 10:
                logger.warning(f"Maximum photo limit reached for piece {piece_label}: {total_count}/10")
                raise HTTPException(status_code=400, detail="Maximum 10 photos per piece reached.")
            
            # FIXED: Calculate the next sequential number properly
            # Get existing temp image numbers for this piece
            existing_temp_numbers = []
            for photo in self.temp_photos:
                if photo['piece_label'] == piece_label:
                    # Extract number from image name (e.g., "G123.12345.123.11_5.jpg" -> 5)
                    match = re.search(r'_(\d+)\.jpg$', photo['image_name'])
                    if match:
                        existing_temp_numbers.append(int(match.group(1)))
            
            # Get existing dataset image numbers
            existing_dataset_numbers = []
            try:
                images_dir = self._get_piece_images_path(piece_label)
                if os.path.exists(images_dir):
                    for filename in os.listdir(images_dir):
                        if filename.lower().endswith('.jpg'):
                            match = re.search(r'_(\d+)\.jpg$', filename)
                            if match:
                                existing_dataset_numbers.append(int(match.group(1)))
            except Exception as e:
                logger.error(f"Error reading dataset directory: {str(e)}")
                existing_dataset_numbers = []
            
            # Combine all existing numbers
            all_existing_numbers = existing_temp_numbers + existing_dataset_numbers
            
            # Find the next available number (1-10)
            next_number = 1
            for i in range(1, 11):  # 1 to 10
                if i not in all_existing_numbers:
                    next_number = i
                    break
            
            # Double-check we haven't exceeded the limit
            if next_number > 10:
                logger.warning(f"Cannot assign number > 10 for piece {piece_label}")
                raise HTTPException(status_code=400, detail="Maximum 10 photos per piece reached.")
            
            # Store the image locally in temp directory
            timestamp = datetime.now()
            image_name = f"{piece_label}_{next_number}.jpg"
            temp_file_path = os.path.join(self.temp_dir, image_name)
            
            logger.info(f"Saving image to temp path: {temp_file_path} with number: {next_number}")
            
            # Ensure temp directory exists
            try:
                os.makedirs(self.temp_dir, exist_ok=True)
                logger.info(f"Temp directory verified: {self.temp_dir}")
            except Exception as dir_error:
                logger.error(f"Error creating temp directory: {str(dir_error)}")
                raise HTTPException(status_code=500, detail=f"Failed to create temp directory: {str(dir_error)}")
            
            # Save image to local temp directory
            try:
                with open(temp_file_path, 'wb') as f:
                    f.write(image_data)
                logger.info(f"Successfully wrote image to file: {temp_file_path}")
            except Exception as file_error:
                logger.error(f"Error writing image file: {str(file_error)}")
                raise HTTPException(status_code=500, detail=f"Failed to save image file: {str(file_error)}")
            
            # Store metadata in temp_photos list
            photo_metadata = {
                'piece_label': piece_label,
                'file_path': temp_file_path,
                'timestamp': timestamp,
                'image_name': image_name
            }
            self.temp_photos.append(photo_metadata)
            logger.info(f"Added photo metadata to temp list. Current temp photos count: {len(self.temp_photos)}")
            
            logger.info(f"Successfully captured image {next_number} for piece {piece_label} (total: {len(all_existing_numbers) + 1})")
            return image_data
            
        except HTTPException:
            # Re-raise HTTP exceptions as-is
            raise
        except ConnectionError as conn_error:
            logger.error(f"Connection error: {str(conn_error)}")
            raise HTTPException(status_code=503, detail="Hardware service unavailable")
        except Exception as e:
            logger.error(f"Unexpected error capturing images: {str(e)}")
            logger.error(f"Error type: {type(e).__name__}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"Failed to capture images: {str(e)}")
    
    def get_temp_photos(self, piece_label: str = None) -> List[Dict[str, Any]]:
        """Get temporary photos, optionally filtered by piece_label."""
        if piece_label:
            return [photo for photo in self.temp_photos if photo['piece_label'] == piece_label]
        return self.temp_photos
    
    def cleanup_temp_photos(self) -> Dict[str, str]:
        """Clean up temporary photos."""
        try:
            # Clean up local temp files
            for photo in self.temp_photos:
                if os.path.exists(photo['file_path']):
                    os.remove(photo['file_path'])
                    logger.info(f"Removed temp file: {photo['file_path']}")
            
            # Clear the temp_photos list
            self.temp_photos.clear()
            
            # Also call hardware service cleanup
            return self.hardware_client.cleanup_temp_photos()
        except ConnectionError:
            raise HTTPException(status_code=503, detail="Hardware service unavailable")
        except Exception as e:
            logger.error(f"Error cleaning up temp photos: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to clean up temporary photos: {str(e)}")
    
    def save_images_to_database(self, db: Session, piece_label: str) -> Dict[str, str]:
        """Save captured images to the dataset and database."""
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

        # Count existing images in dataset
        existing_dataset_images = self._count_existing_images(piece_label)
        
        if existing_dataset_images >= 10:
            raise HTTPException(status_code=400, detail="Maximum number of images reached for this piece.")
        
        try:
            # Get temporary photos for this piece
            piece_photos = self.get_temp_photos(piece_label)
            
            if not piece_photos:
                raise HTTPException(status_code=400, detail=f"No temporary photos found for piece {piece_label}.")
            
            # Check if adding these photos would exceed the limit
            if existing_dataset_images + len(piece_photos) > 10:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Adding {len(piece_photos)} photos would exceed the maximum of 10 images for this piece."
                )
            
            # Create the piece directory structure
            images_dir = self._create_piece_directory(piece_label)
            
            saved_images = []
            
            # Move each photo from temp to dataset and save to database
            for photo_data in piece_photos:
                try:
                    temp_file_path = photo_data['file_path']
                    
                    if not os.path.exists(temp_file_path):
                        logger.warning(f"Temp image file not found: {temp_file_path}")
                        continue
                    
                    # Create final path in dataset
                    final_file_path = os.path.join(images_dir, photo_data['image_name'])
                    
                    # Move file from temp to dataset directory
                    shutil.move(temp_file_path, final_file_path)
                    logger.info(f"Moved image from {temp_file_path} to {final_file_path}")
                    
                    # Create PieceImage record with dataset path
                    piece_image = PieceImage(
                        piece_id=piece.id,
                        file_name=photo_data['image_name'],
                        image_path=final_file_path,  # Store the actual dataset path
                        upload_date=photo_data['timestamp'],
                        is_deleted=False
                    )
                    
                    db.add(piece_image)
                    saved_images.append(photo_data['image_name'])
                    
                except Exception as e:
                    logger.error(f"Error processing image {photo_data['image_name']}: {str(e)}")
                    continue
            
            # Update the piece's image count
            piece.nbre_img = existing_dataset_images + len(saved_images)
            
            # Commit all changes
            db.commit()
            db.refresh(piece)
            
            # Clear temp photos for this piece from memory
            self.temp_photos = [p for p in self.temp_photos if p['piece_label'] != piece_label]
            
            logger.info(f"Successfully saved {len(saved_images)} images for piece {piece_label} to dataset")
            
            return {
                "message": f"Successfully saved {len(saved_images)} images to dataset",
                "piece_label": piece_label,
                "images_saved": saved_images,
                "total_images": piece.nbre_img,
                "dataset_path": images_dir,
                "cleanup_status": {"message": "Temp files moved to dataset"}
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving images: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save images: {str(e)}")

    def get_piece_dataset_info(self, piece_label: str) -> Dict[str, Any]:
        """Get information about a piece's dataset directory."""
        images_dir = self._get_piece_images_path(piece_label)
        
        if not os.path.exists(images_dir):
            return {
                "piece_label": piece_label,
                "dataset_path": images_dir,
                "exists": False,
                "image_count": 0,
                "images": []
            }
        
        # Get list of image files
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
        image_files.sort()  # Sort for consistent ordering
        
        return {
            "piece_label": piece_label,
            "dataset_path": images_dir,
            "exists": True,
            "image_count": len(image_files),
            "images": image_files
        }

    def __del__(self):
        """Cleanup temp directory on service destruction."""
        try:
            if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                logger.info(f"Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            logger.warning(f"Failed to cleanup temp directory: {str(e)}")
    def get_temp_images_for_piece(self, piece_label: str) -> List[Dict[str, Any]]:
        """Get temporary images for a specific piece with file URLs."""
        try:
            piece_photos = self.get_temp_photos(piece_label)
            
            temp_images = []
            for photo in piece_photos:
                # Create a temporary URL that can be served by your API
                temp_images.append({
                    'image_name': photo['image_name'],
                    'timestamp': photo['timestamp'].isoformat(),
                    'file_path': photo['file_path'],
                    # You'll need to create an endpoint to serve these temp files
                    'url': f'/api/artifact_keeper/camera/temp-image/{photo["image_name"]}'
                })
            
            logger.info(f"Retrieved {len(temp_images)} temporary images for piece {piece_label}")
            return temp_images
            
        except Exception as e:
            logger.error(f"Error getting temp images for piece {piece_label}: {str(e)}")
            return []

    def serve_temp_image(self, image_name: str) -> bytes:
        """Serve a temporary image file by name."""
        try:
            # Find the image in temp_photos
            for photo in self.temp_photos:
                if photo['image_name'] == image_name:
                    file_path = photo['file_path']
                    if os.path.exists(file_path):
                        with open(file_path, 'rb') as f:
                            return f.read()
                    else:
                        raise HTTPException(status_code=404, detail=f"Temp image file not found: {image_name}")
            
            raise HTTPException(status_code=404, detail=f"Temp image not found: {image_name}")
            
        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error serving temp image {image_name}: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to serve temp image: {str(e)}")  


    def delete_temp_image(self, piece_label: str, image_name: str) -> bool:
        """Delete a specific temporary image."""
        try:
            # Find and remove the image from temp_photos list
            photo_to_remove = None
            for photo in self.temp_photos:
                if photo['piece_label'] == piece_label and photo['image_name'] == image_name:
                    photo_to_remove = photo
                    break
            
            if not photo_to_remove:
                logger.warning(f"Temp image not found: {image_name} for piece {piece_label}")
                return False
            
            # Remove the physical file if it exists
            if os.path.exists(photo_to_remove['file_path']):
                os.remove(photo_to_remove['file_path'])
                logger.info(f"Deleted temp file: {photo_to_remove['file_path']}")
            
            # Remove from temp_photos list
            self.temp_photos.remove(photo_to_remove)
            logger.info(f"Removed temp image {image_name} for piece {piece_label}")
            
            # Log the current state after deletion
            remaining_temp_count = len([p for p in self.temp_photos if p['piece_label'] == piece_label])
            logger.info(f"Remaining temp images for {piece_label}: {remaining_temp_count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting temp image {image_name}: {str(e)}")
            return False
