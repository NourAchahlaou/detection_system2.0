from datetime import datetime
import os
import re
import tempfile
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import func
from fastapi import HTTPException
import logging
import shutil

from artifact_keeper.app.db.models.piece import Piece
from artifact_keeper.app.db.models.piece_image import PieceImage
from artifact_keeper.app.services.hardwareServiceClient import CameraClient

logger = logging.getLogger(__name__)

class CaptureService:
    """
    Service class for camera-related operations that interacts with both
    the database and the hardware service.
    """  
    def __init__(self, hardware_service_url: str = "http://host.docker.internal:8003", 
                dataset_base_path: str = "/app/shared/dataset"):
        self.hardware_client = CameraClient(base_url=hardware_service_url)
        self.temp_photos = []
        
        # Updated dataset structure: shared_data/dataset/piece/piece/{piece_label}/
        self.dataset_base_path = dataset_base_path
        self.dataset_piece_path = os.path.join(self.dataset_base_path, "piece", "piece")
        
        # Create the base dataset structure if it doesn't exist with error handling
        try:
            os.makedirs(self.dataset_piece_path, exist_ok=True)
            logger.info(f"Successfully created/verified dataset directory: {self.dataset_piece_path}")
        except PermissionError as e:
            logger.error(f"Permission denied creating dataset directory {self.dataset_piece_path}: {str(e)}")
            logger.error("Please check directory permissions and user ownership")
            raise HTTPException(
                status_code=500, 
                detail=f"Cannot create dataset directory due to permission error. Please check directory permissions for {self.dataset_piece_path}"
            )
        except Exception as e:
            logger.error(f"Unexpected error creating dataset directory: {str(e)}")
            raise
        
        # Create a temp directory for storing images before moving to dataset
        try:
            self.temp_dir = tempfile.mkdtemp(prefix="camera_images_")
            logger.info(f"Initialized temp directory: {self.temp_dir}")
        except Exception as e:
            logger.error(f"Failed to create temp directory: {str(e)}")
            raise
        
        logger.info(f"Dataset base path: {self.dataset_base_path}")
        logger.info(f"Dataset piece path: {self.dataset_piece_path}")
    
    def _create_piece_directory(self, piece_label: str) -> tuple[str, str]:
        """Create directory structure for a piece if it doesn't exist."""
        piece_dir = os.path.join(self.dataset_piece_path, piece_label)
        images_dir = os.path.join(piece_dir, "images", "valid")
        labels_dir = os.path.join(piece_dir, "labels", "valid")
        
        # Create the directory structure: shared_data/dataset/piece/piece/{piece_label}/images/valid
        # and shared_data/dataset/piece/piece/{piece_label}/labels/valid
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        logger.info(f"Created/verified directory structure: {images_dir} and {labels_dir}")
        return images_dir, labels_dir

    def _get_piece_images_path(self, piece_label: str) -> str:
        """Get the images directory path for a piece."""
        return os.path.join(self.dataset_piece_path, piece_label, "images", "valid")

    def _get_piece_labels_path(self, piece_label: str) -> str:
        """Get the labels directory path for a piece."""
        return os.path.join(self.dataset_piece_path, piece_label, "labels", "valid")

    def _count_existing_images(self, piece_label: str) -> int:
        """Count existing images in the piece's dataset directory."""
        images_dir = self._get_piece_images_path(piece_label)
        if not os.path.exists(images_dir):
            return 0
        
        # Count .jpg files in the directory
        jpg_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
        return len(jpg_files)
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
            images_dir, labels_dir = self._create_piece_directory(piece_label)
            
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
                    
                    # Create PieceImage record - REMOVED the 'url' parameter that was causing the error
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
                "labels_path": labels_dir,
                "cleanup_status": {"message": "Temp files moved to dataset"}
            }
            
        except Exception as e:
            db.rollback()
            logger.error(f"Error saving images: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Failed to save images: {str(e)}")

    def get_piece_dataset_info(self, piece_label: str) -> Dict[str, Any]:
        """Get information about a piece's dataset directory."""
        images_dir = self._get_piece_images_path(piece_label)
        labels_dir = self._get_piece_labels_path(piece_label)
        
        if not os.path.exists(images_dir):
            return {
                "piece_label": piece_label,
                "images_path": images_dir,
                "labels_path": labels_dir,
                "exists": False,
                "image_count": 0,
                "images": []
            }
        
        # Get list of image files
        image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
        image_files.sort()  # Sort for consistent ordering
        
        # Get list of label files
        label_files = []
        if os.path.exists(labels_dir):
            label_files = [f for f in os.listdir(labels_dir) if f.lower().endswith('.txt')]
            label_files.sort()
        
        return {
            "piece_label": piece_label,
            "images_path": images_dir,
            "labels_path": labels_dir,
            "exists": True,
            "image_count": len(image_files),
            "images": image_files,
            "labels": label_files
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
                    'url': f'/api/artifact_keeper/captureImage/temp-image/{photo["image_name"]}'
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