import os
from datetime import datetime
import cv2
from sqlalchemy import func
from sqlalchemy.orm import Session
from pypylon import pylon

from piece_registry.app.db.models.piece import Piece
from piece_registry.app.db.models.piece_image import PieceImage

class ImageCapture:
    """
    Handles image capturing, storage, and database operations for pieces.
    """
    
    def next_frame(self, frame_source, db: Session, save_folder: str, url: str, piece_label: str):
        """
        Legacy method for capturing frames.
        Note: This method is kept for backward compatibility but using capture_images is recommended.
        """
        assert frame_source.camera_is_running, "Start the camera first by calling the start() method"

        # Check if the piece already exists in the database
        piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()

        if piece is None:
            # If the piece doesn't exist, create a new one
            piece = Piece(piece_label=piece_label, nbre_img=0)
            db.add(piece)
            db.commit()
            db.refresh(piece)

        success, frame = frame_source.capture.read()
        if not success:
            raise SystemError("Failed to capture a frame")

        # Resize the frame if needed
        frame = cv2.resize(frame, (1000, 1000))

        # Generate a filename based on the current time
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{piece_label}_{timestamp}.jpg"
        file_path = os.path.join(save_folder, filename)
        photo_url = os.path.join(url, filename)  # Renamed from `url` to `photo_url` to avoid shadowing the input parameter

        # Save the frame as an image file
        cv2.imwrite(file_path, frame)
        if piece.nbre_img == 10:
            raise SystemError("number of images has already reached 10")
        frame_source.temp_photos.append({
            'piece_id': piece.id,
            'file_path': file_path,
            'timestamp': datetime.now()
        })
        print(len(frame_source.temp_photos))

        # Check if the number of photos captured is equal to max_photos
        if len(frame_source.temp_photos) == 10 and piece.nbre_img <= 10:
            print(piece.nbre_img)
            for photo in frame_source.temp_photos:
                piece.nbre_img += 1
                # Save the file path to the database associated with the piece
                new_photo = PieceImage(piece_id=photo['piece_id'], piece_path=photo['file_path'], timestamp=photo['timestamp'], url=photo_url)
                db.add(new_photo)
                db.commit()
                db.refresh(piece)
            frame_source.temp_photos = [] 
        else:
            print("Number of photos is not compatible with the requested number")

        print("Photo of the piece is registered successfully!")
        return frame

    def capture_images(self, frame_source, save_folder: str, url: str, piece_label: str):
        """
        Captures an image from the camera, resizes it to 1920x1152, and saves it with the specified naming format.
        """
        assert frame_source.camera_is_running, "Start the camera first by calling the start() method"

        # Check if the camera is regular or Basler
        if frame_source.type == "regular":
            success, frame = frame_source.capture.read()
            if not success:
                raise SystemError("Failed to capture a frame")

        elif frame_source.type == "basler":
            if not frame_source.basler_camera.IsGrabbing():
                raise SystemError("Basler camera is not grabbing frames.")
            
            grab_result = frame_source.basler_camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
            if not grab_result.GrabSucceeded():
                raise SystemError("Failed to grab a frame from Basler camera.")
            
            # Convert the Basler grab result to a frame
            frame = frame_source.converter.Convert(grab_result).GetArray()
            grab_result.Release()

        else:
            raise ValueError("Unsupported camera type for image capture")

        # Resize the frame to 1920x1152
        frame = cv2.resize(frame, (1920, 1152))

        # Generate a filename based on the current time
        timestamp = datetime.now()
        image_count = len(frame_source.temp_photos) + 1  # Get the current image count
        image_name = f"{piece_label}_{image_count}.jpg"  # Use the required naming format
        file_path = os.path.join(save_folder, image_name)
        photo_url = os.path.join(url, image_name)

        # Save the frame as an image file
        cv2.imwrite(file_path, frame)

        # Store the captured photo in a temporary list
        frame_source.temp_photos.append({
            'piece_label': piece_label,
            'file_path': file_path,
            'timestamp': timestamp,
            'url': photo_url,
            'image_name': image_name
        })

        # Limit the number of captured photos to 10
        if len(frame_source.temp_photos) > 10:
            raise SystemError("Already 10 pictures captured.")
        
        print(f"Captured {len(frame_source.temp_photos)} photo(s) so far.")
        return frame

    def cleanup_temp_photos(self, frame_source):
        """
        Removes all temporary photos from disk and clears the temp list.
        """
        for photo in frame_source.temp_photos:
            try:
                os.remove(photo['file_path'])
            except FileNotFoundError:
                print(f"File {photo['file_path']} not found for deletion.")
        frame_source.temp_photos = []  # Clear the temp list
        print("Temporary photos have been cleaned up.")

    def save_images_to_database(self, frame_source, db: Session, piece_label: str):
        """
        Saves captured images to the database and associates them with the piece.
        """
        # Extract the group prefix from the piece label (e.g., "D123.12345")
        group_prefix = '.'.join(piece_label.split('.')[:2])
  
        # Find the maximum class_data_id 
        max_class_data_id = db.query(func.max(Piece.class_data_id)).scalar()
        
        if len(frame_source.temp_photos) == 0:
            raise SystemError("There are no photos captured in the temp list.")

        # Check if the piece already exists in the database
        piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()

        if piece is None:
            # If no piece exists, set class_data_id to 0; otherwise, increment max_class_data_id by 1
            next_class_data_id = (max_class_data_id + 1) if max_class_data_id is not None else 0
            
            # Create a new piece with the determined class_data_id
            piece = Piece(
                piece_label=piece_label,
                nbre_img=0,
                class_data_id=next_class_data_id
            )
            db.add(piece) 
            db.commit()
            db.refresh(piece)

        # Iterate over the list of captured photos and save them to the database
        if len(frame_source.temp_photos) == 10 and piece.nbre_img <= 10:
            for index, photo in enumerate(frame_source.temp_photos, start=1):
                piece.nbre_img += 1

                # Save the file path to the database associated with the piece
                new_photo = PieceImage(
                    piece_id=piece.id,
                    image_name=photo['image_name'],  # Use the formatted image name
                    piece_path=photo['file_path'],
                    timestamp=photo['timestamp'],
                    url=photo['url']
                )
                db.add(new_photo)
                db.commit()
                db.refresh(piece)

            frame_source.temp_photos = []  # Clear the temporary list after saving

            print("All captured photos have been saved to the database.")
        else:
            print(len(frame_source.temp_photos))
            frame_source.temp_photos = [] 
        
            raise SystemError("Number of images has already reached 10")