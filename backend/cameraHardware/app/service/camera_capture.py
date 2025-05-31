import os
from datetime import datetime
import cv2
from sqlalchemy import func
from sqlalchemy.orm import Session
from pypylon import pylon


class ImageCapture:
    """
    Handles image capturing, storage, and database operations for pieces.
    """

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

        # # Limit the number of captured photos to 10
        # if len(frame_source.temp_photos) > 10:
        #     raise SystemError("Already 10 pictures captured.")
        
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

