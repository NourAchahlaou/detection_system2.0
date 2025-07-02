import os
from datetime import datetime
import cv2
from sqlalchemy import func
from sqlalchemy.orm import Session
from pypylon import pylon


class ImageCapture:
    """
    Handles image capturing for pieces - NO LOCAL STORAGE.
    """

    def capture_image_only(self, frame_source, piece_label: str):
        """
        Captures an image from the camera, resizes it to 1920x1152, and returns the frame.
        Does NOT save to local storage - only returns the image data.
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

        print(f"Captured image for piece {piece_label} - returning frame data only")
        return frame

    def cleanup_temp_photos(self, frame_source):
        """
        Clears the temp list - no file deletion needed since we don't store locally.
        """
        if hasattr(frame_source, 'temp_photos'):
            frame_source.temp_photos = []  # Clear the temp list
        print("Temporary photos list has been cleared.")