import threading
from datetime import datetime
import cv2
from pypylon import pylon
import time
from typing import Optional


class ImageCapture:
    """
    Handles image capturing for pieces - NO LOCAL STORAGE.
    Enhanced thread safety and camera state management.
    """
    
    # Class-level lock to ensure thread safety across all instances
    _capture_lock = threading.RLock()  # Changed to RLock for reentrant locking
    _last_capture_time = 0
    _min_capture_interval = 0.1  # Minimum 100ms between captures
    
    def __init__(self):
        pass

    def _wait_for_capture_ready(self):
        """Ensure minimum time between captures to prevent camera overload"""
        current_time = time.time()
        time_since_last = current_time - ImageCapture._last_capture_time
        if time_since_last < ImageCapture._min_capture_interval:
            sleep_time = ImageCapture._min_capture_interval - time_since_last
            time.sleep(sleep_time)
        ImageCapture._last_capture_time = time.time()

    def _reset_basler_camera_state(self, frame_source):
        """Reset Basler camera to a known good state"""
        try:
            if frame_source.basler_camera and frame_source.basler_camera.IsOpen():
                # Stop grabbing if it's currently grabbing
                if frame_source.basler_camera.IsGrabbing():
                    frame_source.basler_camera.StopGrabbing()
                    time.sleep(0.1)  # Brief pause
                
                # Restart grabbing
                frame_source.basler_camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                time.sleep(0.1)  # Brief pause to ensure it's ready
                return True
        except Exception as e:
            print(f"Failed to reset Basler camera state: {e}")
            return False
        return False

    def _capture_basler_frame(self, frame_source, max_retries: int = 3) -> Optional[any]:
        """Capture frame from Basler camera with retry logic"""
        for attempt in range(max_retries):
            try:
                if not frame_source.basler_camera.IsGrabbing():
                    print(f"Camera not grabbing on attempt {attempt + 1}, trying to reset...")
                    if not self._reset_basler_camera_state(frame_source):
                        continue
                
                # Use a shorter timeout for faster failure detection
                timeout_ms = 2000  # 2 seconds
                grab_result = frame_source.basler_camera.RetrieveResult(
                    timeout_ms,
                    pylon.TimeoutHandling_ThrowException
                )
                
                if not grab_result.GrabSucceeded():
                    grab_result.Release()
                    print(f"Grab failed on attempt {attempt + 1}")
                    continue
                
                # Convert the grab result to a frame
                frame = frame_source.converter.Convert(grab_result).GetArray()
                grab_result.Release()
                
                if frame is not None and frame.size > 0:
                    return frame
                else:
                    print(f"Empty frame on attempt {attempt + 1}")
                    continue
                    
            except pylon.RuntimeException as e:
                error_msg = str(e)
                print(f"Basler RuntimeException on attempt {attempt + 1}: {error_msg}")
                
                # If it's the "thread waiting" error, try to reset camera state
                if "already a thread waiting" in error_msg:
                    print("Detected concurrent access, resetting camera state...")
                    if self._reset_basler_camera_state(frame_source):
                        time.sleep(0.2)  # Give camera time to stabilize
                    else:
                        break  # If we can't reset, don't keep trying
                elif "timeout" in error_msg.lower():
                    print("Timeout occurred, trying again...")
                    continue
                else:
                    # Unknown error, don't retry
                    raise SystemError(f"Basler camera error: {error_msg}")
                    
            except Exception as e:
                print(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == max_retries - 1:  # Last attempt
                    raise SystemError(f"Basler camera capture failed after {max_retries} attempts: {str(e)}")
                continue
        
        raise SystemError(f"Failed to capture frame after {max_retries} attempts")

    def capture_image_only(self, frame_source, piece_label: str):
        """
        Captures an image from the camera, resizes it to 1920x1152, and returns the frame.
        Does NOT save to local storage - only returns the image data.
        Enhanced thread-safe implementation with better error handling.
        """
        assert frame_source.camera_is_running, "Start the camera first by calling the start() method"

        # Use class-level lock to prevent concurrent access to the camera
        with ImageCapture._capture_lock:
            # Wait for minimum interval between captures
            self._wait_for_capture_ready()
            
            frame = None
            
            # Check if the camera is regular or Basler
            if frame_source.type == "regular":
                success, frame = frame_source.capture.read()
                if not success:
                    raise SystemError("Failed to capture a frame from OpenCV camera")

            elif frame_source.type == "basler":
                if not frame_source.basler_camera or not frame_source.basler_camera.IsOpen():
                    raise SystemError("Basler camera is not open")
                
                frame = self._capture_basler_frame(frame_source)

            else:
                raise ValueError("Unsupported camera type for image capture")

            if frame is None or frame.size == 0:
                raise SystemError("Captured frame is empty or invalid")

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