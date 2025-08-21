import threading
from pypylon import pylon
from typing import Generator
import cv2
import asyncio

stop_event = asyncio.Event()

class FrameSource:
    """
    Allows capturing images from a camera frame-by-frame.
    """

    def __init__(self, cam_id=None,):
        self.camera_is_running = False
        self.cam_id = cam_id
        self.capture = None
        self.detection_service = None
        self.temp_photos = []  # To keep track of temporary photos
        self.basler_camera = None
        self.type = None
        self.confidence_threshold = 0.5  # Set the confidence threshold
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned

    def _check_camera(self):
        if self.type == "regular":
            return self.capture.isOpened()
        elif self.type == "basler":
            return self.basler_camera and self.basler_camera.IsOpen()
        return False

    def start_opencv_camera(self, camera_index):
        """Start a regular OpenCV-compatible camera using its index."""
        if self.camera_is_running:
            print("Camera is already running.")
            return
        
        # For regular cameras (OpenCV compatible)
        self.capture = cv2.VideoCapture(camera_index)
        if not self._check_camera():
            raise SystemError(f"Camera with index {camera_index} not working.")
        
        self.type = "regular"
        self.camera_is_running = True
        print(f"OpenCV camera with index {camera_index} started successfully.")

    def start_basler_camera(self, serial_number):
        """Start a Basler camera using its serial number."""
        if self.camera_is_running:
            print("Camera is already running.")
            return
        
        print(f"Attempting to start Basler camera with serial number: {serial_number}")
        
        try:
            # Set up the Basler camera using its serial number
            device_info = pylon.DeviceInfo()
            serial_number = str(serial_number)  # Convert serial_number to string
            device_info.SetSerialNumber(serial_number)
            
            # Create and open the Basler camera
            self.basler_camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(device_info))
            self.basler_camera.Open()
            
            # Configure camera settings for better performance
            self.basler_camera.MaxNumBuffer = 5
            
            # Set up the converter for the Basler camera
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            
            self.type = "basler"
            print("Basler camera opened successfully.")
            
            # Check if the camera is grabbing frames
            if not self.basler_camera.IsGrabbing():
                print("Starting camera grabbing...")
                self.basler_camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
                print("Camera grabbing started.")
            else:
                print("Camera is already grabbing.")
                
            self.camera_is_running = True
            print(f"Basler camera with serial number {serial_number} started successfully.")
            
        except Exception as e:
            raise SystemError(f"Failed to start Basler camera: {e}")

    def stop(self):
        if not self.camera_is_running:
            print("Camera is not running.")
            return

        # Stop regular OpenCV camera (if applicable)
        if self.capture is not None:
            self.capture.release()
            self.capture = None
            print("Regular OpenCV camera stopped.")

        # Stop Basler camera (if applicable)
        if self.basler_camera is not None:
            try:
                if self.basler_camera.IsGrabbing():
                    self.basler_camera.StopGrabbing()
                self.basler_camera.Close()  # Close the Basler camera to release resources
                self.basler_camera = None
                print("Basler camera stopped and resources released.")
            except Exception as e:
                print(f"Error stopping Basler camera: {e}")

        self.camera_is_running = False
        self.type = None
        print("Camera stopped and resources released.")

    def frame(self):
        """Get a single frame - useful for image capture"""
        assert self.camera_is_running, "Start the camera first by calling the start() method"

        if self.type == "regular":
            success, frame = self.capture.read()
            if not success:
                raise SystemError("Failed to capture a frame from OpenCV camera.")
        elif self.type == "basler":
            if not self.basler_camera.IsGrabbing():
                raise SystemError("Basler camera is not grabbing.")
            
            grab_result = self.basler_camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
            if grab_result.GrabSucceeded():
                frame = self.converter.Convert(grab_result).GetArray()
                grab_result.Release()
            else:
                grab_result.Release()
                raise SystemError("Failed to capture a frame from Basler camera.")
        else:
            raise SystemError("Unknown camera type")
        
        if frame is None or frame.size == 0:
            raise ValueError("Captured frame is empty or invalid.")

        return frame

    def generate_frames(self) -> Generator[bytes, None, None]:
        """Generate MJPEG formatted frames for web streaming"""
        assert self.camera_is_running, "Start the camera first by calling the start() method"
        
        while self.camera_is_running:
            try:
                frame = self.frame()
                
                # Encode frame as JPEG
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                frame_bytes = buffer.tobytes()
                
                # Yield in MJPEG format
                yield (b'--frame\r\n'
                      b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

            except Exception as e:
                print(f"Error during frame generation: {e}")
                break

    def generate_raw_jpeg_stream(self) -> Generator[bytes, None, None]:
        """Generate raw JPEG frames for video streaming service consumption"""
        assert self.camera_is_running, "Start the camera first by calling the start() method"
        
        frame_count = 0
        print(f"Starting raw JPEG stream for {self.type} camera")
        
        while self.camera_is_running:
            try:
                frame = self.frame()
                
                # Encode frame as JPEG
                success, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                if success:
                    frame_bytes = buffer.tobytes()
                    
                    # Add JPEG markers to help with frame boundary detection
                    # Each JPEG starts with FF D8 and ends with FF D9
                    yield frame_bytes
                    frame_count += 1
                    
                    if frame_count % 100 == 0:
                        print(f"Streamed {frame_count} raw JPEG frames from {self.type} camera")
                else:
                    print("Failed to encode frame as JPEG")

            except Exception as e:
                print(f"Error during raw JPEG generation: {e}")
                break
        
        print(f"Raw JPEG stream stopped for {self.type} camera after {frame_count} frames")