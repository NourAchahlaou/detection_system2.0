
import threading
from pypylon import pylon
from typing import  Generator
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
       # To keep track of temporary photos
        self.basler_camera = None
        self.type= None
        self.confidence_threshold = 0.5  # Set the confidence threshold
        self.converter = pylon.ImageFormatConverter()
        self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
        self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
    # Reset virtual_storage whenever needed

    def _check_camera(self):
        return self.capture.isOpened()

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
            
            # Set up the converter for the Basler camera
            self.converter = pylon.ImageFormatConverter()
            self.converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            
            self.type = "basler"
            print("Basler camera opened successfully.")
            
            # Check if the camera is grabbing frames
            if not self.basler_camera.IsGrabbing():
                print("Starting camera grabbing...")
                self.basler_camera.StartGrabbing()
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
                self.basler_camera.StopGrabbing()
                self.basler_camera.Close()  # Close the Basler camera to release resources
                self.basler_camera = None
                print("Basler camera stopped and resources released.")
            except Exception as e:
                print(f"Error stopping Basler camera: {e}")

        self.camera_is_running = False
        self.type = None
        print("Camera stopped and resources released.")

        # Set the event to signal stop
        stop_event = threading.Event() 
        stop_event.set()
        print("Stop event triggered.")

    #NOTE: This method is not used in the current implementation IT IS FOR FUTURE USE detection and identification
    def frame(self):
        assert self.camera_is_running, "Start the camera first by calling the start() method"

        success, frame = self.capture.read()
        if not success:
            raise SystemError("Failed to capture a frame. Ensure the camera is functioning properly.")
        
        if frame is None or frame.size == 0:
            raise ValueError("Captured frame is empty or invalid.")

        # Enhance the frame: adjust brightness and contrast
        enhanced_frame = cv2.convertScaleAbs(frame, alpha=2, beta=-50)

        return enhanced_frame

    def generate_frames(self) -> Generator[bytes, None, None]:
        assert self.camera_is_running, "Start the camera first by calling the start() method"
        if self.type == "basler" and not hasattr(self, 'converter'):
            raise AttributeError("Converter is not initialized for Basler camera")

        while self.camera_is_running:
            try:
                if self.type == "regular":
                    success, frame = self.capture.read()
                    if not success:
                        break
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    frame_bytes = buffer.tobytes()
                    yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                elif self.type == "basler":
                    if self.basler_camera and self.basler_camera.IsGrabbing():
                        grab_result = self.basler_camera.RetrieveResult(1000, pylon.TimeoutHandling_ThrowException)
                        if grab_result.GrabSucceeded():
                            frame = self.converter.Convert(grab_result).GetArray()
                            
                            # Optional: Add threading/multiprocessing for encoding
                            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                            frame_bytes = buffer.tobytes()
                            
                            yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                        grab_result.Release()
                    else:
                        break
            except Exception as e:
                print(f"Error during frame generation: {e}")
                break

