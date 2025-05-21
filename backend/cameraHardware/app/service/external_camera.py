from typing import Dict, List
from pypylon import pylon
import win32com.client
import cv2
import logging

# Set up basic logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_usb_devices() -> List[Dict[str, str]]:
    """Use WMI to find camera devices with more comprehensive detection"""
    logger.info("Searching for USB camera devices...")
    wmi = win32com.client.GetObject("winmgmts:")
    # Use a broader query to catch more potential cameras
    devices = wmi.ExecQuery("SELECT * FROM Win32_PnPEntity WHERE " + 
                          "(Caption LIKE '%Camera%' OR " + 
                          "Caption LIKE '%cam%' OR " + 
                          "Caption LIKE '%Webcam%' OR " + 
                          "Caption LIKE '%video%')")
    
    cameras = []
    for device in devices:
        cameras.append({
            "Caption": device.Caption,
            "DeviceID": device.DeviceID
        })
    
    logger.info(f"Found {len(cameras)} USB camera devices")
    return cameras

def detect_camera_type(camera_caption: str) -> str:
    """Detect the type of camera based on its caption."""
    if "Basler" in camera_caption:
        return "basler"
    elif any(term in camera_caption for term in ["Camera", "USB", "cam", "Webcam", "video"]):
        return "opencv"
    return "unknown"

def test_opencv_direct() -> List[Dict]:
    """Test direct OpenCV camera detection by attempting to open camera indices"""
    logger.info("Testing direct OpenCV camera detection...")
    cameras = []
    
    # Try the first 10 camera indices (or adjust as needed)
    for i in range(10):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Use DirectShow backend
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    logger.info(f"Found working camera at index {i}")
                    cameras.append({
                        "type": "opencv",
                        "index": i,
                        "caption": f"OpenCV Camera Index {i}"
                    })
            cap.release()
        except Exception as e:
            logger.error(f"Error testing camera {i}: {e}")
    
    return cameras

def get_available_cameras() -> List[Dict]:
    """Detect available cameras, including Basler and OpenCV-compatible cameras."""
    logger.info("Detecting available cameras...")
    available_cameras = []
    
    # Detect Basler cameras
    basler_devices = pylon.TlFactory.GetInstance().EnumerateDevices()
    logger.info(f"Found {len(basler_devices)} Basler cameras")
    
    for device in basler_devices:
        available_cameras.append({
            "type": "basler",
            "device": device,
            "caption": device.GetModelName()
        })
    
    # Try to detect OpenCV cameras via WMI
    usb_devices = get_usb_devices()
    opencv_cameras_found = False
    
    for index, camera in enumerate(usb_devices):
        camera_type = detect_camera_type(camera["Caption"])
        if camera_type == "opencv":
            opencv_cameras_found = True
            available_cameras.append({
                "type": "regular",
                "index": index,
                "caption": camera["Caption"]
            })
    
    # If no OpenCV cameras were found via WMI, try direct detection
    if not opencv_cameras_found:
        logger.info("No cameras found via WMI, trying direct OpenCV detection...")
        direct_cameras = test_opencv_direct()
        available_cameras.extend(direct_cameras)
    
    # Log summary
    for camera in available_cameras:
        logger.info(f"Detected camera: {camera.get('caption', 'Unknown')}, Type: {camera['type']}")
    
    return available_cameras

# Function to be called from main app to diagnose camera issues
def diagnose_camera_issues():
    """Run diagnostic tests and return all detected cameras"""
    logger.info("Running camera diagnostics...")
    cameras = get_available_cameras()
    return cameras