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
    for i, camera in enumerate(cameras):
        logger.info(f"  Device {i}: {camera['Caption']}")
    
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
                        "caption": f"Camera at Index {i}"
                    })
                else:
                    logger.info(f"Camera at index {i} opened but cannot read frames")
            cap.release()
        except Exception as e:
            logger.debug(f"No camera at index {i}: {e}")
    
    return cameras

def match_wmi_to_opencv_cameras(wmi_cameras: List[Dict], opencv_cameras: List[Dict]) -> List[Dict]:
    """
    Match WMI camera information with working OpenCV camera indices.
    This is a best-effort matching since there's no direct way to correlate them.
    """
    logger.info("Matching WMI camera info with OpenCV indices...")
    
    # Filter out non-camera devices from WMI results
    camera_devices = [cam for cam in wmi_cameras if detect_camera_type(cam["Caption"]) == "opencv"]
    
    matched_cameras = []
    
    # If we have the same number of cameras, try to match them
    if len(camera_devices) == len(opencv_cameras):
        for i, opencv_cam in enumerate(opencv_cameras):
            if i < len(camera_devices):
                matched_cameras.append({
                    "type": "regular",
                    "index": opencv_cam["index"],
                    "caption": camera_devices[i]["Caption"]
                })
            else:
                matched_cameras.append({
                    "type": "regular",
                    "index": opencv_cam["index"],
                    "caption": f"Camera at Index {opencv_cam['index']}"
                })
    else:
        # If counts don't match, use OpenCV detection with generic names
        logger.warning(f"Mismatch: {len(camera_devices)} WMI cameras vs {len(opencv_cameras)} working OpenCV cameras")
        for opencv_cam in opencv_cameras:
            # Try to find a reasonable name from WMI if available
            caption = f"Camera at Index {opencv_cam['index']}"
            if camera_devices:
                # Use the first available WMI camera name as a fallback
                caption = camera_devices[0]["Caption"]
                camera_devices.pop(0)  # Remove it so we don't reuse it
            
            matched_cameras.append({
                "type": "regular",
                "index": opencv_cam["index"],
                "caption": caption
            })
    
    return matched_cameras

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
    
    # Get WMI camera info
    wmi_cameras = get_usb_devices()
    
    # Get actually working OpenCV cameras
    opencv_cameras = test_opencv_direct()
    
    if opencv_cameras:
        # Match WMI info with working OpenCV cameras
        matched_cameras = match_wmi_to_opencv_cameras(wmi_cameras, opencv_cameras)
        available_cameras.extend(matched_cameras)
        
        logger.info(f"Successfully matched {len(matched_cameras)} OpenCV cameras")
    else:
        logger.warning("No working OpenCV cameras found")
    
    # Log summary
    logger.info(f"Total cameras detected: {len(available_cameras)}")
    for i, camera in enumerate(available_cameras):
        logger.info(f"  Camera {i+1}: {camera.get('caption', 'Unknown')} (Type: {camera['type']}, Index: {camera.get('index', 'N/A')})")
    
    return available_cameras

# Function to be called from main app to diagnose camera issues
def diagnose_camera_issues():
    """Run diagnostic tests and return all detected cameras"""
    logger.info("Running camera diagnostics...")
    
    # Show what WMI finds
    wmi_cameras = get_usb_devices()
    logger.info("WMI Camera Detection Results:")
    for i, cam in enumerate(wmi_cameras):
        logger.info(f"  WMI Device {i}: {cam['Caption']}")
    
    # Show what OpenCV finds
    opencv_cameras = test_opencv_direct()
    logger.info("OpenCV Direct Detection Results:")
    for cam in opencv_cameras:
        logger.info(f"  OpenCV Index {cam['index']}: Working")
    
    # Show final result
    cameras = get_available_cameras()
    return cameras