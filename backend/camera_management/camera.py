from typing import Dict, List
import logging
from pypylon import pylon
import win32com.client
import cv2
import sys
import traceback

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_usb_devices() -> List[Dict[str, str]]:
    """Use WMI to find camera devices with more comprehensive detection"""
    logger.info("Searching for USB camera devices with WMI...")
    try:
        wmi = win32com.client.GetObject("winmgmts:")
        
        # Use a broader query to catch more potential cameras
        query = "SELECT * FROM Win32_PnPEntity WHERE " + \
                "(Caption LIKE '%Camera%' OR " + \
                "Caption LIKE '%cam%' OR " + \
                "Caption LIKE '%Webcam%' OR " + \
                "Caption LIKE '%video%' OR " + \
                "PNPClass = 'Image' OR " + \
                "PNPClass = 'Camera')"
        
        logger.info(f"Executing WMI query: {query}")
        devices = wmi.ExecQuery(query)
        
        logger.info(f"Found {len(devices)} potential camera devices")
        cameras = []
        for device in devices:
            logger.info(f"Found device: {device.Caption}, ID: {device.DeviceID}")
            cameras.append({
                "Caption": device.Caption,
                "DeviceID": device.DeviceID
            })
        
        return cameras
    except Exception as e:
        logger.error(f"Error in get_usb_devices: {str(e)}")
        logger.error(traceback.format_exc())
        return []

def test_opencv_direct() -> List[Dict[str, str]]:
    """Test direct OpenCV camera detection by attempting to open camera indices"""
    logger.info("Testing direct OpenCV camera detection...")
    cameras = []
    
    # Try the first 10 camera indices
    for i in range(10):
        try:
            logger.info(f"Attempting to open camera at index {i}")
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # Try DirectShow as well
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    logger.info(f"Successfully opened camera at index {i}")
                    cameras.append({
                        "type": "opencv",
                        "index": i,
                        "caption": f"OpenCV Camera Index {i}"
                    })
                else:
                    logger.warning(f"Camera at index {i} opened but couldn't read frame")
            cap.release()
        except Exception as e:
            logger.error(f"Error testing camera {i}: {str(e)}")
    
    return cameras

def get_basler_cameras() -> List[Dict]:
    """Specifically detect Basler cameras with error handling"""
    logger.info("Detecting Basler cameras...")
    cameras = []
    
    try:
        basler_devices = pylon.TlFactory.GetInstance().EnumerateDevices()
        logger.info(f"Found {len(basler_devices)} Basler devices")
        
        for device in basler_devices:
            try:
                model = device.GetModelName()
                serial = device.GetSerialNumber()
                logger.info(f"Found Basler camera: {model}, S/N: {serial}")
                cameras.append({
                    "type": "basler",
                    "device": device,
                    "caption": model,
                    "serial": serial
                })
            except Exception as e:
                logger.error(f"Error processing Basler device: {str(e)}")
    except Exception as e:
        logger.error(f"Error enumerating Basler devices: {str(e)}")
        logger.error(traceback.format_exc())
    
    return cameras

def detect_camera_type(camera_caption: str) -> str:
    """Detect the type of camera based on its caption."""
    if "Basler" in camera_caption:
        return "basler"
    elif any(term in camera_caption for term in ["Camera", "USB", "cam", "Webcam", "video"]):
        return "opencv"
    return "unknown"

def get_available_cameras() -> List[Dict]:
    """Detect available cameras with comprehensive detection and logging"""
    logger.info("Detecting all available cameras...")
    available_cameras = []
    
    # Get all USB devices that might be cameras
    usb_devices = get_usb_devices()
    logger.info(f"Found {len(usb_devices)} potential USB camera devices")
    
    # Detect Basler cameras first
    basler_cameras = get_basler_cameras()
    available_cameras.extend(basler_cameras)
    
    # Process USB devices that might be OpenCV-compatible cameras
    for camera in usb_devices:
        camera_type = detect_camera_type(camera["Caption"])
        logger.info(f"USB device: {camera['Caption']} detected as: {camera_type}")
        
        if camera_type == "opencv":
            available_cameras.append({
                "type": "opencv",
                "device_id": camera["DeviceID"],
                "caption": camera["Caption"]
            })
    
    # Try direct OpenCV detection as a fallback
    if not any(cam["type"] == "opencv" for cam in available_cameras):
        logger.info("No OpenCV cameras found via WMI, trying direct detection...")
        opencv_cameras = test_opencv_direct()
        available_cameras.extend(opencv_cameras)
    
    logger.info(f"Total cameras detected: {len(available_cameras)}")
    for i, camera in enumerate(available_cameras):
        logger.info(f"Camera {i+1}: Type: {camera['type']}, Caption: {camera.get('caption', 'Unknown')}")
    
    return available_cameras

# Diagnostic function to list all connected devices
def list_all_connected_devices():
    """List all connected devices to help diagnose camera detection issues"""
    logger.info("Listing all connected devices...")
    try:
        wmi = win32com.client.GetObject("winmgmts:")
        all_devices = wmi.ExecQuery("SELECT * FROM Win32_PnPEntity")
        
        logger.info(f"Found {len(all_devices)} total devices")
        potential_cameras = []
        
        for device in all_devices:
            try:
                caption = device.Caption
                device_id = device.DeviceID
                pnp_class = getattr(device, "PNPClass", "Unknown")
                
                # Check if this might be a camera
                camera_terms = ["camera", "cam", "webcam", "video", "image"]
                if (pnp_class in ["Camera", "Image"] or 
                    any(term.lower() in caption.lower() for term in camera_terms)):
                    potential_cameras.append({
                        "Caption": caption,
                        "DeviceID": device_id,
                        "PNPClass": pnp_class
                    })
                    logger.info(f"Potential camera: {caption}, Class: {pnp_class}")
            except Exception as e:
                logger.error(f"Error processing device: {str(e)}")
        
        logger.info(f"Found {len(potential_cameras)} potential camera devices")
        return potential_cameras
    
    except Exception as e:
        logger.error(f"Error listing devices: {str(e)}")
        logger.error(traceback.format_exc())
        return []

# Main function to run diagnostic tests
def diagnose_camera_issues():
    """Run diagnostic tests to identify camera detection issues"""
    logger.info("Starting camera detection diagnostics...")
    
    # Step 1: List all connected devices
    logger.info("Step 1: Listing all connected devices")
    list_all_connected_devices()
    
    # Step 2: Test direct OpenCV camera access
    logger.info("Step 2: Testing direct OpenCV camera access")
    opencv_cameras = test_opencv_direct()
    logger.info(f"OpenCV direct detection found {len(opencv_cameras)} cameras")
    
    # Step 3: Test Basler camera detection
    logger.info("Step 3: Testing Basler camera detection")
    basler_cameras = get_basler_cameras()
    logger.info(f"Basler detection found {len(basler_cameras)} cameras")
    
    # Step 4: Test combined detection
    logger.info("Step 4: Testing combined camera detection")
    all_cameras = get_available_cameras()
    logger.info(f"Combined detection found {len(all_cameras)} cameras")
    
    logger.info("Camera detection diagnostics completed")
    return all_cameras

if __name__ == "__main__":
    logger.info("Running camera detection diagnostic script")
    cameras = diagnose_camera_issues()
    print("\nDetected Cameras Summary:")
    for i, camera in enumerate(cameras):
        print(f"{i+1}. Type: {camera['type']}, Caption: {camera.get('caption', 'Unknown')}")
    
    if not cameras:
        print("\nNo cameras detected. Possible issues:")
        print("1. Camera drivers not installed or not functioning")
        print("2. Camera is disabled in Device Manager or system settings")
        print("3. Camera is being used by another application")
        print("4. Privacy settings are blocking camera access")