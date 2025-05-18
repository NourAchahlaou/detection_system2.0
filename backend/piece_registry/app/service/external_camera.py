from typing import Dict, List, Optional
import os
import subprocess
import cv2
import logging
from pypylon import pylon

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("camera_detection")

def list_usb_devices_linux() -> List[Dict[str, str]]:
    """Get information about USB devices on Linux."""
    try:
        # Use lsusb to get information about connected USB devices
        result = subprocess.run(['lsusb'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.error(f"Error running lsusb: {result.stderr}")
            return []
        
        devices = []
        for line in result.stdout.strip().split('\n'):
            line_lower = line.lower()
            # Look for camera-related keywords
            if any(keyword in line_lower for keyword in ['camera', 'webcam', 'basler', 'imaging', 'video']):
                # Parse the lsusb output line
                parts = line.split()
                if len(parts) >= 6:
                    bus_id = parts[1]
                    device_id = parts[3].rstrip(':')
                    device_name = ' '.join(parts[6:])
                    devices.append({
                        "Caption": device_name,
                        "DeviceID": f"USB\\{bus_id}\\{device_id}",
                        "BusID": bus_id,
                        "DeviceNumber": device_id
                    })
        
        logger.info(f"Detected {len(devices)} USB camera devices")
        for device in devices:
            logger.info(f"USB device: {device['Caption']} - {device['DeviceID']}")
        return devices
    except Exception as e:
        logger.error(f"Error listing USB devices: {e}")
        return []

def get_v4l2_device_info() -> List[Dict[str, str]]:
    """Get detailed information about video devices using v4l2-ctl."""
    devices = []
    try:
        # List all video devices
        result = subprocess.run(['ls', '-la', '/dev/video*'], capture_output=True, text=True)
        if result.returncode != 0:
            logger.warning(f"Error listing video devices: {result.stderr}")
            return devices
            
        video_paths = [line.split()[-1] for line in result.stdout.strip().split('\n') 
                      if 'video' in line and not 'by-path' in line]
        
        for video_path in video_paths:
            try:
                # Get device info using v4l2-ctl
                info_cmd = ['v4l2-ctl', '--device', video_path, '--info']
                info_result = subprocess.run(info_cmd, capture_output=True, text=True)
                
                if info_result.returncode == 0:
                    info_output = info_result.stdout
                    
                    # Extract device name
                    device_name = "Unknown Camera"
                    for line in info_output.split('\n'):
                        if 'Card type' in line:
                            device_name = line.split(':')[-1].strip()
                    
                    # Extract device index
                    device_index = int(video_path.replace('/dev/video', ''))
                    
                    devices.append({
                        "Caption": device_name,
                        "DeviceID": video_path,
                        "DeviceIndex": device_index
                    })
            except Exception as e:
                logger.warning(f"Error getting info for {video_path}: {e}")
                
        logger.info(f"Found {len(devices)} V4L2 devices")
        for device in devices:
            logger.info(f"V4L2 device: {device['Caption']} - {device['DeviceID']}")
        return devices
    except Exception as e:
        logger.error(f"Error in V4L2 device discovery: {e}")
        return []

def detect_camera_type(camera_caption: str) -> str:
    """Detect the type of camera based on its caption."""
    camera_caption = camera_caption.lower()
    if "basler" in camera_caption:
        return "basler"
    elif any(keyword in camera_caption for keyword in ["camera", "webcam", "capture", "video", "imaging"]):
        return "opencv"
    return "unknown"

def get_opencv_camera_indices() -> List[int]:
    """Find available OpenCV camera indices."""
    indices = []
    # Try more indices for thoroughness
    for i in range(20):  # Test up to 20 indices
        try:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                # Get basic camera info
                width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
                height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
                logger.info(f"OpenCV camera {i} opened: {width}x{height}")
                indices.append(i)
            cap.release()
        except Exception as e:
            logger.warning(f"Error checking OpenCV camera index {i}: {e}")
    
    logger.info(f"Found {len(indices)} available OpenCV camera indices: {indices}")
    return indices

def find_basler_cameras() -> List[Dict]:
    """Specifically find Basler cameras using pypylon."""
    basler_cameras = []
    try:
        # Set environment variables if needed
        if 'PYLON_ROOT' in os.environ:
            logger.info(f"Using PYLON_ROOT: {os.environ['PYLON_ROOT']}")
        
        # Initialize Pylon - Fixed call to match correct API
        # pylon.TlFactory.Initialize()  # This line was incorrect
        tl_factory = pylon.TlFactory.GetInstance()
        
        # Enumerate devices
        devices = tl_factory.EnumerateDevices()
        logger.info(f"Found {len(devices)} Basler devices")
        
        for idx, device in enumerate(devices):
            try:
                model = device.GetModelName() if hasattr(device, "GetModelName") else "Unknown Model"
                serial = device.GetSerialNumber() if hasattr(device, "GetSerialNumber") else f"Basler{idx}"
                
                camera_info = {
                    "type": "basler",
                    "device": device,
                    "index": idx,
                    "caption": f"Basler {model} ({serial})"
                }
                logger.info(f"Basler camera: {camera_info['caption']}")
                basler_cameras.append(camera_info)
            except Exception as e:
                logger.error(f"Error getting info for Basler device {idx}: {e}")
        
    except Exception as e:
        logger.error(f"Error detecting Basler cameras: {e}", exc_info=True)
    
    return basler_cameras

def get_available_cameras() -> List[Dict]:
    """Detect available cameras, including Basler and OpenCV-compatible cameras."""
    available_cameras = []
    
    # Step 1: Find Basler cameras first
    basler_cameras = find_basler_cameras()
    available_cameras.extend(basler_cameras)
    
    # Step 2: Get USB devices using Linux tools
    usb_devices = list_usb_devices_linux()
    
    # Step 3: Get detailed V4L2 device info
    v4l2_devices = get_v4l2_device_info()
    
    # Step 4: Find OpenCV cameras
    opencv_indices = get_opencv_camera_indices()
    
    # Create a map from device index to v4l2 info
    v4l2_map = {device["DeviceIndex"]: device for device in v4l2_devices}
    
    # Add OpenCV cameras that aren't Basler
    for index in opencv_indices:
        # Skip if this appears to be a Basler camera already detected
        if any(camera["type"] == "basler" and getattr(camera.get("index", None), "value", -1) == index 
               for camera in available_cameras):
            logger.info(f"Skipping OpenCV index {index} as it appears to be a Basler camera")
            continue
        
        # Try to get camera properties and match with v4l2 devices
        caption = f"Camera {index}"
        device_id = f"/dev/video{index}"
        
        # Use v4l2 info if available
        if index in v4l2_map:
            v4l2_info = v4l2_map[index]
            caption = v4l2_info["Caption"]
            device_id = v4l2_info["DeviceID"]
        
        # Create camera entry
        camera_info = {
            "type": "opencv",
            "index": index,
            "caption": caption,
            "device_id": device_id
        }
        
        available_cameras.append(camera_info)
    
    # Filter out any duplicates based on caption and type
    unique_cameras = []
    seen_captions = set()
    
    for camera in available_cameras:
        unique_key = f"{camera['type']}_{camera.get('caption', '')}"
        if unique_key not in seen_captions:
            seen_captions.add(unique_key)
            unique_cameras.append(camera)
    
    logger.info(f"Final camera count: {len(unique_cameras)}")
    for camera in unique_cameras:
        logger.info(f"Available camera: {camera['type']} - {camera.get('caption', 'Unnamed')}")
    
    return unique_cameras

# Testing function
def print_camera_info():
    logger.info("Detecting cameras...")
    cameras = get_available_cameras()
    logger.info(f"Found {len(cameras)} total cameras:")
    
    for i, camera in enumerate(cameras):
        logger.info(f"Camera {i+1}:")
        logger.info(f"  Type: {camera['type']}")
        if camera['type'] == 'opencv':
            logger.info(f"  Index: {camera['index']}")
        logger.info(f"  Caption: {camera.get('caption', 'Unnamed')}")
        logger.info("")

if __name__ == "__main__":
    print_camera_info()