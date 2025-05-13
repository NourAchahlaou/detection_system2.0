from typing import Dict, List
from pypylon import pylon
import win32com.client

def get_usb_devices() -> List[Dict[str, str]]:
    wmi = win32com.client.GetObject("winmgmts:")
    devices = wmi.ExecQuery("SELECT * FROM Win32_PnPEntity WHERE Caption LIKE '%Camera%'")

    cameras = []
    for device in devices:
        cameras.append({
            "Caption": device.Caption,
            "DeviceID": device.DeviceID
        })

    return cameras

def detect_camera_type(camera_caption: str) -> str:
    """Detect the type of camera based on its caption."""
    if "Basler" in camera_caption:
        return "basler"
    elif "Camera" in camera_caption or "USB" in camera_caption:
        return "opencv"
    return "unknown"

def get_available_cameras() -> List[Dict]:
    """Detect available cameras, including Basler and OpenCV-compatible cameras."""
    usb_devices = get_usb_devices()
    available_cameras = []

    # Detect Basler cameras
    basler_devices = pylon.TlFactory.GetInstance().EnumerateDevices()
    for device in basler_devices:
        available_cameras.append({
            "type": "basler",
            "device": device,
            "caption": device.GetModelName()
        })

    # Detect OpenCV-compatible cameras
    for index, camera in enumerate(usb_devices):
        camera_type = detect_camera_type(camera["Caption"])
        if camera_type == "opencv":
            available_cameras.append({
                "type": "opencv",
                "index": index,
                "caption": camera["Caption"]
            })

    return available_cameras