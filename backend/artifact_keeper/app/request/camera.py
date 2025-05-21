from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# Camera Request Models
class OpenCVCameraRequest(BaseModel):
    camera_index: int = Field(..., description="The index of the OpenCV camera")

class BaslerCameraRequest(BaseModel):
    serial_number: str = Field(..., description="The serial number of the Basler camera")

class CameraIDRequest(BaseModel):
    camera_id: int = Field(..., description="The database ID of the camera")




# Camera Settings Update Model
class UpdateCameraSettings(BaseModel):
    exposure: Optional[float] = None
    contrast: Optional[float] = None
    brightness: Optional[float] = None
    focus: Optional[float] = None
    aperture: Optional[float] = None
    gain: Optional[float] = None
    white_balance: Optional[float] = None


