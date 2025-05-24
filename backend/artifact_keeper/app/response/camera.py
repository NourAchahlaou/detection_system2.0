from datetime import datetime
from typing import Optional
from pydantic import BaseModel, ConfigDict


class CameraClientResponse(BaseModel):
    type: str  # 'regular' or 'basler'
    caption: str  # Use 'caption' as the model name
    index: Optional[int] = None  # For OpenCV cameras
    serial_number: Optional[str] = None  # For Basler cameras
    
class CameraStatusResponse(BaseModel):
    camera_opened: bool
    circuit_breaker_active: Optional[bool] = None
# Utility response models
class CleanupResponse(BaseModel):
    message: str

class CameraSettingsResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)  # This replaces orm_mode = True
    
    id: int
    exposure: Optional[float] = None
    contrast: Optional[float] = None
    brightness: Optional[float] = None
    focus: Optional[float] = None
    aperture: Optional[float] = None
    gain: Optional[float] = None  # Changed from float to match your DB model
    white_balance: Optional[str] = None  # Changed from float to str

class CameraResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    camera_type: str
    camera_index: Optional[int] = None
    serial_number: Optional[str] = None
    model: str
    status: bool
    settings: Optional[CameraSettingsResponse] = None 

class CameraStatusResponse(BaseModel):
    camera_opened: bool
    circuit_breaker_active: Optional[bool] = None

class CameraStopResponse(BaseModel):
    message: str


class CircuitBreakerStatusResponse(BaseModel):
    state: str
    failure_count: int
    last_failure_time: Optional[datetime] = None