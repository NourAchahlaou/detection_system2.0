from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class CameraClientResponse(BaseModel):
    id: int
    type: str
    identifier: str  # Either camera_index or serial_number
    model: str

class CameraStatusResponse(BaseModel):
    camera_opened: bool
    circuit_breaker_active: Optional[bool] = None
# Utility response models
class CleanupResponse(BaseModel):
    message: str


class CameraResponse(BaseModel):
    id: int
    camera_type: str
    camera_index: Optional[int] = None
    serial_number: Optional[str] = None
    model: str
    status: bool

    class Config:
        orm_mode = True

class CameraSettingsResponse(BaseModel):
    id: int
    exposure: Optional[float] = None
    contrast: Optional[float] = None
    brightness: Optional[float] = None
    focus: Optional[float] = None
    aperture: Optional[float] = None
    gain: Optional[float] = None
    white_balance: Optional[float] = None

    class Config:
        orm_mode = True

class CameraWithSettingsResponse(CameraResponse):
    settings: CameraSettingsResponse

    class Config:
        orm_mode = True

class CameraStatusResponse(BaseModel):
    camera_opened: bool
    circuit_breaker_active: Optional[bool] = None

class CameraStopResponse(BaseModel):
    message: str


class CircuitBreakerStatusResponse(BaseModel):
    state: str
    failure_count: int
    last_failure_time: Optional[datetime] = None