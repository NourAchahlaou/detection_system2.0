from typing import Optional
from pydantic import BaseModel
class UpdateCameraSettings(BaseModel):

    exposure: Optional[float] = None
    contrast: Optional[float] = None
    brightness: Optional[float] = None
    focus: Optional[float] = None
    aperture: Optional[float] = None
    gain: Optional[float] = None
    white_balance: Optional[str] = None

    class Config:
        orm_mode = True
class CameraSettingsBase(BaseModel):
    exposure: Optional[float] = None
    contrast: Optional[float] = None
    brightness: Optional[float] = None
    focus: Optional[float] = None
    aperture: Optional[float] = None
    gain: Optional[float] = None
    white_balance: Optional[str] = None

class CameraSettingsCreate(CameraSettingsBase):
    pass

class CameraSettingsResponse(CameraSettingsBase):
    id: int

    class Config:
        orm_mode = True 