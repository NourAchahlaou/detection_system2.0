from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from camera_management.app.db.schemas.camera import CameraBase

class CameraResponse(BaseModel):
    """Pydantic model for camera response"""
    id: int
    camera_index: int
    model: str
    settings_id: Optional[int] = None
    
    class Config:
        from_attributes = True  #
class CameraIndexResponse(BaseModel):
    camera_index: Any

class CameraStartResponse(BaseModel):
    message: str

class CameraStopResponse(BaseModel):
    message: str
class CameraStatusResponse(BaseModel):
    camera_opened: bool

class CameraResponse(CameraBase):
    id: int
    settings_id: int

    class Config:
        orm_mode = True

class CameraWithSettingsResponse(CameraResponse):
    settings: List[Dict[str, Any]]

    class Config:
        orm_mode = True
