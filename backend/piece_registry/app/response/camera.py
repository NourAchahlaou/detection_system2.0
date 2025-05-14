from pydantic import BaseModel
from typing import List, Optional, Dict, Any
from backend.piece_registry.app.db.schemas.camera import CameraBase

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
