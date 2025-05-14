from pydantic import BaseModel
from typing import Optional

# Base schemas for Camera
class CameraBase(BaseModel):
    camera_type: str
    camera_index: Optional[int] = None
    serial_number: Optional[str] = None
    model: str
    status: bool = False

class CameraCreate(CameraBase):
    settings_id: Optional[int] = None


# Request/Response schemas for routes
class CameraID(BaseModel):
    camera_id: int


class CameraListItem(BaseModel):
    camera_index: Optional[int]
    model: str
    camera_id: int


