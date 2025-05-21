from typing import Optional
from pydantic import BaseModel


class CameraResponse(BaseModel):
    type: str
    caption: str
    index: Optional[int] = None
    device: Optional[dict] = None
    settings : Optional[dict] = None


class CameraStopResponse(BaseModel):
    message: str
class CameraStatusResponse(BaseModel):
    camera_opened: bool

