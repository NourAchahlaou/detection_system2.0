from pydantic import BaseModel





class CameraStopResponse(BaseModel):
    message: str
class CameraStatusResponse(BaseModel):
    camera_opened: bool

