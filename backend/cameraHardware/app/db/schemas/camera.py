from pydantic import BaseModel

# Base schemas for Camera START
class OpenCVCameraRequest(BaseModel):
    camera_index: int

class BaslerCameraRequest(BaseModel):
    serial_number: str
