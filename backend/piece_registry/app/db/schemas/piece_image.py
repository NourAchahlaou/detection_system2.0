from pydantic import BaseModel


# Schemas for image capture operations
class CaptureImageRequest(BaseModel):
    piece_label: str

class PieceImageBase(BaseModel):
    piece_id: int
    image_name: str
    piece_path: str
    url: str

