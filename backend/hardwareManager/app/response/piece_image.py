from pydantic import BaseModel
from db.schemas.piece_image import PieceImageBase

class CleanupResponse(BaseModel):
    message: str

class SaveImagesResponse(BaseModel):
    message: str

class PieceImageResponse(PieceImageBase):
    id: int
    timestamp: str

    class Config:
        orm_mode = True