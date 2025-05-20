
# Piece Models
from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel


class PieceImageResponse(BaseModel):
    id: int
    image_name: str
    piece_path: str
    url: str
    timestamp: datetime

    class Config:
        orm_mode = True

class PieceResponse(BaseModel):
    id: int
    piece_label: str
    nbre_img: int
    class_data_id: Optional[int] = None
    created_at: datetime
    images: List[PieceImageResponse] = []

    class Config:
        orm_mode = True