
# Piece Models
from datetime import datetime
from typing import Dict, List, Optional
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
    nbre_img: Optional[int] = None
    images: List[PieceImageResponse] = []

    class Config:
        orm_mode = True

class SaveImagesResponse(BaseModel):
    message: str
    piece_label: str
    images_saved: List[str]  # List of image names
    total_images: int        # Integer count
    cleanup_status: Dict[str, str]  # Dictionary with cleanup info        
