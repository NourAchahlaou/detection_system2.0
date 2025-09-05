from pydantic import BaseModel
from typing import List, Optional
from datetime import datetime

class DetectedPieceResponse(BaseModel):
    id: int
    detected_label: str
    confidence_score: float
    bounding_box_x1: int
    bounding_box_y1: int
    bounding_box_x2: int
    bounding_box_y2: int
    is_correct_piece: bool
    piece_id: Optional[int] = None
    created_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class DetectionSessionWithPieces(BaseModel):
    session_id: int
    lot_id: int
    correct_pieces_count: int
    misplaced_pieces_count: int
    total_pieces_detected: int
    confidence_score: float
    is_target_match: bool
    detection_rate: float
    created_at: Optional[datetime] = None
    detected_pieces: List[DetectedPieceResponse] = []

    class Config:
        from_attributes = True