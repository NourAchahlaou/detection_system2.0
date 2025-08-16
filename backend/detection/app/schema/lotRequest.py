from typing import Optional
from dataclasses import dataclass
import time


@dataclass
class DetectionRequest:
    """Enhanced detection request structure"""
    camera_id: int
    target_label: str
    lot_id: Optional[int] = None
    expected_piece_id: Optional[int] = None
    expected_piece_number: Optional[int] = None
    timestamp: float = None
    quality: int = 85
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()



@dataclass
class LotCreationRequest:
    """Request to create a new detection lot"""
    lot_name: str
    expected_piece_id: int
    expected_piece_number: int