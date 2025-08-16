from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from datetime import datetime
import json


from detection.app.db.models.detectionLot import DetectionLot


@dataclass
class LotResponse:
    """Response after creating or updating a lot with proper date serialization"""
    lot_id: int
    lot_name: str
    expected_piece_id: int
    expected_piece_number: int
    is_target_match: bool
    created_at: str  # ISO format string instead of datetime
    completed_at: Optional[str] = None  # ISO format string instead of datetime
    total_sessions: int = 0
    successful_detections: int = 0

    @classmethod
    def from_db_model(cls, lot: DetectionLot, total_sessions: int = 0, successful_detections: int = 0):
        """Create LotResponse from database model with proper date conversion"""
        return cls(
            lot_id=lot.id,
            lot_name=lot.lot_name,
            expected_piece_id=lot.expected_piece_id,
            expected_piece_number=lot.expected_piece_number,
            is_target_match=lot.is_target_match,
            created_at=lot.created_at.isoformat() if lot.created_at else datetime.utcnow().isoformat(),
            completed_at=lot.completed_at.isoformat() if lot.completed_at else None,
            total_sessions=total_sessions,
            successful_detections=successful_detections
        )
    
@dataclass
class DetectionResponse:
    """Enhanced detection response structure"""
    camera_id: int
    target_label: str
    detected_target: bool
    non_target_count: int
    processing_time_ms: float
    confidence: Optional[float]
    frame_with_overlay: str
    timestamp: float
    stream_frozen: bool
    lot_id: Optional[int] = None
    session_id: Optional[int] = None
    is_target_match: bool = False
    detection_rate: float = 0.0
    # Enhanced validation fields
    lot_validation_result: Dict[str, Any] = None
    validation_errors: List[str] = None

@dataclass
class LotValidationResult:
    """Result of lot validation against detection results"""
    is_valid: bool
    expected_count: int
    actual_correct_count: int
    actual_incorrect_count: int
    expected_label: str
    detected_labels: List[str]
    errors: List[str]
    confidence_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'is_valid': self.is_valid,
            'expected_count': self.expected_count,
            'actual_correct_count': self.actual_correct_count,
            'actual_incorrect_count': self.actual_incorrect_count,
            'expected_label': self.expected_label,
            'detected_labels': self.detected_labels,
            'errors': self.errors,
            'confidence_score': self.confidence_score
        }
