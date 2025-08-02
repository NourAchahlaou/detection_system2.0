
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, Null, String,Boolean, func
from sqlalchemy.orm import relationship
from detection.app.db.session import Base

class DetectionSession(Base):
    __tablename__ = 'detection_session'
    __table_args__ = {"schema": "detection"}
    
    id = Column(Integer, primary_key=True, index=True)
    lot_id = Column(Integer, ForeignKey('detection.detection_lot.id'), nullable=False)
    correct_pieces_count = Column(Integer, default=0)
    misplaced_pieces_count = Column(Integer, default=0)
    total_pieces_detected = Column(Integer, default=0)
    confidence_score = Column(Float, nullable=False)
    is_target_match = Column(Boolean, default=False)  # True if detected piece matches expected piece
    detection_rate = Column(Float, default=0.0)  # Detection success rate for this piece
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    lot = relationship("DetectionLot", back_populates="detection_sessions")
