from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Boolean, func
from sqlalchemy.orm import relationship
from detection.app.db.session import Base

class DetectedPiece(Base):
    __tablename__ = 'detected_piece'
    __table_args__ = {"schema": "detection"}
    
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(Integer, ForeignKey('detection.detection_session.id'), nullable=False)
    piece_id = Column(Integer, nullable=True)  # Reference to actual piece if known/matched
    detected_label = Column(String, nullable=False)  # The label that was detected
    confidence_score = Column(Float, nullable=False)
    bounding_box_x1 = Column(Integer, nullable=False)
    bounding_box_y1 = Column(Integer, nullable=False)
    bounding_box_x2 = Column(Integer, nullable=False)
    bounding_box_y2 = Column(Integer, nullable=False)
    is_correct_piece = Column(Boolean, default=False)  # True if matches expected piece
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    # Relationships
    session = relationship("DetectionSession", back_populates="detected_pieces")
    # Note: piece relationship would need to be handled carefully due to cross-schema reference