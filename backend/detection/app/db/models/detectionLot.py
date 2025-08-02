
from sqlalchemy import Column, DateTime, Float, Integer, Null, String,Boolean, func
from sqlalchemy.orm import relationship
from detection.app.db.session import Base

class DetectionLot(Base):
    __tablename__ = 'detection_lot'
    __table_args__ = {"schema": "detection"}
    
    id = Column(Integer, primary_key=True, index=True)
    lot_name = Column(String, nullable=False)
    expected_piece_id = Column(Integer, nullable=False)
    expected_piece_number = Column(Integer, nullable=False)  
    is_target_match = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Relationships
    detection_sessions = relationship("DetectionSession", back_populates="lot", cascade="all, delete-orphan")
