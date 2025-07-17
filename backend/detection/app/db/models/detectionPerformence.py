
from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, Null, String,Boolean, func
from sqlalchemy.orm import relationship
from detection.app.db.session import Base

class DetectionPerformance(Base):
    __tablename__ = 'detection_performance'
    __table_args__ = {"schema": "detection"}
    
    id = Column(Integer, primary_key=True, index=True)
    lot_id = Column(Integer, ForeignKey('detection.detection_lot.id'), nullable=False)
    total_processing_time = Column(Float, default=0.0)  # Total time in seconds
    average_fps = Column(Float, default=0.0)  # Frames per second
    gpu_utilization = Column(Float, default=0.0)  # GPU usage percentage
    device_used = Column(String, default='cpu')  # 'cpu' or 'cuda'
    model_precision = Column(String, default='fp32')  # 'fp16' or 'fp32'
    
    # Relationships
    lot = relationship("DetectionLot")