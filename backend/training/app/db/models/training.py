from sqlalchemy import Column, DateTime, Float, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from training.app.db.session import Base

class TrainingSession(Base):
    __tablename__ = 'training_session'
    __table_args__ = {"schema": "training"}

    id = Column(Integer, primary_key=True, index=True)
    session_name = Column(String, nullable=False)
    model_type = Column(String, default="YOLOV8X")
    epochs = Column(Integer, default=25)
    batch_size = Column(Integer, default=8)
    learning_rate = Column(Float, default=0.0001)
    image_size = Column(Integer, default=640)
    device_used = Column(String, nullable=True)  # 'cuda' or 'cpu'
    piece_id = Column(Integer, nullable=True)  # Make nullable if not all sessions need a piece
    
    # Timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    
    # Results
    final_accuracy = Column(Float, nullable=True)
    final_loss = Column(Float, nullable=True)
    model_path = Column(String, nullable=True)
    
