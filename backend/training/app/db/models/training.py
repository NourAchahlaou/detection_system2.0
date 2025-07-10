from sqlalchemy import Column, DateTime, Float, Integer, String, ForeignKey, Text, Boolean
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import JSON
from training.app.db.session import Base
from datetime import datetime


class TrainingSession(Base):
    __tablename__ = 'training_session'
    __table_args__ = {"schema": "training"}

    id = Column(Integer, primary_key=True, index=True)
    session_name = Column(String, nullable=False)
    model_type = Column(String, default="YOLO")
    epochs = Column(Integer, default=25)
    batch_size = Column(Integer, default=8)
    learning_rate = Column(Float, default=0.0001)
    image_size = Column(Integer, default=640)
    device_used = Column(String, nullable=True)  # 'cuda' or 'cpu'
    piece_id = Column(Integer, nullable=True)  # Make nullable if not all sessions need a piece
    # Training progress fields
    current_epoch = Column(Integer, default=0)
    progress_percentage = Column(Float, default=0.0)
    is_training = Column(Boolean, default=False)
    
    # Training data counts
    total_images = Column(Integer, default=0)
    augmented_images = Column(Integer, default=0)
    validation_images = Column(Integer, default=0)
    
    # Current losses (JSON field to store loss values)
    current_losses = Column(JSON, default=lambda: {
        "box_loss": 0.0,
        "cls_loss": 0.0,
        "dfl_loss": 0.0,
    })
    
    # Current metrics (JSON field to store metrics)
    current_metrics = Column(JSON, default=lambda: {
        "instances": 0,
        "lr": 0.002,
        "momentum": 0.9,
    })
    
    # Piece labels being trained (JSON array)
    piece_labels = Column(JSON, default=lambda: [])
    
    # Training logs (JSON array of log entries)
    training_logs = Column(JSON, default=lambda: [])
    
    # Timing
    started_at = Column(DateTime(timezone=True), nullable=True)
    completed_at = Column(DateTime(timezone=True), nullable=True)
    last_updated = Column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Results
    final_accuracy = Column(Float, nullable=True)
    final_loss = Column(Float, nullable=True)
    model_path = Column(String, nullable=True)
    
    # Status methods
    def is_active(self):
        """Check if the training session is currently active."""
        return self.is_training and self.completed_at is None
    
    def get_status(self):
        """Get the current status of the training session."""
        if self.completed_at:
            return "completed"
        elif self.is_training:
            return "running"
        else:
            return "stopped"
    
    def add_log(self, level: str, message: str):
        """Add a log entry to the training session."""
        if self.training_logs is None:
            self.training_logs = []
        
        log_entry = {
            "level": level,
            "message": message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        self.training_logs.append(log_entry)
        
        # Keep only the last 100 log entries to prevent memory issues
        if len(self.training_logs) > 100:
            self.training_logs = self.training_logs[-100:]
    
    def update_progress(self, **kwargs):
        """Update training progress with provided fields."""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.last_updated = datetime.utcnow()
    
    def update_losses(self, box_loss=None, cls_loss=None, dfl_loss=None):
        """Update current losses."""
        if self.current_losses is None:
            self.current_losses = {}
        
        if box_loss is not None:
            self.current_losses["box_loss"] = box_loss
        if cls_loss is not None:
            self.current_losses["cls_loss"] = cls_loss
        if dfl_loss is not None:
            self.current_losses["dfl_loss"] = dfl_loss
    
    def update_metrics(self, instances=None, lr=None, momentum=None):
        """Update current metrics."""
        if self.current_metrics is None:
            self.current_metrics = {}
        
        if instances is not None:
            self.current_metrics["instances"] = instances
        if lr is not None:
            self.current_metrics["lr"] = lr
        if momentum is not None:
            self.current_metrics["momentum"] = momentum
    
    def start_training(self, piece_labels: list):
        """Start the training session."""
        self.is_training = True
        self.started_at = datetime.utcnow()
        self.piece_labels = piece_labels
        self.current_epoch = 0
        self.progress_percentage = 0.0
        self.add_log("INFO", f"Training started for pieces: {', '.join(piece_labels)}")
    
    def complete_training(self, final_accuracy=None, final_loss=None):
        """Complete the training session."""
        self.is_training = False
        self.completed_at = datetime.utcnow()
        self.progress_percentage = 100.0
        
        if final_accuracy is not None:
            self.final_accuracy = final_accuracy
        if final_loss is not None:
            self.final_loss = final_loss
        
        self.add_log("SUCCESS", "Training completed successfully")
    
    def stop_training(self):
        """Stop the training session."""
        self.is_training = False
        self.completed_at = datetime.utcnow()
        self.add_log("INFO", "Training stopped by user request")
    
    def fail_training(self, error_message: str):
        """Mark the training session as failed."""
        self.is_training = False
        self.completed_at = datetime.utcnow()
        self.add_log("ERROR", f"Training failed: {error_message}")
    
    def get_recent_logs(self, limit: int = 50):
        """Get recent training logs."""
        if not self.training_logs:
            return []
        
        return self.training_logs[-limit:] if len(self.training_logs) > limit else self.training_logs
    
    def to_dict(self):
        """Convert training session to dictionary for API responses."""
        return {
            "id": self.id,
            "session_name": self.session_name,
            "model_type": self.model_type,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate,
            "image_size": self.image_size,
            "device_used": self.device_used,
            "piece_id": self.piece_id,
            "current_epoch": self.current_epoch,
            "progress_percentage": self.progress_percentage,
            "is_training": self.is_training,
            "total_images": self.total_images,
            "augmented_images": self.augmented_images,
            "validation_images": self.validation_images,
            "current_losses": self.current_losses,
            "current_metrics": self.current_metrics,
            "piece_labels": self.piece_labels,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "last_updated": self.last_updated.isoformat() if self.last_updated else None,
            "final_accuracy": self.final_accuracy,
            "final_loss": self.final_loss,
            "model_path": self.model_path,
            "status": self.get_status()
        }