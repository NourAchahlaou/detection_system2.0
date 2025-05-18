from datetime import datetime
from sqlalchemy import  Column, DateTime, Integer, String, Text, func, ForeignKey
from user_management.app.db.session import Base
from sqlalchemy.orm import  relationship

class Task(Base):
    __tablename__ = "tasks"
    __table_args__ = {"schema": "user_mgmt"}    

    id = Column(Integer, primary_key=True)
    title = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    assigned_user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    status = Column(String(50), nullable=False, default="pending")  # pending, in_progress, completed
    due_date = Column(DateTime, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    assigned_user = relationship("User", back_populates="assigned_tasks")
    

