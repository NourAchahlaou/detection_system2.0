from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, Integer, Time
from user_management.app.db.session import Base
from sqlalchemy.orm import relationship

class Shift(Base) :

    __tablename__ = "shifts"
    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)
    day_of_week = Column(Integer, nullable=False)  # 0 = Monday, 6 = Sunday
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="shifts")
 

