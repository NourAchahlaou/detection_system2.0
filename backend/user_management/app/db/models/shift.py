from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, Integer, Time
from user_management.app.db.session import Base
from sqlalchemy.orm import relationship
from sqlalchemy import Enum as SQLEnum
from user_management.app.db.models.shiftDay import ShiftDay
class Shift(Base) :

    __tablename__ = "shifts"
    __table_args__ = {"schema": "user_mgmt"}    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user_mgmt.users.id"), nullable=False)
    start_time = Column(Time, nullable=False)
    end_time = Column(Time, nullable=False)
    day_of_week = Column(SQLEnum(ShiftDay,schema="user_mgmt"), nullable=False)

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    user = relationship("User", back_populates="shifts")
 

