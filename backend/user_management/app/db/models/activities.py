from datetime import datetime
from sqlalchemy import Column, DateTime, ForeignKey, Integer, Text
from sqlalchemy.orm import relationship
from user_management.app.db.models.actionType import ActionType
from user_management.app.db.session import Base
from sqlalchemy import Enum as SQLEnum

class Activity(Base):
    __tablename__ = "activities"
    __table_args__ = {"schema": "user_mgmt"}    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("user_mgmt.users.id"), nullable=False)
    action_type = Column(SQLEnum(ActionType,schema="user_mgmt"), nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)
    details = Column(Text, nullable=True)
    
    # Relationships
    user = relationship("User", back_populates="activities")
    # piece = relationship("Piece", back_populates="activities")
    
 

