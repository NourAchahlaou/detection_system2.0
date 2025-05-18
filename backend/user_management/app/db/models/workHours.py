from sqlalchemy import Column, DateTime, Integer, String, func, ForeignKey
from user_management.app.db.models.roleType import RoleType
from user_management.app.db.session import Base
from sqlalchemy.orm import relationship

class WorkHours(Base):
    __tablename__ = "work_hours"
    __table_args__ = {"schema": "user_mgmt"}    
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    login_time = Column(DateTime, nullable=False)
    logout_time = Column(DateTime, nullable=True)
    total_minutes = Column(Integer, nullable=True)  # Calculated on logout
    
    # Relationships
    user = relationship("User")
    

