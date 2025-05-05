from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, Integer, String, func, ForeignKey
from user_management.app.db.models.roleType import RoleType
from user_management.app.db.session import Base
from sqlalchemy.orm import mapped_column, relationship
from sqlalchemy import Enum as SQLEnum

class User(Base):
    __tablename__ = 'users'
    id = Column(Integer, primary_key=True, autoincrement=True)
    airbus_id = Column(Integer, unique=True)
    name = Column(String(150))
    email = Column(String(255), unique=True, index=True)
    password = Column(String(100))
    is_active = Column(Boolean, default=False)
    verified_at = Column(DateTime, nullable=True, default=None)
    updated_at = Column(DateTime, nullable=True, default=None, onupdate=datetime.now)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    activation_code = Column(String(8), nullable=True)
    activation_code_expires_at = Column(DateTime, nullable=True)
    tokens = relationship("UserToken", back_populates="user")

    role = Column(SQLEnum(RoleType), nullable=False)
    shifts = relationship("Shift", back_populates="user")
    activities = relationship("Activity", back_populates="user")
    assigned_tasks = relationship("Task", back_populates="assigned_user")

    def get_context_string(self, context: str):
        return f"{context}{self.password[-6:]}{self.updated_at.strftime('%m%d%Y%H%M%S')}".strip()
    
    

class UserToken(Base):
    __tablename__ = "user_tokens"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = mapped_column(ForeignKey('users.id'))
    access_key = Column(String(250), nullable=True, index=True, default=None)
    refresh_key = Column(String(250), nullable=True, index=True, default=None)
    created_at = Column(DateTime, nullable=False, server_default=func.now())
    expires_at = Column(DateTime, nullable=False)
    
    user = relationship("User", back_populates="tokens")