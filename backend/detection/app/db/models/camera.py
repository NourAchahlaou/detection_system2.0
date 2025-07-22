from sqlalchemy import Boolean, String, Column,Integer,ForeignKey
from detection.app.db.session import Base
from sqlalchemy.orm import relationship

class Camera(Base):
    __tablename__ ='camera'
    __table_args__ = {"schema": "artifact_keeper"}

    id=Column(Integer, primary_key=True, index=True)
    camera_index=Column(Integer)
    camera_type = Column(String, nullable=False)  # "regular" or "industrial"
    serial_number = Column(String, unique=True, nullable=True)  # For industrial cameras
    model = Column(String, index=True)  
    status = Column(Boolean, default='False')
    settings_id = Column(Integer, ForeignKey('artifact_keeper.cameraSettings.id'))
    settings = relationship("CameraSettings", back_populates="camera", uselist=False)

