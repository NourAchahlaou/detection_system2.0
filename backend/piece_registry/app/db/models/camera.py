from sqlalchemy import Boolean, String, Column,Integer,ForeignKey
from piece_registry.app.db.session import Base
from sqlalchemy.orm import relationship

class Camera(Base):
    __tablename__ ='camera'

    id=Column(Integer, primary_key=True, index=True)
    camera_index=Column(Integer)
    model = Column(String, index=True)  
    status = Column(Boolean, default='False')
    settings_id = Column(Integer, ForeignKey('cameraSettings.id'))
    sittings = relationship("CameraSettings", back_populates="camera")

