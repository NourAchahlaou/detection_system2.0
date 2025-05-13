from sqlalchemy import Integer, Column, String, Float
from piece_registry.app.db.session import Base
from sqlalchemy.orm import relationship

class CameraSettings (Base):
    __tablename__ = 'cameraSettings'

    id = Column(Integer, primary_key=True, index=True)
    exposure = Column(Float)
    contrast = Column(Float)
    brightness = Column(Float)
    focus = Column(Float)
    aperture= Column (Float)
    gain = Column(Integer)
    white_balance = Column(String)


    camera = relationship("Camera", back_populates="sittings", uselist=True)
