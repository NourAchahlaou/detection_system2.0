from sqlalchemy import Column, Float, Integer, String, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from datetime import datetime

from database.defectDetectionDB import Base

class Annotation(Base):
    __tablename__ = "annotation"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    annotationTXT_name = Column(String, primary_key=True, nullable=False)  
    type = Column(String, nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
  

    piece_image_id = Column(Integer, ForeignKey('piece_image.id'))
    piece_image = relationship("PieceImage", back_populates="annotations")
