from sqlalchemy import Column, Float, Integer, String
from annotation.app.db.session import Base

class Annotation(Base):
    __tablename__ = "annotation"
    __table_args__ = {"schema": "annotation"}
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    annotationTXT_name = Column(String, primary_key=True, nullable=False)
    type = Column(String, nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    piece_image_id = Column(Integer, nullable=False)  
    
