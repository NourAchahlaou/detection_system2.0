from datetime import datetime
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from piece_registry.app.db.session import Base

class PieceImage(Base):
    __tablename__ = 'piece_image'
    
    id = Column(Integer, primary_key=True, index=True)
    image_name =  Column(String, nullable=False, unique=True)
    piece_path = Column(String, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    url = Column(String, nullable=False, unique= True)
    is_annotated = Column(Boolean,default=False)
    # Foreign key to reference Piece
    piece_id = Column(Integer, ForeignKey('piece.id'), nullable=False)

    # Define the relationship back to Piece
    piece = relationship("Piece", back_populates="piece_img")   
    # annotations = relationship("Annotation", back_populates="piece_image")