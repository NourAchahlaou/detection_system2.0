
from sqlalchemy import Column, DateTime, Integer, Null, String,Boolean, func
from sqlalchemy.orm import relationship
from artifact_keeper.app.db.session import Base

class Piece(Base):
    __tablename__ = 'piece'
    __table_args__ = {"schema": "artifact_keeper"}

    id = Column(Integer, primary_key=True, index=True)
    class_data_id = Column(Integer,default = Null )
    piece_label = Column(String, nullable=False, unique=True)
    is_annotated = Column(Boolean,default=False)
    is_yolo_trained = Column(Boolean,default=False)
    nbre_img= Column(Integer,default = Null)
    created_at = Column(DateTime(timezone=True), server_default=func.now()) 
 
    piece_img = relationship("PieceImage", back_populates="piece", viewonly=True)
