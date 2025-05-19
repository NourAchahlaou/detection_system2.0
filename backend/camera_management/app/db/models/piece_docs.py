from sqlalchemy import Column, Integer, String, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from db.session import Base

 
class PieceDocument(Base):
    __tablename__ = 'piece_document'
    __table_args__ = {"schema": "piece_reg"}  
    id = Column(Integer, primary_key=True, index=True)
    piece_id = Column(Integer, ForeignKey('piece_reg.piece.id'), nullable=False)
    document_type = Column(String, nullable=False)
    document_path = Column(String, nullable=False)
    file_name = Column(String, nullable=False)
    source = Column(String, nullable=False)  # e.g., "Airbus Dataset"
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    is_deleted = Column(Boolean, default=False)
    

    piece = relationship("Piece", back_populates="documents")