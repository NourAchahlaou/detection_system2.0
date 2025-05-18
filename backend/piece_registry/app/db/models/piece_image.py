from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String
from sqlalchemy.orm import relationship
from piece_registry.app.db.session import Base
from sqlalchemy.sql import func

class PieceImage(Base):
    __tablename__ = 'piece_image'
    __table_args__ = {"schema": "piece_reg"}

    id = Column(Integer, primary_key=True, index=True)
    piece_id = Column(Integer, ForeignKey('piece_reg.piece.id'), nullable=False)
    file_name = Column(String, nullable=False)
    image_path = Column(String, nullable=False)
    upload_date = Column(DateTime(timezone=True), server_default=func.now())
    is_deleted = Column(Boolean, default=False)
    piece = relationship("Piece", back_populates="piece_img")

