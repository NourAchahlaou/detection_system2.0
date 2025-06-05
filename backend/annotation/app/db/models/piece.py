from sqlalchemy import Column, DateTime, Integer, String, Boolean, func
from sqlalchemy.orm import relationship
from annotation.app.db.session import Base

class Piece(Base):
    """
    Read-only model for querying pieces from artifact_keeper schema.
    This service does NOT own this table - it's owned by artifact_keeper.
    """
    __tablename__ = 'piece'
    __table_args__ = {"schema": "artifact_keeper"}

    id = Column(Integer, primary_key=True, index=True)
    class_data_id = Column(Integer, nullable=True)
    piece_label = Column(String, nullable=False, unique=True)
    is_annotated = Column(Boolean, default=False)
    is_yolo_trained = Column(Boolean, default=False)
    nbre_img = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
 
    # Relationship to piece images (read-only)
    piece_img = relationship("PieceImage", back_populates="piece", viewonly=True)
