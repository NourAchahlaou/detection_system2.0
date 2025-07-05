# artifact_keeper/app/db/models/annotation_view.py
"""
Read-only view of Annotation model for artifact_keeper service.
This service can only READ annotation data, not modify it.
All annotation modifications must go through the annotation service.
"""

from sqlalchemy import Column, Float, Integer, String
from training.app.db.session import Base


class Annotation(Base):
    """
    Read-only view of the Annotation table from the annotation service.
    
    This model allows artifact_keeper to read annotation data for display
    purposes but should NOT be used for any write operations.
    
    All annotation CRUD operations should be done through the annotation service API.
    """
    __tablename__ = "annotation"
    __table_args__ = {"schema": "annotation"}
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    annotationTXT_name = Column(String, nullable=False)
    type = Column(String, nullable=False)
    x = Column(Float, nullable=False)
    y = Column(Float, nullable=False)
    width = Column(Float, nullable=False)
    height = Column(Float, nullable=False)
    piece_image_id = Column(Integer, nullable=False)
    
    class Config:
        # Make this model read-only
        from_attributes = True
        
    def __repr__(self):
        return f"<AnnotationView(id={self.id}, type={self.type}, piece_image_id={self.piece_image_id})>"