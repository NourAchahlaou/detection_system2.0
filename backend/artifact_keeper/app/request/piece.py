from pydantic import BaseModel, Field


class SaveImagesRequest(BaseModel):
    piece_label: str = Field(..., description="The label of the piece to save images for")
class PieceAnnotationStatusUpdate(BaseModel):
    is_annotated: bool