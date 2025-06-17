from fastapi import APIRouter, Depends, HTTPException, Response
from sqlalchemy.orm import Session
from typing import List, Dict, Annotated
import logging
from artifact_keeper.app.db.session import get_session
from artifact_keeper.app.services.captureService import CaptureService

from artifact_keeper.app.response.camera import (


    CleanupResponse,
    CircuitBreakerStatusResponse,
)
from artifact_keeper.app.response.piece import (
    PieceResponse,
    SaveImagesResponse
)

from artifact_keeper.app.request.piece import (
    SaveImagesRequest,
    PieceAnnotationStatusUpdate,
)
from artifact_keeper.app.db.models.piece import Piece

captureImage_router = APIRouter(
    prefix="/captureImage",
    tags=["CaptureImage"],
    responses={404: {"description": "Not found"}},
)

logger = logging.getLogger(__name__)
capture_service = CaptureService()

# Database dependency
db_dependency = Annotated[Session, Depends(get_session)]



@captureImage_router.get("/capture_images/{piece_label}")
async def capture_images(piece_label: str):
    """
    Capture images for a piece.
    """
    image_content = capture_service.capture_images(piece_label)
    return Response(content=image_content, media_type="image/jpeg")

@captureImage_router.post("/cleanup-temp-photos", response_model=CleanupResponse)
async def cleanup_temp_photos():
    """
    Clean up temporary photos.
    """
    return capture_service.cleanup_temp_photos()

@captureImage_router.post("/save-images", response_model=SaveImagesResponse)
async def save_images(request: SaveImagesRequest, db: db_dependency):
    """
    Save captured images to the database.
    """
    return capture_service.save_images_to_database(db, request.piece_label)

@captureImage_router.get("/pieces/{piece_label}", response_model=PieceResponse)
async def get_piece_by_label(piece_label: str, db: db_dependency):
    """
    Get a piece by its label.
    """
    from sqlalchemy.orm import joinedload
    from artifact_keeper.app.db.models.piece import Piece
    
    piece = db.query(Piece).filter(Piece.piece_label == piece_label).options(
        joinedload(Piece.piece_img)  # Changed from Piece.images to Piece.piece_img
    ).first()
    
    if not piece:
        raise HTTPException(status_code=404, detail=f"Piece with label {piece_label} not found")
    
    return piece

@captureImage_router.get("/pieces", response_model=List[PieceResponse])
async def get_all_pieces(db: db_dependency, skip: int = 0, limit: int = 100):
    """
    Get all pieces.
    """
    from sqlalchemy.orm import joinedload
    from artifact_keeper.app.db.models.piece import Piece
    
    pieces = db.query(Piece).options(
        joinedload(Piece.images)
    ).offset(skip).limit(limit).all()
    
    return pieces

@captureImage_router.get("/circuit-breaker-status", response_model=Dict[str, CircuitBreakerStatusResponse])
async def get_circuit_breaker_status():
    """
    Get the status of all circuit breakers.
    """
    return capture_service.get_circuit_breaker_status()

@captureImage_router.post("/reset-circuit-breaker/{breaker_name}", response_model=Dict[str, str])
async def reset_circuit_breaker(breaker_name: str):
    """
    Reset a specific circuit breaker.
    """
    return capture_service.reset_circuit_breaker(breaker_name)



@captureImage_router.get("/temp-photos/{piece_label}")
async def get_temp_photos_endpoint(
    piece_label: str,
    db: Session = Depends(get_session),
    
):
    """Get temporary photos for a piece."""
    try:
        temp_images = capture_service.get_temp_images_for_piece(piece_label)
        return temp_images
    except Exception as e:
        logger.error(f"Error in get_temp_photos_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@captureImage_router.get("/temp-image/{image_name}")
async def serve_temp_image_endpoint(
    image_name: str,
):
    """Serve a temporary image file."""
    try:
        image_data = capture_service.serve_temp_image(image_name)
        return Response(
            content=image_data,
            media_type="image/jpeg",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"}
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in serve_temp_image_endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

@captureImage_router.delete("/temp-image/{piece_label}/{image_name}")
async def delete_temp_image(
    piece_label: str,
    image_name: str,

):
    """Delete a specific temporary image."""
    try:
        success = capture_service.delete_temp_image(piece_label, image_name)
        if success:
            return {"message": f"Temporary image {image_name} deleted successfully"}
        else:
            raise HTTPException(status_code=404, detail="Temporary image not found")
    except Exception as e:
        logger.error(f"Error deleting temp image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete temporary image: {str(e)}")    
    

@captureImage_router.patch("/pieces/{piece_label}/annotation-status")
def update_piece_annotation_status(
    piece_label: str, 
    status_update: PieceAnnotationStatusUpdate,
    db: Session = Depends(get_session)
):
    """
    Update the annotation status of a piece.
    This endpoint should only be called by the annotation service.
    """
    try:
        # Find the piece
        piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
        
        if not piece:
            raise HTTPException(status_code=404, detail="Piece not found")
        
        # Update the annotation status
        piece.is_annotated = status_update.is_annotated
        
        # Commit the changes
        db.commit()
        db.refresh(piece)
        
        return {
            "message": f"Piece {piece_label} annotation status updated to {status_update.is_annotated}",
            "piece_label": piece_label,
            "is_annotated": status_update.is_annotated
        }
        
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update piece annotation status: {str(e)}")