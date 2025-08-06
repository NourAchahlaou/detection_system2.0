from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel
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
@captureImage_router.get("/piece_label_byid/{piece_id}", response_model=str)
async def get_piece_labels_by_id(piece_id: int, db: db_dependency):
    """
    Get a piece label by its ID.    

     """
    piece = capture_service.get_piece_by_id(db, piece_id)
    if not piece:
        raise HTTPException(status_code=404, detail="Piece not found")
    return piece.piece_label

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
async def get_all_pieces(db: db_dependency):
    """
    Get all pieces.
    """
    pieces = db.query(Piece).all()
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
    
# Add these endpoints to your existing captureImage_router in artifact_keeper

@captureImage_router.patch("/piece-images/{image_id}/annotation-status")
def update_piece_image_annotation_status(
    image_id: int, 
    status_update: PieceAnnotationStatusUpdate,
    db: Session = Depends(get_session)
):
    """
    Update the annotation status of a piece image.
    This endpoint should only be called by the annotation service.
    """
    try:
        from artifact_keeper.app.db.models.piece_image import PieceImage
        
        # Find the piece image
        piece_image = db.query(PieceImage).filter(PieceImage.id == image_id).first()
        
        if not piece_image:
            raise HTTPException(status_code=404, detail="Piece image not found")
        
        # Update the annotation status
        piece_image.is_annotated = status_update.is_annotated
        
        # Commit the changes
        db.commit()
        db.refresh(piece_image)
        
        logger.info(f"Updated piece image {image_id} annotation status to {status_update.is_annotated}")
        
        return {
            "message": f"Piece image {image_id} annotation status updated to {status_update.is_annotated}",
            "image_id": image_id,
            "is_annotated": status_update.is_annotated
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to update piece image annotation status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to update piece image annotation status: {str(e)}")

@captureImage_router.get("/pieces/{piece_label}/annotation-status")
def get_piece_annotation_status(
    piece_label: str,
    db: Session = Depends(get_session)
):
    """
    Get the annotation status of a piece and count of remaining unannotated images.
    This endpoint is used by the annotation service to check completion status.
    """
    try:
        from artifact_keeper.app.db.models.piece import Piece
        from artifact_keeper.app.db.models.piece_image import PieceImage
        
        # Find the piece
        piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
        
        if not piece:
            raise HTTPException(status_code=404, detail="Piece not found")
        
        # Count remaining unannotated images
        remaining_unannotated = db.query(PieceImage).filter(
            PieceImage.piece_id == piece.id,
            PieceImage.is_annotated == False,
            PieceImage.is_deleted == False  # Don't count deleted images
        ).count()
        
        # Count total active images
        total_active_images = db.query(PieceImage).filter(
            PieceImage.piece_id == piece.id,
            PieceImage.is_deleted == False
        ).count()
        
        logger.info(f"Piece {piece_label} status: {remaining_unannotated} unannotated out of {total_active_images} total images")
        
        return {
            "piece_label": piece_label,
            "is_annotated": piece.is_annotated,
            "remaining_unannotated_images": remaining_unannotated,
            "total_images": total_active_images,
            "piece_id": piece.id
        }
        
    except Exception as e:
        logger.error(f"Failed to get piece annotation status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get piece annotation status: {str(e)}")

@captureImage_router.get("/pieces/{piece_label}/images")
def get_piece_images_status(
    piece_label: str,
    db: Session = Depends(get_session)
):
    """
    Get detailed information about all images for a piece.
    This can be useful for debugging annotation status issues.
    """
    try:
        from artifact_keeper.app.db.models.piece import Piece
        from artifact_keeper.app.db.models.piece_image import PieceImage
        
        # Find the piece
        piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
        
        if not piece:
            raise HTTPException(status_code=404, detail="Piece not found")
        
        # Get all images for this piece
        images = db.query(PieceImage).filter(
            PieceImage.piece_id == piece.id,
            PieceImage.is_deleted == False
        ).all()
        
        image_details = []
        for img in images:
            image_details.append({
                "id": img.id,
                "file_name": img.file_name,
                "is_annotated": img.is_annotated,
                "upload_date": img.upload_date.isoformat() if img.upload_date else None,
                "image_path": img.image_path
            })
        
        return {
            "piece_label": piece_label,
            "piece_id": piece.id,
            "piece_is_annotated": piece.is_annotated,
            "total_images": len(image_details),
            "annotated_images": sum(1 for img in image_details if img["is_annotated"]),
            "images": image_details
        }
        
    except Exception as e:
        logger.error(f"Failed to get piece images status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get piece images status: {str(e)}")

class BatchUpdateRequest(BaseModel):
    image_ids: List[int]
    is_annotated: bool

@captureImage_router.patch("/pieces/{piece_label}/batch-update-images")
def batch_update_piece_images_annotation_status(
    piece_label: str,
    request: BatchUpdateRequest,  # Use a proper request model
    db: Session = Depends(get_session)
):
    """
    Batch update annotation status for multiple images.
    This is more efficient when updating many images at once.
    """
    try:
        from artifact_keeper.app.db.models.piece import Piece
        from artifact_keeper.app.db.models.piece_image import PieceImage
        
        # Find the piece first
        piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
        
        if not piece:
            logger.error(f"Piece not found: {piece_label}")
            raise HTTPException(status_code=404, detail="Piece not found")
        
        logger.info(f"Attempting to update {len(request.image_ids)} images for piece {piece_label}")
        logger.info(f"Image IDs: {request.image_ids}")
        
        # Check which images exist before updating
        existing_images = db.query(PieceImage).filter(
            PieceImage.id.in_(request.image_ids),
            PieceImage.piece_id == piece.id,
            PieceImage.is_deleted == False
        ).all()
        
        logger.info(f"Found {len(existing_images)} existing images out of {len(request.image_ids)} requested")
        
        # Update all specified images
        updated_count = db.query(PieceImage).filter(
            PieceImage.id.in_(request.image_ids),
            PieceImage.piece_id == piece.id,  # Security: ensure images belong to this piece
            PieceImage.is_deleted == False
        ).update(
            {"is_annotated": request.is_annotated},
            synchronize_session=False
        )
        
        if updated_count == 0:
            logger.warning(f"No images were updated for piece {piece_label}")
            raise HTTPException(status_code=404, detail="No matching images found to update")
        
        # Commit the changes
        db.commit()
        
        logger.info(f"Batch updated {updated_count} images for piece {piece_label} to annotation status {request.is_annotated}")
        
        return {
            "message": f"Successfully updated {updated_count} images",
            "piece_label": piece_label,
            "updated_image_count": updated_count,
            "is_annotated": request.is_annotated
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to batch update piece images: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to batch update piece images: {str(e)}")