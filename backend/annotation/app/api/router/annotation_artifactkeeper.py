# annotation/app/api/router/annotationRouter.py
from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from annotation.app.db.session import get_db
from annotation.app.services.delete_annotation import (
    delete_annotation_service,
    delete_all_annotations_for_piece,
    get_annotations_for_piece
)

router = APIRouter(prefix="/annotations", tags=["annotations"])


@router.delete("/{annotation_id}")
def delete_annotation(
    annotation_id: int,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Delete a specific annotation by ID.
    This endpoint is owned by the annotation service.
    """
    try:
        result = delete_annotation_service(annotation_id, db)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete annotation: {str(e)}")


@router.delete("/piece/{piece_label}")
def delete_all_annotations_for_piece_endpoint(
    piece_label: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Delete all annotations for a specific piece.
    This is typically called by the artifact_keeper service when deleting a piece.
    """
    try:
        result = delete_all_annotations_for_piece(piece_label, db)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete annotations for piece: {str(e)}")


@router.get("/piece/{piece_label}")
def get_annotations_for_piece_endpoint(
    piece_label: str,
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get all annotations for a specific piece.
    """
    try:
        result = get_annotations_for_piece(piece_label, db)
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get annotations for piece: {str(e)}")


@router.get("/health")
def health_check():
    """Health check endpoint for the annotation service"""
    return {"status": "healthy", "service": "annotation"}