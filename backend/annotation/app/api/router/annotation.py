import os
import re
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException
from annotation.app.db.session import get_session
from sqlalchemy.orm import Session
from sqlalchemy import func, exists, and_, case  # Add these imports

from annotation.app.db.models.piece_image import PieceImage
from annotation.app.services.piece_service import (
    get_images_of_piece,
    get_img_non_annotated,
    save_annotation_in_memory,
    save_annotations_to_db,
    delete_annotation_service,
    delete_virtual_annotation_service,
    get_virtual_annotations_service,
    virtual_storage
)
from annotation.app.db.models.annotation import Annotation
from annotation.app.db.models.piece import Piece
db_dependency = Annotated[Session, Depends(get_session)]

from pydantic import BaseModel, Field

class AnnotationData(BaseModel):
    image_id: int = Field(..., description="ID of the image to annotate")
    type: str = Field(..., description="Type of the annotation (e.g., 'object', 'label')")
    x: float = Field(..., description="X coordinate of the annotation bounding box")
    y: float = Field(..., description="Y coordinate of the annotation bounding box")
    width: float = Field(..., description="Width of the annotation bounding box")
    height: float = Field(..., description="Height of the annotation bounding box")

annotation_router = APIRouter(
    prefix="/annotations",
    tags=["Annotations"],
    responses={404: {"description": "Not found"}},
)

@annotation_router.get("/image-id")
def get_image_id(image_path: str, db: db_dependency):
    """Get image ID by image_path"""
    print(f"Looking for image with path: {image_path}")
    
    # Try exact match first
    piece_image = db.query(PieceImage).filter(PieceImage.image_path == image_path).first()
    
    if not piece_image:
        # Try with different path formats if exact match fails
        # Remove any leading/trailing slashes and try again
        cleaned_path = image_path.strip('/')
        piece_image = db.query(PieceImage).filter(PieceImage.image_path == cleaned_path).first()
        
        if not piece_image:
            # Try with backslashes converted to forward slashes
            path_with_forward_slashes = image_path.replace('\\', '/')
            piece_image = db.query(PieceImage).filter(PieceImage.image_path == path_with_forward_slashes).first()
            
        if not piece_image:
            # Try with forward slashes converted to backslashes
            path_with_backslashes = image_path.replace('/', '\\')
            piece_image = db.query(PieceImage).filter(PieceImage.image_path == path_with_backslashes).first()
    
    if not piece_image:
        # Debug: Show all available image paths
        all_images = db.query(PieceImage.image_path).all()
        available_paths = [img.image_path for img in all_images]
        print(f"Available image paths in database: {available_paths}")
        raise HTTPException(status_code=404, detail=f"Image not found for path: {image_path}")
    
    print(f"Found image with ID: {piece_image.id}")
    return {"image_id": piece_image.id}

@annotation_router.get("/get_images_of_piece/{piece_label}")
async def get_image_of_piece_byLabel(db: db_dependency, piece_label: str):
    return get_images_of_piece(piece_label, db)

@annotation_router.get("/get_Img_nonAnnotated")
def get_img_non_annotated_route(db: Session = Depends(get_session)):
    return get_img_non_annotated(db)

@annotation_router.post("/{piece_label}")
def create_annotation(piece_label: str, annotation_data: AnnotationData):
    # Extract image_id from the request body
    image_id = annotation_data.image_id

    if not image_id:
        raise HTTPException(status_code=400, detail="Missing image_id in annotation data.")

    # Save the annotation in virtual storage
    save_annotation_in_memory(piece_label, image_id, annotation_data.dict())

    return {"status": "Annotation saved in memory"}

@annotation_router.post("/saveAnnotation/{piece_label}")
def saveAnnotation(piece_label: str, db: db_dependency):
    print(f"Received piece_label: {piece_label}")

    # Ensure piece_label is provided
    if not piece_label:
        raise HTTPException(status_code=422, detail="piece_label is required")

    match = re.match(r'([A-Z]\d{3}\.\d{5})', piece_label)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid piece_label format.")
    extracted_label = match.group(1)

    # Define save_folder using the extracted_label
    save_folder = os.path.join("dataset", "Pieces", "Pieces", "labels", "valid", extracted_label, piece_label)
    os.makedirs(save_folder, exist_ok=True)
    
    try:
        result = save_annotations_to_db(db, piece_label, save_folder)
        result1 = get_images_of_piece(piece_label, db)
    except SystemError as e:
        raise HTTPException(status_code=500, detail=str(e))

    if result is None:
        raise HTTPException(status_code=500, detail="Failed to capture frame from the camera.")
    if result1 is None:
        raise HTTPException(status_code=500, detail="Failed to capture frame from the camera.")
    
    return piece_label, save_folder, result, result1

@annotation_router.get("/image/{image_id}/annotations")
def get_image_annotations(image_id: int, db: db_dependency):
    """Get all annotations for a specific image"""
    try:
        # Fetch the image to verify it exists
        piece_image = db.query(PieceImage).filter(PieceImage.id == image_id).first()
        
        if not piece_image:
            raise HTTPException(status_code=404, detail="Image not found")
        
        # Fetch all annotations for this image
        annotations = db.query(Annotation).filter(Annotation.piece_image_id == image_id).all()
        
        # Convert annotations to frontend format (percentage-based coordinates)
        result = []
        for annotation in annotations:
            # Convert from YOLO format (normalized) back to percentage coordinates
            # YOLO format: x_center, y_center, width, height (all normalized 0-1)
            # Frontend format: x, y, width, height (percentage 0-100)
            
            x_center_normalized = annotation.x
            y_center_normalized = annotation.y
            width_normalized = annotation.width
            height_normalized = annotation.height
            
            # Convert to top-left corner coordinates in percentage
            x_percentage = (x_center_normalized - width_normalized / 2) * 100
            y_percentage = (y_center_normalized - height_normalized / 2) * 100
            width_percentage = width_normalized * 100
            height_percentage = height_normalized * 100
            
            result.append({
                "id": annotation.id,
                "type": annotation.type,
                "x": x_percentage,
                "y": y_percentage,
                "width": width_percentage,
                "height": height_percentage,
                "annotationTXT_name": annotation.annotationTXT_name
            })
        
        return {
            "image_id": image_id,
            "annotations": result,
            "count": len(result)
        }
        
    except Exception as e:
        print(f"Error fetching annotations for image {image_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching annotations: {str(e)}")

@annotation_router.get("/piece/{piece_label}/annotations")
def get_piece_annotations(piece_label: str, db: db_dependency):
    """Get all annotations for all images of a piece"""
    try:
        # Find the piece
        piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
        if not piece:
            raise HTTPException(status_code=404, detail="Piece not found")
        
        # Get all images for this piece
        piece_images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).all()
        
        result = {}
        for image in piece_images:
            # Get annotations for each image
            annotations = db.query(Annotation).filter(Annotation.piece_image_id == image.id).all()
            
            image_annotations = []
            for annotation in annotations:
                # Convert from YOLO format back to percentage
                x_center_normalized = annotation.x
                y_center_normalized = annotation.y
                width_normalized = annotation.width
                height_normalized = annotation.height
                
                x_percentage = (x_center_normalized - width_normalized / 2) * 100
                y_percentage = (y_center_normalized - height_normalized / 2) * 100
                width_percentage = width_normalized * 100
                height_percentage = height_normalized * 100
                
                image_annotations.append({
                    "id": annotation.id,
                    "type": annotation.type,
                    "x": x_percentage,
                    "y": y_percentage,
                    "width": width_percentage,
                    "height": height_percentage,
                    "annotationTXT_name": annotation.annotationTXT_name
                })
            
            result[str(image.id)] = {
                "image_path": image.image_path,
                "file_name": image.file_name,
                "annotations": image_annotations,
                "count": len(image_annotations)
            }
        
        return {
            "piece_label": piece_label,
            "images": result
        }
        
    except Exception as e:
        print(f"Error fetching annotations for piece {piece_label}: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching annotations: {str(e)}")
    
@annotation_router.delete("/{annotation_id}")
def delete_annotation(annotation_id: int, db: db_dependency):
    """Delete a specific annotation from the database"""
    return delete_annotation_service(annotation_id, db)

@annotation_router.delete("/virtual/{piece_label}/{image_id}/{annotation_id}")
def delete_virtual_annotation(piece_label: str, image_id: int, annotation_id: str):
    """Delete a specific annotation from virtual storage"""
    return delete_virtual_annotation_service(piece_label, image_id, annotation_id, virtual_storage)

@annotation_router.get("/virtual/{piece_label}")
def get_virtual_annotations(piece_label: str):
    """Get all annotations currently in virtual storage for a piece"""
    return get_virtual_annotations_service(piece_label, virtual_storage)    

@annotation_router.get("/get_all_pieces")
def get_all_pieces_route(db: db_dependency):
    """Get all pieces in the system with their annotation status"""
    try:
        # Simplified and corrected query
        pieces = db.query(Piece).all()
        
        result = []
        for piece in pieces:
            # Get total image count for this piece
            total_images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).count()
            
            # Get annotated image count for this piece
            annotated_images = db.query(PieceImage).filter(
                PieceImage.piece_id == piece.id,
                PieceImage.is_annotated == True
            ).count()
            
            # Get a sample image for preview
            sample_image = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).first()
            
            piece_data = {
                "piece_label": piece.piece_label,
                "nbr_img": total_images,
                "annotated_count": annotated_images,
                "url": sample_image.image_path if sample_image else None,
                "is_fully_annotated": annotated_images >= total_images
            }
            result.append(piece_data)

        return result

    except Exception as e:
        print(f"Error fetching all pieces: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching pieces: {str(e)}")
