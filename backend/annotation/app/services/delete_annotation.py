# annotation/app/services/annotationService.py
import os
import yaml
import requests
from typing import Dict, Any
from fastapi import HTTPException
from sqlalchemy.orm import Session

from annotation.app.db.models.annotation import Annotation
from annotation.app.db.models.piece import Piece
from annotation.app.db.models.piece_image import PieceImage


def delete_annotation_service(annotation_id: int, db: Session) -> Dict[str, Any]:
    """
    Delete a specific annotation from the database and handle file cleanup.
    This service is owned by the annotation container.
    """
    try:
        # Find the annotation
        annotation = db.query(Annotation).filter(Annotation.id == annotation_id).first()
        
        if not annotation:
            raise HTTPException(status_code=404, detail="Annotation not found")
        
        # Get the associated image to update its annotation status if needed
        piece_image = db.query(PieceImage).filter(PieceImage.id == annotation.piece_image_id).first()
        
        if not piece_image:
            raise HTTPException(status_code=404, detail="Associated image not found")
        
        # Get the piece for file path reconstruction
        piece = db.query(Piece).filter(Piece.id == piece_image.piece_id).first()
        
        if not piece:
            raise HTTPException(status_code=404, detail="Associated piece not found")
        
        # Store information before deletion for file cleanup
        annotation_txt_name = annotation.annotationTXT_name
        piece_label = piece.piece_label
        
        # Delete the annotation first
        db.delete(annotation)
        db.flush()  # Ensure deletion is processed before checking remaining annotations
        
        # Check if there are any remaining annotations for this image
        remaining_annotations = db.query(Annotation).filter(
            Annotation.piece_image_id == piece_image.id
        ).count()
        
        print(f"Remaining annotations for image {piece_image.id}: {remaining_annotations}")
        
        # If no annotations remain, mark image as not annotated
        if remaining_annotations == 0:
            piece_image.is_annotated = False
            print(f"Marked image {piece_image.id} as not annotated")
            
            # Check if any other images in this piece are still annotated
            other_annotated_images = db.query(PieceImage).filter(
                PieceImage.piece_id == piece.id,
                PieceImage.is_annotated == True,
                PieceImage.id != piece_image.id  # Exclude current image
            ).count()
            
            print(f"Other annotated images in piece {piece.id}: {other_annotated_images}")
            
            # If no other images are annotated, update piece status via API
            if other_annotated_images == 0:
                print(f"Updating piece {piece_label} status to not annotated")
                update_success = _update_piece_annotation_status_via_api(piece_label, False)
                if not update_success:
                    print(f"Warning: Failed to update piece {piece_label} status via API")
        
        # Delete the corresponding annotation text file if it exists
        _delete_annotation_file(piece_label, annotation_txt_name)
        
        # Update data.yaml if needed
        _update_data_yaml_after_deletion(piece_label, piece.class_data_id, db)
        
        # Commit all changes
        db.commit()
        
        return {
            "status": "success",
            "message": f"Annotation {annotation_id} deleted successfully",
            "image_still_annotated": remaining_annotations > 0,
            "piece_label": piece_label,
            "annotation_file_deleted": True
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions as-is
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        error_msg = str(e)
        print(f"Error deleting annotation {annotation_id}: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Error deleting annotation: {error_msg}")


def delete_all_annotations_for_piece(piece_label: str, db: Session) -> Dict[str, Any]:
    """
    Delete all annotations for a specific piece.
    This is called when a piece is being deleted from artifact_keeper.
    """
    try:
        # Get the piece
        piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
        if not piece:
            return {"status": "success", "message": "No piece found, nothing to delete"}
        
        # Get all images for this piece
        images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).all()
        
        total_deleted = 0
        
        for image in images:
            # Get all annotations for this image
            annotations = db.query(Annotation).filter(Annotation.piece_image_id == image.id).all()
            
            # Delete each annotation
            for annotation in annotations:
                db.delete(annotation)
                total_deleted += 1
            
            # Mark image as not annotated
            image.is_annotated = False
        
        # Delete annotation files for this piece
        _delete_piece_annotation_files(piece_label)
        
        # Update data.yaml
        _update_data_yaml_after_piece_deletion(piece_label, piece.class_data_id, db)
        
        # Commit changes
        db.commit()
        
        return {
            "status": "success",
            "message": f"Deleted {total_deleted} annotations for piece {piece_label}",
            "deleted_count": total_deleted
        }
        
    except Exception as e:
        db.rollback()
        error_msg = str(e)
        print(f"Error deleting annotations for piece {piece_label}: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Error deleting annotations for piece: {error_msg}")


def get_annotations_for_piece(piece_label: str, db: Session) -> Dict[str, Any]:
    """
    Get all annotations for a specific piece.
    """
    try:
        # Get the piece
        piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
        if not piece:
            raise HTTPException(status_code=404, detail="Piece not found")
        
        # Get all images for this piece
        images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).all()
        
        result = {
            "piece_label": piece_label,
            "piece_id": piece.id,
            "images": []
        }
        
        for image in images:
            # Get all annotations for this image
            annotations = db.query(Annotation).filter(Annotation.piece_image_id == image.id).all()
            
            image_data = {
                "image_id": image.id,
                "file_name": image.file_name,
                "is_annotated": image.is_annotated,
                "annotations": []
            }
            
            for annotation in annotations:
                annotation_data = {
                    "id": annotation.id,
                    "type": annotation.type,
                    "x": annotation.x,
                    "y": annotation.y,
                    "width": annotation.width,
                    "height": annotation.height,
                    "annotationTXT_name": annotation.annotationTXT_name
                }
                image_data["annotations"].append(annotation_data)
            
            result["images"].append(image_data)
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        print(f"Error getting annotations for piece {piece_label}: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Error getting annotations for piece: {error_msg}")


def _update_piece_annotation_status_via_api(piece_label: str, is_annotated: bool) -> bool:
    """
    Update piece annotation status via artifact_keeper API.
    This respects service boundaries and data ownership.
    """
    try:
        response = requests.patch(
            f"http://artifact_keeper:8000/captureImage/pieces/{piece_label}/annotation-status",
            json={"is_annotated": is_annotated},
            timeout=10
        )
        
        if response.status_code == 200:
            print(f"Successfully updated piece {piece_label} annotation status to {is_annotated}")
            return True
        else:
            print(f"Failed to update piece annotation status: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Error calling artifact_keeper API: {str(e)}")
        return False


def _delete_annotation_file(piece_label: str, annotation_txt_name: str) -> None:
    """Delete the corresponding annotation text file if it exists"""
    try:
        # Use the unified dataset path for YOLO structure
        dataset_base_path = os.getenv('DATASET_BASE_PATH', '/app/shared/dataset')
        labels_folder = os.path.join(dataset_base_path, "piece", piece_label, "labels", "valid")
        
        # Get the annotation file name
        annotation_file_path = os.path.join(labels_folder, annotation_txt_name)
        
        if os.path.exists(annotation_file_path):
            os.remove(annotation_file_path)
            print(f"Deleted annotation file: {annotation_file_path}")
        else:
            print(f"Annotation file not found: {annotation_file_path}")
            
    except Exception as file_error:
        print(f"Warning: Could not delete annotation file: {str(file_error)}")


def _delete_piece_annotation_files(piece_label: str) -> None:
    """Delete all annotation files for a piece"""
    try:
        dataset_base_path = os.getenv('DATASET_BASE_PATH', '/app/shared/dataset')
        labels_folder = os.path.join(dataset_base_path, "piece", piece_label, "labels")
        
        if os.path.exists(labels_folder):
            import shutil
            shutil.rmtree(labels_folder)
            print(f"Deleted all annotation files for piece: {piece_label}")
        else:
            print(f"Labels folder not found for piece: {piece_label}")
            
    except Exception as e:
        print(f"Warning: Could not delete annotation files for piece {piece_label}: {str(e)}")


def _update_data_yaml_after_deletion(piece_label: str, class_data_id: int, db: Session) -> None:
    """Update data.yaml after annotation deletion"""
    try:
        dataset_base_path = os.getenv('DATASET_BASE_PATH', '/app/shared/dataset')
        data_yaml_path = os.path.join(dataset_base_path, "data.yaml")
        
        if not os.path.exists(data_yaml_path):
            return
        
        # Load existing data.yaml
        with open(data_yaml_path, 'r') as yaml_file:
            data_yaml = yaml.safe_load(yaml_file)
        
        # Check if there are any remaining annotations for this class
        remaining_annotations = db.query(Annotation).join(PieceImage).join(Piece).filter(
            Piece.class_data_id == class_data_id
        ).count()
        
        # If no annotations remain for this class, remove it from data.yaml
        if remaining_annotations == 0 and str(class_data_id) in data_yaml.get('names', {}):
            del data_yaml['names'][str(class_data_id)]
            data_yaml['nc'] = len(data_yaml['names'])
            
            # Write updated data.yaml
            with open(data_yaml_path, 'w') as yaml_file:
                yaml.dump(data_yaml, yaml_file, default_flow_style=False)
            
            print(f"Updated data.yaml: removed class {class_data_id}")
        
    except Exception as e:
        print(f"Warning: Could not update data.yaml after deletion: {str(e)}")


def _update_data_yaml_after_piece_deletion(piece_label: str, class_data_id: int, db: Session) -> None:
    """Update data.yaml after piece deletion"""
    _update_data_yaml_after_deletion(piece_label, class_data_id, db)