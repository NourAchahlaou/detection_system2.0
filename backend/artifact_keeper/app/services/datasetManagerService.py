import os
import re
import shutil
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import HTTPException
from sqlalchemy.orm import Session
from sqlalchemy import String, func, desc, asc, or_
from sqlalchemy.exc import SQLAlchemyError
from artifact_keeper.app.db.models.piece import Piece
from artifact_keeper.app.db.models.piece_image import PieceImage
from artifact_keeper.app.db.models.annotation import Annotation

import logging
logger = logging.getLogger(__name__)

def get_all_datasets_with_filters(
    db: Session,
    page: int = 1,
    page_size: int = 10,
    search: Optional[str] = None,
    status_filter: Optional[str] = None,
    training_filter: Optional[str] = None,
    group_filter: Optional[str] = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    min_images: Optional[int] = None,
    max_images: Optional[int] = None
) -> Dict[str, Any]:
    """
    Enhanced function to fetch datasets with filtering, searching, sorting, and pagination.
    """
    try:
        # Base query
        query = db.query(Piece)
        
        # Apply search filter
        if search:
            search_term = f"%{search}%"
            query = query.filter(
                or_(
                    Piece.piece_label.ilike(search_term),
                    func.cast(Piece.class_data_id, String).ilike(search_term)
                )
            )
        
        # Apply status filter (annotation status)
        if status_filter:
            if status_filter == "annotated":
                query = query.filter(Piece.is_annotated == True)
            elif status_filter == "not_annotated":
                query = query.filter(Piece.is_annotated == False)
        
        # Apply training filter
        if training_filter:
            if training_filter == "trained":
                query = query.filter(Piece.is_yolo_trained == True)
            elif training_filter == "not_trained":
                query = query.filter(Piece.is_yolo_trained == False)
        
        # Apply group filter
        if group_filter:
            query = query.filter(Piece.piece_label.like(f"{group_filter}%"))
        
        # Apply date range filter
        if date_from:
            date_from_obj = datetime.fromisoformat(date_from.replace('Z', '+00:00'))
            query = query.filter(Piece.created_at >= date_from_obj)
        
        if date_to:
            date_to_obj = datetime.fromisoformat(date_to.replace('Z', '+00:00'))
            query = query.filter(Piece.created_at <= date_to_obj)
        
        # Apply image count filter
        if min_images is not None:
            query = query.filter(Piece.nbre_img >= min_images)
        
        if max_images is not None:
            query = query.filter(Piece.nbre_img <= max_images)
        
        # Get total count before pagination
        total_count = query.count()
        
        # Apply sorting
        sort_column = getattr(Piece, sort_by, Piece.created_at)
        if sort_order.lower() == "desc":
            query = query.order_by(desc(sort_column))
        else:
            query = query.order_by(asc(sort_column))
        
        # Apply pagination
        offset = (page - 1) * page_size
        pieces = query.offset(offset).limit(page_size).all()
        
        # Process pieces data
        datasets = []
        for piece_record in pieces:
            piece_data = {
                "id": piece_record.id,
                "class_data_id": piece_record.class_data_id,
                "label": piece_record.piece_label,
                "is_annotated": piece_record.is_annotated,
                "is_yolo_trained": piece_record.is_yolo_trained,
                "nbre_img": piece_record.nbre_img,
                "created_at": piece_record.created_at.isoformat() if piece_record.created_at else None,
                "images": []
            }

            # Fetch images with their annotation counts
            images = db.query(PieceImage).filter(PieceImage.piece_id == piece_record.id).all()
            
            total_annotations = 0
            for image in images:
                annotation_count = db.query(Annotation).filter(
                    Annotation.piece_image_id == image.id
                ).count()
                
                image_data = {
                    "id": image.id,
                    "file_name": image.file_name,
                    "image_path": image.image_path,
                    "is_annotated": image.is_annotated,
                    "annotation_count": annotation_count,
                    # Fixed: Use upload_date instead of created_at
                    "created_at": image.upload_date.isoformat() if image.upload_date else None
                }
                piece_data["images"].append(image_data)
                total_annotations += annotation_count
            
            piece_data["total_annotations"] = total_annotations
            piece_data["group"] = piece_record.piece_label.split(".")[0] if "." in piece_record.piece_label else "Other"
            
            datasets.append(piece_data)
        
        # Calculate pagination info
        total_pages = (total_count + page_size - 1) // page_size
        
        return {
            "data": datasets,
            "pagination": {
                "current_page": page,
                "page_size": page_size,
                "total_count": total_count,
                "total_pages": total_pages,
                "has_next": page < total_pages,
                "has_previous": page > 1
            },
            "filters_applied": {
                "search": search,
                "status_filter": status_filter,
                "training_filter": training_filter,
                "group_filter": group_filter,
                "date_from": date_from,
                "date_to": date_to,
                "min_images": min_images,
                "max_images": max_images,
                "sort_by": sort_by,
                "sort_order": sort_order
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching datasets with filters: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching datasets: {str(e)}")


def get_dataset_statistics(db: Session) -> Dict[str, Any]:
    """
    Get comprehensive statistics about the datasets.
    """
    try:
        # Basic counts
        total_pieces = db.query(Piece).count()
        annotated_pieces = db.query(Piece).filter(Piece.is_annotated == True).count()
        trained_pieces = db.query(Piece).filter(Piece.is_yolo_trained == True).count()
        total_images = db.query(PieceImage).count()
        total_annotations = db.query(Annotation).count()
        
        # Group statistics - Fixed PostgreSQL syntax
        # Use SPLIT_PART function which is PostgreSQL specific
        group_stats = db.query(
            func.split_part(Piece.piece_label, '.', 1).label('group'),
            func.count(Piece.id).label('count')
        ).group_by(func.split_part(Piece.piece_label, '.', 1)).all()
        
        # Recent activity (pieces created in last 7 days)
        seven_days_ago = datetime.now() - timedelta(days=7)
        recent_pieces = db.query(Piece).filter(
            Piece.created_at >= seven_days_ago
        ).count()
        
        # Average images per piece
        avg_images = db.query(func.avg(Piece.nbre_img)).scalar() or 0
        
        # Annotation completion rate
        annotation_rate = (annotated_pieces / total_pieces * 100) if total_pieces > 0 else 0
        training_rate = (trained_pieces / total_pieces * 100) if total_pieces > 0 else 0
        
        return {
            "overview": {
                "total_pieces": total_pieces,
                "total_images": total_images,
                "total_annotations": total_annotations,
                "annotated_pieces": annotated_pieces,
                "trained_pieces": trained_pieces,
                "recent_pieces": recent_pieces,
                "avg_images_per_piece": round(avg_images, 2),
                "annotation_completion_rate": round(annotation_rate, 2),
                "training_completion_rate": round(training_rate, 2)
            },
            "groups": [{"group": group, "count": count} for group, count in group_stats],
            "status_distribution": {
                "annotated": annotated_pieces,
                "not_annotated": total_pieces - annotated_pieces,
                "trained": trained_pieces,
                "not_trained": total_pieces - trained_pieces
            }
        }
        
    except Exception as e:
        logger.error(f"Error fetching dataset statistics: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching statistics: {str(e)}")


def get_available_groups(db: Session) -> List[str]:
    """
    Get all available groups from piece labels.
    """
    try:
        pieces = db.query(Piece.piece_label).all()
        groups = set()
        
        for piece_record in pieces:
            if "." in piece_record.piece_label:
                group = piece_record.piece_label.split(".")[0]
                groups.add(group)
        
        return sorted(list(groups))
        
    except Exception as e:
        logger.error(f"Error fetching available groups: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error fetching groups: {str(e)}")


def export_dataset_report(db: Session, format_type: str = "json") -> Dict[str, Any]:
    """
    Export dataset information in various formats.
    """
    try:
        datasets = get_all_datasets(db)
        statistics = get_dataset_statistics(db)
        
        report = {
            "generated_at": datetime.now().isoformat(),
            "statistics": statistics,
            "datasets": datasets,
            "format": format_type
        }
        
        return report
        
    except Exception as e:
        logger.error(f"Error generating dataset report: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error generating report: {str(e)}")


def bulk_update_pieces(db: Session, piece_ids: List[int], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Bulk update multiple pieces with given updates.
    """
    try:
        # Validate piece IDs
        pieces = db.query(Piece).filter(Piece.id.in_(piece_ids)).all()
        
        if len(pieces) != len(piece_ids):
            found_ids = [p.id for p in pieces]
            missing_ids = [pid for pid in piece_ids if pid not in found_ids]
            raise HTTPException(
                status_code=404, 
                detail=f"Pieces not found: {missing_ids}"
            )
        
        # Apply updates
        updated_count = 0
        for piece_record in pieces:
            for field, value in updates.items():
                if hasattr(piece_record, field):
                    setattr(piece_record, field, value)
                    updated_count += 1
        
        db.commit()
        
        return {
            "status": "success",
            "updated_pieces": len(pieces),
            "updated_fields": updated_count,
            "piece_labels": [p.piece_label for p in pieces]
        }
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error in bulk update: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in bulk update: {str(e)}")


# Keep all existing functions unchanged
def get_all_datasets(db: Session) -> Dict[str, Any]:
    """
    Original function - kept for backward compatibility
    """
    datasets = {}
    pieces = db.query(Piece).all()

    for piece_record in pieces:
        piece_data = {
            "id": piece_record.id,
            "class_data_id": piece_record.class_data_id,
            "label": piece_record.piece_label,
            "is_annotated": piece_record.is_annotated,
            "is_yolo_trained": piece_record.is_yolo_trained,
            "nbre_img": piece_record.nbre_img,
            "images": []
        }

        images = db.query(PieceImage).filter(PieceImage.piece_id == piece_record.id).all()

        for image in images:
            image_data = {
                "id": image.id,
                "file_name": image.file_name,
                "image_path": image.image_path,
                "is_annotated": image.is_annotated,
                "annotations": []
            }

            annotations = db.query(Annotation).filter(Annotation.piece_image_id == image.id).all()

            for annotation in annotations:
                annotation_data = {
                    "id": annotation.id,
                    "type": annotation.type,
                    "x": annotation.x,
                    "y": annotation.y,
                    "width": annotation.width,
                    "height": annotation.height
                }
                image_data["annotations"].append(annotation_data)

            piece_data["images"].append(image_data)

        datasets[piece_record.piece_label] = piece_data

    return datasets


def get_piece_labels_by_group(group_label: str, db: Session) -> List[str]:
    """
    Get piece labels by group.
    """
    try:
        pieces = db.query(Piece).filter(Piece.piece_label.like(f'{group_label}%')).all()
        piece_labels = [piece_record.piece_label for piece_record in pieces]
        
        if not piece_labels:
            logger.info(f"No pieces found for group '{group_label}'.")
        
        return piece_labels

    except Exception as e:
        logger.error(f"An error occurred while fetching piece labels: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching piece labels: {e}")

def get_piece_group_by_label(piece_label: str, db: Session) -> Optional[str]:
    """
    Extract the group from a piece label and return it.
    Assumes group is the prefix before the first dot or a specific pattern.
    """
    try:
        piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
        if not piece:
            return None
        
        # Extract group from piece_label (assuming group is before first dot)
        # Adjust this logic based on your actual group naming convention
        if '.' in piece_label:
            group = piece_label.split('.')[0]
        else:
            # If no dot, maybe the whole label is the group or use another pattern
            group = piece_label
        
        return group
        
    except SQLAlchemyError as e:
        raise Exception(f"Database error getting piece group: {str(e)}")


def delete_directory(directory_path: str) -> None:
    """Recursively delete a directory and its contents with better error handling."""
    try:
        if os.path.exists(directory_path):
            # Check if it's actually a directory
            if os.path.isdir(directory_path):
                shutil.rmtree(directory_path)
                logger.info(f"Deleted directory: {directory_path}")
            else:
                logger.warning(f"Path exists but is not a directory: {directory_path}")
        else:
            logger.info(f"Directory not found (may already be deleted): {directory_path}")
    except PermissionError as e:
        logger.error(f"Permission denied when deleting {directory_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error deleting directory {directory_path}: {e}")
        raise




def _call_annotation_service_delete_piece(piece_label: str) -> bool:
    """
    Call the annotation service to delete all annotations for a piece.
    """
    try:
        response = requests.delete(
            f"http://annotation:8000/annotations/piece/{piece_label}",
            timeout=30
        )
        
        if response.status_code == 200:
            print(f"Successfully deleted annotations for piece {piece_label}")
            return True
        elif response.status_code == 404:
            print(f"No annotations found for piece {piece_label}")
            return True
        else:
            print(f"Failed to delete annotations for piece {piece_label}: {response.status_code}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Error calling annotation service for piece {piece_label}: {str(e)}")
        return False


# def delete_piece_by_label(piece_label: str, db: Session) -> Dict[str, Any]:
#     """
#     Delete a piece, its images, and related folders.
#     """
#     try:
#         piece_to_delete = db.query(Piece).filter(Piece.piece_label == piece_label).first()
#         if not piece_to_delete:
#             raise HTTPException(status_code=404, detail="Piece not found in delete_piece_by_label")

#         annotation_deletion_success = _call_annotation_service_delete_piece(piece_label)
#         if not annotation_deletion_success:
#             print(f"Warning: Failed to delete annotations for piece {piece_label}")

#         images = db.query(PieceImage).filter(PieceImage.piece_id == piece_to_delete.id).all()
        
#         for image in images:
#             db.delete(image)
        
#         db.delete(piece_to_delete)
#         db.commit()

#         match = re.match(r'([A-Z]\d{3}\.\d{5})', piece_label)
#         if not match:
#             raise HTTPException(status_code=400, detail="Invalid piece_label format.")
        
#         extracted_label = match.group(1)
        
#         folder_paths = [
#             os.path.join("app","shared","dataset", 'piece', 'piece', extracted_label, piece_label),
#             os.path.join("app","shared","dataset", 'dataset_custom', extracted_label, 'labels', 'valid', piece_label),
#             os.path.join("app","shared","dataset", 'dataset_custom', extracted_label, 'images', 'valid', piece_label),
#             os.path.join("app","shared","dataset", 'dataset_custom', extracted_label, 'labels', 'train', piece_label),
#             os.path.join("app","shared","dataset", 'dataset_custom', extracted_label, 'images', 'train', piece_label),
#             os.path.join("app","shared","dataset", 'dataset_custom', f"{extracted_label}_cropped", 'labels', 'valid', piece_label),
#             os.path.join("app","shared","dataset", 'dataset_custom', f"{extracted_label}_cropped", 'images', 'valid', piece_label),
#             os.path.join("app","shared","dataset", 'dataset_custom', f"{extracted_label}_cropped", 'labels', 'train', piece_label),
#             os.path.join("app","shared","dataset", 'dataset_custom', f"{extracted_label}_cropped", 'images', 'train', piece_label)           
#         ]
        
#         for folder_path in folder_paths:
#             delete_directory(folder_path)

                
#         return {
#             "status": "Piece and associated data deleted successfully",
#             "piece_label": piece_label,
#             "annotations_deleted": annotation_deletion_success,
#         }
        
#     except HTTPException:
#         db.rollback()
#         raise
#     except Exception as e:
#         db.rollback()
#         error_msg = str(e)
#         print(f"Error deleting piece {piece_label}: {error_msg}")
#         raise HTTPException(status_code=500, detail=f"Error deleting piece! :( : {error_msg}")

def delete_all_pieces(db: Session) -> Dict[str, Any]:
    """
    Delete all pieces and their associated data.
    """
    try:
        pieces = db.query(Piece).all()
        
        if not pieces:
            raise HTTPException(status_code=404, detail="No pieces found to delete")

        deleted_pieces = []
        failed_pieces = []

        for piece_record in pieces:
            try:
                annotation_deletion_success = _call_annotation_service_delete_piece(piece_record.piece_label)
                
                images = db.query(PieceImage).filter(PieceImage.piece_id == piece_record.id).all()

                for image in images:
                    db.delete(image)
                
                db.delete(piece_record)
                
                deleted_pieces.append({
                    "piece_label": piece_record.piece_label,
                    "annotations_deleted": annotation_deletion_success
                })
                
            except Exception as e:
                print(f"Error deleting piece {piece_record.piece_label}: {str(e)}")
                failed_pieces.append({
                    "piece_label": piece_record.piece_label,
                    "error": str(e)
                })

        folder_path = os.path.join("dataset", "piece")
        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)
            print(f"Folder {folder_path} deleted successfully.")
        
        unified_folder_path = os.path.join("dataset", "piece")
        if os.path.exists(unified_folder_path):
            shutil.rmtree(unified_folder_path)
            print(f"Unified folder {unified_folder_path} deleted successfully.")

        db.commit()
        
        return {
            "status": "Piece deletion completed",
            "deleted_pieces": deleted_pieces,
            "failed_pieces": failed_pieces,
            "total_deleted": len(deleted_pieces),
            "total_failed": len(failed_pieces)
        }
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        error_msg = str(e)
        print(f"Error deleting all pieces: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Error deleting all pieces: {error_msg}")


def get_piece_annotations_via_api(piece_label: str) -> Dict[str, Any]:
    """
    Get annotations for a piece via the annotation service API.
    """
    try:
        response = requests.get(
            f"http://annotation:8000/annotations/piece/{piece_label}",
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return {
                "piece_label": piece_label,
                "images": [],
                "message": "No annotations found for this piece"
            }
        else:
            raise HTTPException(
                status_code=response.status_code, 
                detail=f"Failed to get annotations: {response.text}"
            )
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503, 
            detail=f"Annotation service unavailable: {str(e)}"
        )


def delete_annotation_via_api(annotation_id: int) -> Dict[str, Any]:
    """
    Delete a specific annotation via the annotation service API.
    """
    try:
        response = requests.delete(
            f"http://annotation:8000/annotations/{annotation_id}",
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            raise HTTPException(status_code=404, detail="Annotation not found")
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to delete annotation: {response.text}"
            )
            
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=503,
            detail=f"Annotation service unavailable: {str(e)}"
        )
def update_group_training_status(group: str, db: Session) -> dict:
    """
    Update the is_yolo_trained status to False for all pieces in a specific group.
    This function handles the training status update logic separately from deletion.
    """
    try:
        # Find all pieces in the group (adjust the filter based on your group logic)
        if group:
            # Assuming group is a prefix of the piece_label
            pieces_in_group = db.query(Piece).filter(
                Piece.piece_label.like(f"{group}.%")
            ).all()
            
            # Alternative: if group is stored in a separate field
            # pieces_in_group = db.query(Piece).filter(Piece.group == group).all()
            
            updated_count = 0
            for piece in pieces_in_group:
                if piece.is_yolo_trained:  # Only update if currently True
                    piece.is_yolo_trained = False
                    updated_count += 1
            
            db.commit()
            
            return {
                "group": group,
                "total_pieces_in_group": len(pieces_in_group),
                "pieces_updated": updated_count
            }
        else:
            return {"group": None, "total_pieces_in_group": 0, "pieces_updated": 0}
            
    except SQLAlchemyError as e:
        db.rollback()
        raise Exception(f"Database error updating group training status: {str(e)}")
def get_pieces_by_group(group: str, db: Session) -> List[dict]:
    """
    Helper function to get all pieces belonging to a specific group.
    """
    try:
        pieces = db.query(Piece).filter(
            Piece.piece_label.like(f"{group}.%")
        ).all()
        
        return [
            {
                "id": piece.id,
                "label": piece.piece_label,
                "is_yolo_trained": piece.is_yolo_trained,
                "is_annotated": piece.is_annotated
            }
            for piece in pieces
        ]
        
    except SQLAlchemyError as e:
        raise Exception(f"Database error getting pieces by group: {str(e)}")

def delete_pieces_batch(piece_labels: List[str], db: Session) -> Dict[str, Any]:
    """
    Delete multiple pieces by their labels with corrected file paths
    """
    if not piece_labels:
        raise HTTPException(status_code=400, detail="piece_labels list cannot be empty")
    
    if len(piece_labels) > 100:
        raise HTTPException(status_code=400, detail="Cannot delete more than 100 pieces at once.")
    
    try:
        logger.info(f"Starting batch delete for {len(piece_labels)} pieces: {piece_labels}")
        
        # Get the dataset base path from environment variable
        dataset_base_path = os.getenv('DATASET_BASE_PATH', '/app/shared/dataset')
        logger.info(f"Using dataset base path: {dataset_base_path}")
        
        # Step 1: Validate all pieces exist before starting deletion
        pieces_to_delete = []
        not_found_labels = []
        
        for piece_label in piece_labels:
            logger.info(f"Validating piece: {piece_label}")
            try:
                piece_to_delete = db.query(Piece).filter(Piece.piece_label == piece_label).first()
                if not piece_to_delete:
                    logger.warning(f"Piece not found: {piece_label}")
                    not_found_labels.append(piece_label)
                else:
                    logger.info(f"Found piece: {piece_label} (ID: {piece_to_delete.id})")
                    pieces_to_delete.append(piece_label)
            except Exception as validation_error:
                logger.error(f"Error validating piece {piece_label}: {type(validation_error).__name__} - {validation_error}")
                not_found_labels.append(piece_label)
        
        if not_found_labels:
            error_msg = f"Pieces not found: {not_found_labels}"
            logger.error(error_msg)
            raise HTTPException(status_code=404, detail=error_msg)
        
        logger.info(f"Validation complete. {len(pieces_to_delete)} pieces will be deleted")
        
        # Step 2: Delete each piece
        deleted_pieces = []
        failed_pieces = []
        
        for piece_label in pieces_to_delete:
            logger.info(f"Processing piece: {piece_label}")
            
            try:
                piece_to_delete = db.query(Piece).filter(Piece.piece_label == piece_label).first()
                if not piece_to_delete:
                    failed_pieces.append({
                        "piece_label": piece_label,
                        "error": "Piece not found during deletion",
                        "error_type": "NotFound"
                    })
                    continue

                # Delete annotations via API
                annotation_deletion_success = _call_annotation_service_delete_piece(piece_label)
                
                # Delete images from database
                images = db.query(PieceImage).filter(PieceImage.piece_id == piece_to_delete.id).all()
                for image in images:
                    db.delete(image)
                
                # Delete the piece from database
                db.delete(piece_to_delete)
                db.commit()

                # Delete file system folders - CORRECTED PATHS
                match = re.match(r'([A-Z]\d{3}\.\d{5})', piece_label)
                if match:
                    extracted_label = match.group(1)
                    logger.info(f"Extracted label: {extracted_label} from piece: {piece_label}")
                    
                    # List all possible folder paths that need to be deleted
                    folder_paths = [
                        # Original piece folder (check both possible structures)
                        os.path.join(dataset_base_path, 'piece', 'piece', extracted_label, piece_label),
                        os.path.join(dataset_base_path, 'piece', extracted_label, piece_label),
                        
                        # Dataset custom folders for the specific piece
                        os.path.join(dataset_base_path, 'dataset_custom', extracted_label, 'labels', 'valid', piece_label),
                        os.path.join(dataset_base_path, 'dataset_custom', extracted_label, 'images', 'valid', piece_label),
                        os.path.join(dataset_base_path, 'dataset_custom', extracted_label, 'labels', 'train', piece_label),
                        os.path.join(dataset_base_path, 'dataset_custom', extracted_label, 'images', 'train', piece_label),
                        
                        # Cropped dataset folders
                        os.path.join(dataset_base_path, 'dataset_custom', f"{extracted_label}_cropped", 'labels', 'valid', piece_label),
                        os.path.join(dataset_base_path, 'dataset_custom', f"{extracted_label}_cropped", 'images', 'valid', piece_label),
                        os.path.join(dataset_base_path, 'dataset_custom', f"{extracted_label}_cropped", 'labels', 'train', piece_label),
                        os.path.join(dataset_base_path, 'dataset_custom', f"{extracted_label}_cropped", 'images', 'train', piece_label),
                        
                        # Additional possible locations (add more if you know other patterns)
                        os.path.join(dataset_base_path, 'images', extracted_label, piece_label),
                        os.path.join(dataset_base_path, 'labels', extracted_label, piece_label),
                    ]
                    
                    deleted_folders = []
                    not_found_folders = []
                    failed_folders = []
                    
                    # Log and delete each folder
                    for folder_path in folder_paths:
                        logger.info(f"Checking folder: {folder_path}")
                        if os.path.exists(folder_path):
                            try:
                                delete_directory(folder_path)
                                deleted_folders.append(folder_path)
                                logger.info(f"Successfully deleted folder: {folder_path}")
                            except Exception as folder_error:
                                failed_folders.append(folder_path)
                                logger.error(f"Failed to delete folder {folder_path}: {folder_error}")
                        else:
                            not_found_folders.append(folder_path)
                            logger.debug(f"Folder does not exist: {folder_path}")
                    
                    # Also check for any directories that might contain the piece_label directly
                    # This is a more thorough search in case the structure is different
                    try:
                        for root, dirs, files in os.walk(dataset_base_path):
                            if piece_label in dirs:
                                piece_dir = os.path.join(root, piece_label)
                                logger.info(f"Found additional piece directory: {piece_dir}")
                                if piece_dir not in [fp for fp in folder_paths]:  # Avoid duplicates
                                    try:
                                        delete_directory(piece_dir)
                                        deleted_folders.append(piece_dir)
                                        logger.info(f"Successfully deleted additional folder: {piece_dir}")
                                    except Exception as folder_error:
                                        failed_folders.append(piece_dir)
                                        logger.error(f"Failed to delete additional folder {piece_dir}: {folder_error}")
                    except Exception as walk_error:
                        logger.warning(f"Error during directory walk: {walk_error}")
                    
                    logger.info(f"Folder deletion summary for {piece_label}:")
                    logger.info(f"  - Deleted: {len(deleted_folders)} folders")
                    logger.info(f"  - Not found: {len(not_found_folders)} folders")
                    logger.info(f"  - Failed: {len(failed_folders)} folders")
                    
                    if failed_folders:
                        logger.warning(f"Some folders could not be deleted for {piece_label}: {failed_folders}")
                else:
                    logger.warning(f"Could not extract label pattern from: {piece_label}")
                
                # Record success
                deleted_pieces.append({
                    "piece_label": piece_label,
                    "annotations_deleted": annotation_deletion_success,
                    "status": "deleted"
                })
                logger.info(f"Successfully deleted piece: {piece_label}")
                
            except Exception as e:
                db.rollback()  # Rollback on individual piece failure
                error_msg = str(e) if str(e) else f"Unknown {type(e).__name__} error"
                logger.error(f"Exception for piece {piece_label}: {type(e).__name__} - {error_msg}")
                failed_pieces.append({
                    "piece_label": piece_label,
                    "error": error_msg,
                    "error_type": type(e).__name__
                })
                continue
        
        result = {
            "status": "Batch piece deletion completed",
            "deleted_pieces": deleted_pieces,
            "failed_pieces": failed_pieces,
            "total_deleted": len(deleted_pieces),
            "total_failed": len(failed_pieces)
        }
        
        logger.info(f"Batch delete completed: {len(deleted_pieces)} deleted, {len(failed_pieces)} failed")
        return result
        
    except HTTPException as he:
        logger.error(f"HTTPException in batch delete: {he.status_code} - {he.detail}")
        raise
    except Exception as e:
        db.rollback()
        error_msg = str(e) if str(e) else f"Unknown {type(e).__name__} error"
        full_error = f"Error in batch piece deletion: {error_msg}"
        
        logger.error(f"Critical error in batch delete: {type(e).__name__} - {error_msg}")
        
        raise HTTPException(status_code=500, detail=full_error)


def delete_directory(directory_path: str) -> None:
    """Recursively delete a directory and its contents with better error handling."""
    try:
        if os.path.exists(directory_path):
            # Check if it's actually a directory
            if os.path.isdir(directory_path):
                shutil.rmtree(directory_path)
                logger.info(f"Deleted directory: {directory_path}")
            else:
                logger.warning(f"Path exists but is not a directory: {directory_path}")
        else:
            logger.info(f"Directory not found (may already be deleted): {directory_path}")
    except PermissionError as e:
        logger.error(f"Permission denied when deleting {directory_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Error deleting directory {directory_path}: {e}")
        raise

