import os
import re
import shutil
import requests
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from fastapi import HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, asc, and_, or_

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
                    Piece.class_data_id.ilike(search_term)
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
        for piece in pieces:
            piece_data = {
                "id": piece.id,
                "class_data_id": piece.class_data_id,
                "label": piece.piece_label,
                "is_annotated": piece.is_annotated,
                "is_yolo_trained": piece.is_yolo_trained,
                "nbre_img": piece.nbre_img,
                "created_at": piece.created_at.isoformat() if piece.created_at else None,
                "images": []
            }

            # Fetch images with their annotation counts
            images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).all()
            
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
            piece_data["group"] = piece.piece_label.split(".")[0] if "." in piece.piece_label else "Other"
            
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
        
        for piece in pieces:
            if "." in piece.piece_label:
                group = piece.piece_label.split(".")[0]
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
        for piece in pieces:
            for field, value in updates.items():
                if hasattr(piece, field):
                    setattr(piece, field, value)
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

    for piece in pieces:
        piece_data = {
            "id": piece.id,
            "class_data_id": piece.class_data_id,
            "label": piece.piece_label,
            "is_annotated": piece.is_annotated,
            "is_yolo_trained": piece.is_yolo_trained,
            "nbre_img": piece.nbre_img,
            "images": []
        }

        images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).all()

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

        datasets[piece.piece_label] = piece_data

    return datasets


# Keep all other existing functions unchanged...
def get_piece_labels_by_group(group_label: str, db: Session) -> List[str]:
    """
    Get piece labels by group.
    """
    try:
        pieces = db.query(Piece).filter(Piece.piece_label.like(f'{group_label}%')).all()
        piece_labels = [piece.piece_label for piece in pieces]
        
        if not piece_labels:
            logger.info(f"No pieces found for group '{group_label}'.")
        
        return piece_labels

    except Exception as e:
        logger.error(f"An error occurred while fetching piece labels: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching piece labels: {e}")


def delete_directory(directory_path: str) -> None:
    """Recursively delete a directory and its contents."""
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
        print(f"Deleted directory: {directory_path}")
    else:
        print(f"Directory not found: {directory_path}")


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


def delete_piece_by_label(piece_label: str, db: Session) -> Dict[str, Any]:
    """
    Delete a piece, its images, and related folders.
    """
    try:
        piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
        if not piece:
            raise HTTPException(status_code=404, detail="Piece not found")

        annotation_deletion_success = _call_annotation_service_delete_piece(piece_label)
        if not annotation_deletion_success:
            print(f"Warning: Failed to delete annotations for piece {piece_label}")

        images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).all()
        
        for image in images:
            db.delete(image)
        
        db.delete(piece)
        db.commit()

        match = re.match(r'([A-Z]\d{3}\.\d{5})', piece_label)
        if not match:
            raise HTTPException(status_code=400, detail="Invalid piece_label format.")
        
        extracted_label = match.group(1)
        
        folder_paths = [
            os.path.join("dataset", 'Pieces', 'Pieces', 'labels', 'valid', extracted_label, piece_label),
            os.path.join("dataset", 'Pieces', 'Pieces', 'images', 'valid', extracted_label, piece_label),
            os.path.join("dataset", 'Pieces', 'Pieces', 'labels', 'train', extracted_label, piece_label),
            os.path.join("dataset", 'Pieces', 'Pieces', 'images', 'train', extracted_label, piece_label),
            os.path.join("dataset", "piece", piece_label)
        ]
        
        for folder_path in folder_paths:
            delete_directory(folder_path)
        
        return {
            "status": "Piece and associated data deleted successfully",
            "piece_label": piece_label,
            "annotations_deleted": annotation_deletion_success
        }
        
    except HTTPException:
        db.rollback()
        raise
    except Exception as e:
        db.rollback()
        error_msg = str(e)
        print(f"Error deleting piece {piece_label}: {error_msg}")
        raise HTTPException(status_code=500, detail=f"Error deleting piece: {error_msg}")


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

        for piece in pieces:
            try:
                annotation_deletion_success = _call_annotation_service_delete_piece(piece.piece_label)
                
                images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).all()

                for image in images:
                    db.delete(image)
                
                db.delete(piece)
                
                deleted_pieces.append({
                    "piece_label": piece.piece_label,
                    "annotations_deleted": annotation_deletion_success
                })
                
            except Exception as e:
                print(f"Error deleting piece {piece.piece_label}: {str(e)}")
                failed_pieces.append({
                    "piece_label": piece.piece_label,
                    "error": str(e)
                })

        folder_path = os.path.join("dataset", "Pieces")
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