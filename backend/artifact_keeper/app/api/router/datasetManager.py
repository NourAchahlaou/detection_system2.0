from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Annotated, List, Optional
from datetime import datetime

from artifact_keeper.app.services.datasetManagerService import (
    get_all_datasets,
    get_all_datasets_with_filters,
    get_dataset_statistics,
    get_available_groups,
    export_dataset_report,
    bulk_update_pieces,
    delete_all_pieces,
    delete_piece_by_label,
    get_piece_labels_by_group,
    get_piece_annotations_via_api,
    delete_annotation_via_api
)
from artifact_keeper.app.db.session import get_session

datasetManager_router = APIRouter(
    prefix="/datasetManager",
    tags=["Dataset Manager"],
    responses={404: {"description": "Not found"}},
)

db_dependency = Annotated[Session, Depends(get_session)]


@datasetManager_router.get("/datasets", tags=["Dataset"])
def get_datasets_route(db: Session = Depends(get_session)):
    """Route to fetch all datasets (legacy endpoint)."""
    datasets = get_all_datasets(db)
    if not datasets:
        raise HTTPException(status_code=404, detail="No datasets found")
    return datasets


@datasetManager_router.get("/datasets/enhanced", tags=["Dataset"])
def get_datasets_enhanced_route(
    db: db_dependency,
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search term for piece labels"),
    status_filter: Optional[str] = Query(None, description="Filter by annotation status: annotated, not_annotated"),
    training_filter: Optional[str] = Query(None, description="Filter by training status: trained, not_trained"),
    group_filter: Optional[str] = Query(None, description="Filter by group prefix"),
    sort_by: str = Query("created_at", description="Sort by field: created_at, piece_label, nbre_img, is_annotated, is_yolo_trained"),
    sort_order: str = Query("desc", description="Sort order: asc, desc"),
    date_from: Optional[str] = Query(None, description="Filter from date (ISO format)"),
    date_to: Optional[str] = Query(None, description="Filter to date (ISO format)"),
    min_images: Optional[int] = Query(None, ge=0, description="Minimum number of images"),
    max_images: Optional[int] = Query(None, ge=0, description="Maximum number of images")
):
    """Enhanced route to fetch datasets with filtering, searching, sorting, and pagination."""
    
    # Validate sort_by parameter
    valid_sort_fields = ["created_at", "piece_label", "nbre_img", "is_annotated", "is_yolo_trained", "updated_at"]
    if sort_by not in valid_sort_fields:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid sort_by field. Must be one of: {', '.join(valid_sort_fields)}"
        )
    
    # Validate sort_order parameter
    if sort_order.lower() not in ["asc", "desc"]:
        raise HTTPException(status_code=400, detail="sort_order must be 'asc' or 'desc'")
    
    # Validate status_filter parameter
    if status_filter and status_filter not in ["annotated", "not_annotated"]:
        raise HTTPException(
            status_code=400, 
            detail="status_filter must be 'annotated' or 'not_annotated'"
        )
    
    # Validate training_filter parameter
    if training_filter and training_filter not in ["trained", "not_trained"]:
        raise HTTPException(
            status_code=400, 
            detail="training_filter must be 'trained' or 'not_trained'"
        )
    
    # Validate date formats
    if date_from:
        try:
            datetime.fromisoformat(date_from.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date_from format. Use ISO format.")
    
    if date_to:
        try:
            datetime.fromisoformat(date_to.replace('Z', '+00:00'))
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid date_to format. Use ISO format.")
    
    # Validate image range
    if min_images is not None and max_images is not None and min_images > max_images:
        raise HTTPException(status_code=400, detail="min_images cannot be greater than max_images")
    
    return get_all_datasets_with_filters(
        db=db,
        page=page,
        page_size=page_size,
        search=search,
        status_filter=status_filter,
        training_filter=training_filter,
        group_filter=group_filter,
        sort_by=sort_by,
        sort_order=sort_order,
        date_from=date_from,
        date_to=date_to,
        min_images=min_images,
        max_images=max_images
    )


@datasetManager_router.get("/statistics", tags=["Dataset"])
def get_statistics_route(db: db_dependency):
    """Route to get comprehensive dataset statistics."""
    return get_dataset_statistics(db)


@datasetManager_router.get("/groups", tags=["Dataset"])
def get_groups_route(db: db_dependency):
    """Route to get all available groups."""
    return {"groups": get_available_groups(db)}


@datasetManager_router.get("/export", tags=["Dataset"])
def export_dataset_route(
    db: db_dependency,
    format_type: str = Query("json", description="Export format: json, csv")
):
    """Route to export dataset report."""
    if format_type not in ["json", "csv"]:
        raise HTTPException(status_code=400, detail="format_type must be 'json' or 'csv'")
    
    return export_dataset_report(db, format_type)


@datasetManager_router.patch("/pieces/bulk-update", tags=["Dataset"])
def bulk_update_pieces_route(
    piece_ids: List[int],
    updates: dict,
    db: db_dependency
):
    """Route to bulk update multiple pieces."""
    if not piece_ids:
        raise HTTPException(status_code=400, detail="piece_ids cannot be empty")
    
    if not updates:
        raise HTTPException(status_code=400, detail="updates cannot be empty")
    
    # Validate update fields
    valid_fields = ["is_annotated", "is_yolo_trained", "piece_label", "class_data_id"]
    invalid_fields = [field for field in updates.keys() if field not in valid_fields]
    if invalid_fields:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid update fields: {', '.join(invalid_fields)}. Valid fields: {', '.join(valid_fields)}"
        )
    
    return bulk_update_pieces(db, piece_ids, updates)


@datasetManager_router.get("/pieces/group/{group_label}", tags=["Dataset"])
def get_piece_labels_by_group_route(group_label: str, db: db_dependency) -> List[str]:
    """Route to get piece labels by group."""
    return get_piece_labels_by_group(group_label, db)


@datasetManager_router.get("/pieces/{piece_label}/annotations", tags=["Annotations"])
def get_piece_annotations_route(piece_label: str):
    """Route to get annotations for a specific piece via annotation service API."""
    return get_piece_annotations_via_api(piece_label)


@datasetManager_router.delete("/pieces/{piece_label}", tags=["Dataset"])
def delete_piece_route(piece_label: str, db: db_dependency):
    """Route to delete a specific piece and its associated data."""
    return delete_piece_by_label(piece_label, db)


@datasetManager_router.delete("/pieces", tags=["Dataset"])
def delete_all_pieces_route(db: db_dependency):
    """Route to delete all pieces and their associated data."""
    return delete_all_pieces(db)


@datasetManager_router.delete("/annotations/{annotation_id}", tags=["Annotations"])
def delete_annotation_route(annotation_id: int):
    """Route to delete a specific annotation via annotation service API."""
    return delete_annotation_via_api(annotation_id)