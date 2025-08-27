import os
import re
import shutil
from sqlalchemy.exc import SQLAlchemyError
from fastapi import HTTPException
    
import requests
from typing import Dict, Any, List
from sqlalchemy.orm import Session
from artifact_keeper.app.db.models.piece import Piece
from artifact_keeper.app.db.models.piece_image import PieceImage

def delete_pieces_batch_isolated(piece_labels: List[str], db: Session) -> Dict[str, Any]:
    """
    COMPLETELY ISOLATED batch deletion function that doesn't call ANY other functions
    from this module to avoid any circular or unexpected calls.
    """

    # Set up logging for this specific function
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info(f"🟢 ISOLATED_BATCH: Starting isolated batch delete for {len(piece_labels)} pieces")
    
    if not piece_labels:
        raise HTTPException(status_code=400, detail="ISOLATED_BATCH: piece_labels list cannot be empty")
    
    if len(piece_labels) > 100:
        raise HTTPException(status_code=400, detail="ISOLATED_BATCH: Cannot delete more than 100 pieces at once.")
    
    # Results tracking
    deleted_pieces = []
    failed_pieces = []
    
    # Process each piece individually with complete isolation
    for piece_label in piece_labels:
        logger.info(f"🟢 ISOLATED_BATCH: Processing piece: {piece_label}")
        
        try:
            # Step 1: Find the piece in database
            logger.info(f"🟢 ISOLATED_BATCH: Querying database for piece: {piece_label}")
            piece_to_delete = db.query(Piece).filter(Piece.piece_label == piece_label).first()
            
            if not piece_to_delete:
                logger.error(f"🟢 ISOLATED_BATCH: Piece not found: {piece_label}")
                failed_pieces.append({
                    "piece_label": piece_label,
                    "error": f"ISOLATED_BATCH: Piece not found in database: {piece_label}",
                    "error_type": "NotFound"
                })
                continue
            
            logger.info(f"🟢 ISOLATED_BATCH: Found piece {piece_label} with ID {piece_to_delete.id}")
            
            # Step 2: Delete annotations via API
            annotation_deletion_success = False
            try:
                logger.info(f"🟢 ISOLATED_BATCH: Deleting annotations for piece: {piece_label}")
                response = requests.delete(
                    f"http://annotation:8000/annotations/piece/{piece_label}",
                    timeout=30
                )
                
                if response.status_code in [200, 404]:
                    annotation_deletion_success = True
                    logger.info(f"🟢 ISOLATED_BATCH: Successfully handled annotations for piece {piece_label} (status: {response.status_code})")
                else:
                    logger.warning(f"🟢 ISOLATED_BATCH: Unexpected annotation service response for piece {piece_label}: {response.status_code}")
                    annotation_deletion_success = True  # Continue anyway
                    
            except requests.exceptions.RequestException as req_error:
                logger.warning(f"🟢 ISOLATED_BATCH: Annotation service error for piece {piece_label}: {req_error}")
                annotation_deletion_success = False  # Continue anyway
            
            # Step 3: Delete piece images from database
            logger.info(f"🟢 ISOLATED_BATCH: Deleting images for piece: {piece_label}")
            try:
                images = db.query(PieceImage).filter(PieceImage.piece_id == piece_to_delete.id).all()
                logger.info(f"🟢 ISOLATED_BATCH: Found {len(images)} images for piece {piece_label}")
                
                for image in images:
                    logger.info(f"🟢 ISOLATED_BATCH: Deleting image {image.id} for piece {piece_label}")
                    db.delete(image)
                    
                logger.info(f"🟢 ISOLATED_BATCH: Deleted all images for piece {piece_label}")
                
            except SQLAlchemyError as sql_error:
                logger.error(f"🟢 ISOLATED_BATCH: Database error deleting images for piece {piece_label}: {sql_error}")
                db.rollback()
                failed_pieces.append({
                    "piece_label": piece_label,
                    "error": f"ISOLATED_BATCH: Database error deleting images: {str(sql_error)}",
                    "error_type": "DatabaseError"
                })
                continue
            
            # Step 4: Delete piece from database
            logger.info(f"🟢 ISOLATED_BATCH: Deleting piece from database: {piece_label}")
            try:
                db.delete(piece_to_delete)
                db.commit()
                logger.info(f"🟢 ISOLATED_BATCH: Successfully deleted piece from database: {piece_label}")
                
            except SQLAlchemyError as sql_error:
                logger.error(f"🟢 ISOLATED_BATCH: Database error deleting piece {piece_label}: {sql_error}")
                db.rollback()
                failed_pieces.append({
                    "piece_label": piece_label,
                    "error": f"ISOLATED_BATCH: Database error deleting piece: {str(sql_error)}",
                    "error_type": "DatabaseError"
                })
                continue
            
            # Step 5: Delete file system directories
            logger.info(f"🟢 ISOLATED_BATCH: Deleting file system directories for piece: {piece_label}")
            try:
                # Extract the group label using regex
                match = re.match(r'([A-Z]\d{3}\.\d{5})', piece_label)
                if match:
                    extracted_label = match.group(1)
                    logger.info(f"🟢 ISOLATED_BATCH: Extracted label: {extracted_label}")
                    
                    # Define all possible folder paths
                    folder_paths = [
                        os.path.join("app", "shared", "dataset", 'piece', 'piece', extracted_label, piece_label),
                        os.path.join("app", "shared", "dataset", 'dataset_custom', extracted_label, 'labels', 'valid', piece_label),
                        os.path.join("app", "shared", "dataset", 'dataset_custom', extracted_label, 'images', 'valid', piece_label),
                        os.path.join("app", "shared", "dataset", 'dataset_custom', extracted_label, 'labels', 'train', piece_label),
                        os.path.join("app", "shared", "dataset", 'dataset_custom', extracted_label, 'images', 'train', piece_label),
                        os.path.join("app", "shared", "dataset", 'dataset_custom', f"{extracted_label}_cropped", 'labels', 'valid', piece_label),
                        os.path.join("app", "shared", "dataset", 'dataset_custom', f"{extracted_label}_cropped", 'images', 'valid', piece_label),
                        os.path.join("app", "shared", "dataset", 'dataset_custom', f"{extracted_label}_cropped", 'labels', 'train', piece_label),
                        os.path.join("app", "shared", "dataset", 'dataset_custom', f"{extracted_label}_cropped", 'images', 'train', piece_label)           
                    ]
                    
                    # Delete each folder if it exists
                    deleted_folders = 0
                    for folder_path in folder_paths:
                        if os.path.exists(folder_path):
                            try:
                                shutil.rmtree(folder_path)
                                logger.info(f"🟢 ISOLATED_BATCH: Deleted directory: {folder_path}")
                                deleted_folders += 1
                            except Exception as folder_error:
                                logger.warning(f"🟢 ISOLATED_BATCH: Could not delete folder {folder_path}: {folder_error}")
                        else:
                            logger.info(f"🟢 ISOLATED_BATCH: Directory does not exist: {folder_path}")
                    
                    logger.info(f"🟢 ISOLATED_BATCH: Deleted {deleted_folders} directories for piece {piece_label}")
                
                else:
                    logger.warning(f"🟢 ISOLATED_BATCH: Invalid piece_label format for folder deletion: {piece_label}")
                
            except Exception as folder_error:
                logger.warning(f"🟢 ISOLATED_BATCH: Error during folder deletion for piece {piece_label}: {folder_error}")
                # Don't fail the entire operation for folder deletion errors
            
            # Step 6: Record successful deletion
            deleted_pieces.append({
                "piece_label": piece_label,
                "annotations_deleted": annotation_deletion_success,
                "status": "deleted_successfully"
            })
            
            logger.info(f"🟢 ISOLATED_BATCH: Successfully completed all deletion steps for piece: {piece_label}")
            
        except HTTPException as he:
            logger.error(f"🟢 ISOLATED_BATCH: HTTPException for piece {piece_label}: {he.status_code} - {he.detail}")
            failed_pieces.append({
                "piece_label": piece_label,
                "error": f"ISOLATED_BATCH: HTTP Error: {he.detail}",
                "error_type": "HTTPException"
            })
            continue
            
        except Exception as e:
            logger.error(f"🟢 ISOLATED_BATCH: Unexpected error for piece {piece_label}: {type(e).__name__} - {str(e)}")
            failed_pieces.append({
                "piece_label": piece_label,
                "error": f"ISOLATED_BATCH: Unexpected error: {str(e)}",
                "error_type": type(e).__name__
            })
            continue
    
    # Final result
    result = {
        "status": "ISOLATED_BATCH: Batch piece deletion completed",
        "deleted_pieces": deleted_pieces,
        "failed_pieces": failed_pieces,
        "total_deleted": len(deleted_pieces),
        "total_failed": len(failed_pieces),
        "total_processed": len(piece_labels)
    }
    
    logger.info(f"🟢 ISOLATED_BATCH: Batch delete completed - {len(deleted_pieces)} deleted, {len(failed_pieces)} failed")
    
    return result