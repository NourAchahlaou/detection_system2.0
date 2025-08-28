# annotation/app/services/annotationService.py
import os
import re
import yaml
import requests
from typing import Dict, Any
from fastapi import HTTPException
from sqlalchemy.orm import Session
import shutil
from annotation.app.db.models.annotation import Annotation
from annotation.app.db.models.piece import Piece
from annotation.app.db.models.piece_image import PieceImage

import logging
logger = logging.getLogger(__name__)
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
        
        # Store the class_data_id and extracted label for later use
        class_data_id = piece.class_data_id
        match = re.match(r'([A-Z]\d{3})', piece_label)
        extracted_label = match.group(1) if match else None
        
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
        
        # Commit changes first
        db.commit()
        
        # Update data.yaml and renumber class IDs after successful deletion
        if extracted_label:
            _update_data_yaml_after_piece_deletion(extracted_label, piece_label, class_data_id, db)
        
        return {
            "status": "success",
            "message": f"Deleted {total_deleted} annotations for piece {piece_label}",
            "deleted_count": total_deleted
        }
        
    except Exception as e:
        db.rollback()
        error_msg = str(e)
        logger.error(f"Error deleting annotations for piece {piece_label}: {error_msg}")
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
        match = re.match(r'([A-Z]\d{3}\.\d{5})', piece_label)
        if match:
            extracted_label = match.group(1)
            # Use the unified dataset path for YOLO structure
            dataset_base_path = os.getenv('DATASET_BASE_PATH', '/app/shared/dataset')
            labels_folder = os.path.join(dataset_base_path, "piece", "piece", extracted_label, piece_label, "labels", "valid")
            
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
        match = re.match(r'([A-Z]\d{3}\.\d{5})', piece_label)
        if match:
            extracted_label = match.group(1)
            dataset_base_path = os.getenv('DATASET_BASE_PATH', '/app/shared/dataset')
            folder_paths = [
                # Original piece folder - FIXED TYPO
                os.path.join(dataset_base_path, 'piece', 'piece', extracted_label, piece_label, "labels"),
                
                # Dataset custom folders for the specific piece
                os.path.join(dataset_base_path, 'dataset_custom', extracted_label, 'labels', 'valid', piece_label),
                os.path.join(dataset_base_path, 'dataset_custom', extracted_label, 'labels', 'train', piece_label),
                
                # Cropped dataset folders
                os.path.join(dataset_base_path, 'dataset_custom', f"{extracted_label}_cropped", 'labels', 'valid', piece_label),
                os.path.join(dataset_base_path, 'dataset_custom', f"{extracted_label}_cropped", 'labels', 'train', piece_label),
            ]        
                        
            deleted_folders = []
            not_found_folders = []
            failed_folders = []
            
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
        
    except Exception as e:
        logger.error(f"Warning: Could not delete annotation files for piece {piece_label}: {str(e)}")


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



def _update_data_yaml_after_deletion(piece_label: str, class_data_id: int, db: Session) -> None:
    """Update data.yaml after annotation deletion"""
    try:
        match = re.match(r'([A-Z]\d{3}\.\d{5})', piece_label)
        if match:
            extracted_label = match.group(1)
        dataset_base_path = os.getenv('DATASET_BASE_PATH', '/app/shared/dataset')
        data_yaml_path = os.path.join(dataset_base_path,"piece","piece",extracted_label, "data.yaml")
        
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


def _update_data_yaml_after_piece_deletion(extracted_label: str, deleted_piece_label: str, deleted_class_id: int, db: Session) -> None:
    """
    Update data.yaml after piece deletion by removing the deleted piece and renumbering classes.
    Also update all annotation files with new class IDs.
    """
    try:
        dataset_base_path = os.getenv('DATASET_BASE_PATH', '/app/shared/dataset')
        data_yaml_path = os.path.join(dataset_base_path, "piece", "piece", extracted_label, "data.yaml")
        
        if not os.path.exists(data_yaml_path):
            logger.warning(f"data.yaml not found at {data_yaml_path}")
            return
        
        # Load existing data.yaml
        with open(data_yaml_path, 'r') as yaml_file:
            data_yaml = yaml.safe_load(yaml_file)
        
        if 'names' not in data_yaml:
            logger.warning("No 'names' section in data.yaml")
            return
        
        # Get current names mapping
        current_names = data_yaml['names']
        
        # Remove the deleted piece from names
        piece_found = False
        for class_id, piece_name in list(current_names.items()):
            if piece_name == deleted_piece_label:
                del current_names[class_id]
                piece_found = True
                logger.info(f"Removed {deleted_piece_label} (class {class_id}) from data.yaml")
                break
        
        if not piece_found:
            logger.warning(f"Piece {deleted_piece_label} not found in data.yaml names")
            return
        
        # Create new mapping with sequential numbering (0, 1, 2, ...)
        remaining_pieces = list(current_names.values())
        new_names = {i: piece for i, piece in enumerate(remaining_pieces)}
        
        # Create mapping from old class IDs to new class IDs for annotation file updates
        old_to_new_mapping = {}
        for old_id, piece_name in current_names.items():
            # Find the new ID for this piece
            for new_id, mapped_piece in new_names.items():
                if mapped_piece == piece_name:
                    old_to_new_mapping[int(old_id)] = int(new_id)
                    break
        
        # Update data.yaml
        data_yaml['names'] = new_names
        data_yaml['nc'] = len(new_names)
        
        # Write updated data.yaml
        with open(data_yaml_path, 'w') as yaml_file:
            yaml.dump(data_yaml, yaml_file, default_flow_style=False)
        
        logger.info(f"Updated data.yaml: nc={len(new_names)}, names={new_names}")
        
        # Update all annotation files with new class IDs
        _update_annotation_files_class_ids(extracted_label, old_to_new_mapping)
        
        # Update database annotations with new class IDs
        _update_database_annotations_class_ids(extracted_label, old_to_new_mapping, db)
        
    except Exception as e:
        logger.error(f"Error updating data.yaml after piece deletion: {str(e)}")


def _update_annotation_files_class_ids(extracted_label: str, old_to_new_mapping: Dict[int, int]) -> None:
    """
    Update all annotation .txt files to use new class IDs after a piece deletion.
    """
    try:
        dataset_base_path = os.getenv('DATASET_BASE_PATH', '/app/shared/dataset')
        
        # Define all possible paths where annotation files might exist
        search_paths = [
            os.path.join(dataset_base_path, 'piece', 'piece', extracted_label),
            os.path.join(dataset_base_path, 'dataset_custom', extracted_label),
            os.path.join(dataset_base_path, 'dataset_custom', f"{extracted_label}_cropped"),
        ]
        
        updated_files = []
        
        for base_path in search_paths:
            if not os.path.exists(base_path):
                continue
                
            # Walk through all subdirectories to find .txt files
            for root, dirs, files in os.walk(base_path):
                # Only process files in 'labels' directories
                if 'labels' not in root:
                    continue
                    
                for file in files:
                    if file.endswith('.txt'):
                        file_path = os.path.join(root, file)
                        
                        try:
                            # Read the annotation file
                            with open(file_path, 'r') as f:
                                lines = f.readlines()
                            
                            updated_lines = []
                            file_updated = False
                            
                            for line in lines:
                                line = line.strip()
                                if not line:
                                    updated_lines.append(line)
                                    continue
                                    
                                parts = line.split()
                                if len(parts) >= 5:  # class_id x y width height
                                    try:
                                        old_class_id = int(parts[0])
                                        if old_class_id in old_to_new_mapping:
                                            new_class_id = old_to_new_mapping[old_class_id]
                                            parts[0] = str(new_class_id)
                                            file_updated = True
                                        updated_lines.append(' '.join(parts))
                                    except ValueError:
                                        # If class_id is not a valid integer, keep the line as is
                                        updated_lines.append(line)
                                else:
                                    updated_lines.append(line)
                            
                            # Write back to file if updated
                            if file_updated:
                                with open(file_path, 'w') as f:
                                    f.write('\n'.join(updated_lines))
                                    if updated_lines and updated_lines[-1]:  # Add final newline if content exists
                                        f.write('\n')
                                updated_files.append(file_path)
                                
                        except Exception as e:
                            logger.error(f"Error updating annotation file {file_path}: {str(e)}")
        
        logger.info(f"Updated {len(updated_files)} annotation files with new class IDs")
        if updated_files:
            logger.debug(f"Updated files: {updated_files}")
            
    except Exception as e:
        logger.error(f"Error updating annotation files class IDs: {str(e)}")



def _update_database_annotations_class_ids(extracted_label: str, old_to_new_mapping: Dict[int, int], db: Session) -> None:
    """
    Update class_data_id in the database for all pieces affected by the renumbering.
    """
    try:
        # Get all pieces that match the extracted label pattern and need updating
        pieces = db.query(Piece).filter(Piece.piece_label.like(f"{extracted_label}.%")).all()
        
        updated_pieces = []
        
        for piece in pieces:
            if piece.class_data_id in old_to_new_mapping:
                old_id = piece.class_data_id
                new_id = old_to_new_mapping[old_id]
                piece.class_data_id = new_id
                updated_pieces.append(f"{piece.piece_label}: {old_id} -> {new_id}")
        
        if updated_pieces:
            db.commit()
            logger.info(f"Updated class_data_id in database for {len(updated_pieces)} pieces")
            logger.debug(f"Updated pieces: {updated_pieces}")
        else:
            logger.info("No database pieces required class_data_id updates")
            
    except Exception as e:
        logger.error(f"Error updating database class IDs: {str(e)}")
        db.rollback()
