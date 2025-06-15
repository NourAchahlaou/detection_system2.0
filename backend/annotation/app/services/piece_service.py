from datetime import datetime
import os
import re
from typing import Dict, Union, Any
from fastapi import HTTPException, logger
from sqlalchemy.orm import Session
import yaml
import shutil
import requests
from annotation.app.db.models.annotation import Annotation
from annotation.app.db.models.piece import Piece
from annotation.app.db.models.piece_image import PieceImage

# Virtual storage dictionary
virtual_storage: Dict[str, Dict] = {}

def get_images_of_piece(piece_label: str, db: Session):
    """Fetch all images of a piece with their annotation status."""
    db_piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
    
    if db_piece:
        # Get ALL images for this piece (both annotated and non-annotated)
        db_images = db.query(PieceImage).filter(
            PieceImage.piece_id == db_piece.id
        ).all()
        
        if not db_images:
            print("This piece has no images registered yet.")
            return []
        
        print(f"Retrieved {len(db_images)} images for piece {piece_label}")
        
        # FIXED: Use artifact_keeper service to serve images
        urlbase = "http://localhost/api/artifact_keeper/images/"
        
        try:
            result = []
            for image in db_images:
                # Convert absolute path to relative path for URL serving
                dataset_base = "/app/shared/dataset"
                if image.image_path.startswith(dataset_base):
                    relative_path = image.image_path[len(dataset_base):].lstrip("/")
                else:
                    relative_path = image.image_path
                
                clean_path = relative_path.replace("\\", "/")
                
                result.append({
                    "url": urlbase + clean_path,
                    "name": image.id,
                    "is_annotated": image.is_annotated,  # Include annotation status
                    "image_path": image.image_path  # Optional: include original path if needed
                })
            
            return result
        
        except Exception as e:
            print(f"Error retrieving images: {e}")
            return []
    
    print("Piece doesn't exist.")
    return []

def get_img_non_annotated(db: Session):
    db_pieces = db.query(Piece).filter(Piece.is_annotated == False).all()
    result = []

    if not db_pieces:
        print("No non-annotated pieces found.")
        return result

    # FIXED: Use artifact_keeper service to serve images
    urlbase = "http://localhost/api/artifact_keeper/images/"

    for piece in db_pieces:
        non_annotated_images_count = db.query(PieceImage).filter(
            PieceImage.piece_id == piece.id,
            PieceImage.is_annotated == False
        ).count()

        print(f"Non-annotated images count for piece {piece.id}: {non_annotated_images_count}")

        db_image = db.query(PieceImage).filter(
            PieceImage.piece_id == piece.id,
            PieceImage.is_annotated == False
        ).first()

        if db_image:
            dataset_base = "/app/shared/dataset"
            if db_image.image_path.startswith(dataset_base):
                relative_path = db_image.image_path[len(dataset_base):].lstrip("/")
            else:
                relative_path = db_image.image_path
            
            clean_path = relative_path.replace("\\", "/")
            
            image_info = {
                "piece_label": piece.piece_label,
                "url": urlbase + clean_path,
                "name": db_image.id,
                "nbr_img": piece.nbre_img
            }
            result.append(image_info)
        else:
            print(f"No non-annotated images registered for piece {piece.piece_label}.")

    return result

def save_annotation_in_memory(piece_label: str, image_id: int, annotation_data: dict):
    """Save the annotation in virtual storage instead of the database."""
    
    # Generate a unique key for the piece label
    if piece_label not in virtual_storage:
        virtual_storage[piece_label] = {'annotations': [], 'images': set()}
    
    # Validate the input data
    required_fields = ['type', 'x', 'y', 'width', 'height']
    missing_fields = [field for field in required_fields if field not in annotation_data]
    
    if missing_fields:
        raise ValueError(f"Missing fields in annotation data: {', '.join(missing_fields)}")
    
    # Add the annotation data to the virtual storage
    virtual_storage[piece_label]['annotations'].append({
        'type': str(annotation_data['type']),
        'x': float(annotation_data['x']),
        'y': float(annotation_data['y']),
        'width': float(annotation_data['width']),
        'height': float(annotation_data['height']),
        'image_id': int(image_id),
    })
    
    # Mark the image as annotated in virtual storage
    virtual_storage[piece_label]['images'].add(image_id)
   
    print(virtual_storage[piece_label]['annotations'])
    print(f"{annotation_data['x']} {annotation_data['y']} {annotation_data['width']} {annotation_data['height']}\n")

def update_piece_annotation_status(piece_label: str, is_annotated: bool):
    """
    Update piece annotation status via artifact_keeper API.
    This respects service boundaries and data ownership.
    """
    try:
        response = requests.patch(
            f"http://localhost/api/artifact_keeper/camera/pieces/{piece_label}/annotation-status",
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

def save_annotations_to_db(db: Session, piece_label: str, save_folder: str):
    """Save all annotations from virtual storage to the database."""

    # Ensure the piece exists in the database
    piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
    if not piece:
        raise HTTPException(status_code=404, detail="Piece not found")

    print(f"Piece found: {piece}")

    # Check if the piece has any annotations in virtual storage
    if piece_label not in virtual_storage or not virtual_storage[piece_label]['annotations']:
        raise HTTPException(status_code=404, detail="No annotations to save for this piece")

    print(f"Annotations found for piece: {virtual_storage[piece_label]['annotations']}")

    # Extract label and create proper annotation save path
    match = re.match(r'([A-Z]\d{3}\.\d{5})', piece_label)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid piece_label format.")
    extracted_label = match.group(1)

    # FIXED: Use the annotations volume instead of dataset volume
    # Use the annotations path from environment or default
    annotations_base_path = os.getenv('ANNOTATIONS_PATH', '/app/shared/annotations')
    save_folder = os.path.join(annotations_base_path, "labels", "valid", extracted_label, piece_label)
    
    # Create directory if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)
    print(f"Created/verified annotation directory: {save_folder}")

    # Collect unique class IDs and labels from annotations
    class_id_to_label = {}

    # Save each annotation from virtual storage to the database
    for annotation_data in virtual_storage[piece_label]['annotations']:
        print(f"Processing annotation: {annotation_data}")

        # Validate required keys are present
        required_keys = ['type', 'x', 'y', 'width', 'height', 'image_id']
        missing_keys = [key for key in required_keys if key not in annotation_data]

        if missing_keys:
            print(f"Missing keys in annotation data: {missing_keys}")
            raise HTTPException(status_code=400, detail=f"Missing keys in annotation data: {', '.join(missing_keys)}")

        # Additional validation for key values
        for key in required_keys:
            if annotation_data[key] is None or annotation_data[key] == '':
                print(f"Invalid value for {key}: {annotation_data[key]}")
                raise HTTPException(status_code=400, detail=f"Invalid value for {key}: {annotation_data[key]}")

        # Fetch the image record for the current annotation
        piece_image = db.query(PieceImage).filter(PieceImage.id == annotation_data['image_id']).first()

        if not piece_image:
            print("Image not found for image_id:", annotation_data['image_id'])
            raise HTTPException(status_code=404, detail="Image not found")

        print(f"Piece image found: {piece_image}")

        # Convert from percentage to YOLO format
        width_normalized = annotation_data['width'] / 100
        height_normalized = annotation_data['height'] / 100
        x_center_normalized = (annotation_data['x'] + annotation_data['width'] / 2) / 100
        y_center_normalized = (annotation_data['y'] + annotation_data['height'] / 2) / 100

        # Generate the annotationTXT_name from the file_name (not image_name)
        image_name_without_extension = os.path.splitext(piece_image.file_name)[0]
        annotationTXT_name = f"{image_name_without_extension}.txt"

        # Set the file path for saving the annotation
        file_path = os.path.join(save_folder, annotationTXT_name)

        # Prepare annotation in YOLO format
        annotationTXT = f"{piece.class_data_id} {x_center_normalized} {y_center_normalized} {width_normalized} {height_normalized}\n"

        # FIXED: Add error handling for file writing
        try:
            # Save the annotation inside a text file
            with open(file_path, 'w') as file:
                file.write(annotationTXT)
            print(f"Annotation saved to text file: {file_path}")
        except IOError as e:
            print(f"Error writing annotation file {file_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to write annotation file: {str(e)}")

        # Save the annotation in the database
        annotation = Annotation(
            annotationTXT_name=annotationTXT_name,
            type=annotation_data['type'],
            x=x_center_normalized,
            y=y_center_normalized,
            width=width_normalized,
            height=height_normalized,
            piece_image_id=annotation_data['image_id']
        )

        db.add(annotation)

        # Update the is_annotated field of the image (this is owned by annotation service)
        piece_image.is_annotated = True
        class_id_to_label[piece.class_data_id] = piece.piece_label

    # Check if all images related to the piece are annotated
    remaining_unannotated = db.query(PieceImage).filter(
        PieceImage.piece_id == piece.id,
        PieceImage.is_annotated == False
    ).count()
    
    print(f"Remaining unannotated images: {remaining_unannotated}")
    
    # If all images are annotated, update piece status via API
    if remaining_unannotated == 0:
        print("All images annotated. Updating piece to annotated via API.")
        
        # Update piece status via artifact_keeper API
        if not update_piece_annotation_status(piece_label, True):
            print("Warning: Failed to update piece annotation status via API")
        
        # FIXED: Update data.yaml in the annotations volume
        data_yaml_path = os.path.join(annotations_base_path, "data.yaml")

        # Load existing data if it exists
        if os.path.exists(data_yaml_path):
            with open(data_yaml_path, 'r') as yaml_file:
                data_yaml = yaml.safe_load(yaml_file)
                print(f"Existing data_yaml loaded: {data_yaml}")
        else:
            # FIXED: Use relative paths for train/val directories
            data_yaml = {
                'names': {},
                'nc': 0,
                'val': os.path.join(annotations_base_path, "images", "valid"),
                'train': os.path.join(annotations_base_path, "images", "train")
            }
            print("No existing data_yaml found. Creating new.")

        # Update the data_yaml with new class data
        class_id_to_label[piece.class_data_id] = piece.piece_label
        data_yaml['names'].update(class_id_to_label)
        data_yaml['nc'] = len(data_yaml['names'])

        # Create directories if needed
        os.makedirs(os.path.dirname(data_yaml_path), exist_ok=True)

        # Write to data.yaml
        try:
            with open(data_yaml_path, 'w') as yaml_file:
                yaml.dump(data_yaml, yaml_file, default_flow_style=False)
            print(f"data.yaml file updated at: {data_yaml_path}")
        except IOError as e:
            print(f"Warning: Failed to write data.yaml: {e}")

    # Commit changes to annotation service's data only
    db.commit()
    db.refresh(annotation)
    db.refresh(piece_image)
          
    # Clear virtual storage
    virtual_storage.pop(piece_label, None)
    print("Annotations saved successfully and virtual storage cleared.")
    
    return {"status": "Annotations saved successfully", "save_folder": save_folder}

# http://localhost/api/artifact_keeper/camera/cameras/stop
def delete_annotation_service(annotation_id: int, db: Session) -> Dict[str, Any]:
    """Delete a specific annotation from the database"""
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
                PieceImage.is_annotated == True
            ).count()
            
            print(f"Other annotated images in piece {piece.id}: {other_annotated_images}")
            
            if other_annotated_images == 0:
                # Update piece status via artifact_keeper API
                print(f"Updating piece {piece_label} status to not annotated")
                update_success = update_piece_annotation_status(piece_label, False)
                if not update_success:
                    print(f"Warning: Failed to update piece {piece_label} status via API")
        
        # Delete the corresponding annotation text file if it exists
        try:
            # FIXED: Use the correct path structure
            match = re.match(r'([A-Z]\d{3}\.\d{5})', piece_label)
            if match:
                extracted_label = match.group(1)
                
                # Use the annotations volume path
                annotations_base_path = os.getenv('ANNOTATIONS_PATH', '/app/shared/annotations')
                save_folder = os.path.join(annotations_base_path, "labels", "valid", extracted_label, piece_label)
                
                # Get the annotation file name
                annotation_file_path = os.path.join(save_folder, annotation_txt_name)
                
                if os.path.exists(annotation_file_path):
                    os.remove(annotation_file_path)
                    print(f"Deleted annotation file: {annotation_file_path}")
                else:
                    print(f"Annotation file not found: {annotation_file_path}")
            else:
                print(f"Could not extract label from piece_label: {piece_label}")
                
        except Exception as file_error:
            print(f"Warning: Could not delete annotation file: {str(file_error)}")
        
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
    
def delete_virtual_annotation_service(piece_label: str, image_id: int, annotation_id: str, virtual_storage: Dict) -> Dict[str, Any]:
    """Delete a specific annotation from virtual storage"""
    try:
        # Check if piece exists in virtual storage
        if piece_label not in virtual_storage:
            raise HTTPException(status_code=404, detail="Piece not found in virtual storage")
        
        # Find and remove the annotation from virtual storage
        annotations = virtual_storage[piece_label]['annotations']
        
        # Convert annotation_id to appropriate type for comparison
        try:
            annotation_id_float = float(annotation_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid annotation ID format")
        
        # Find the annotation by image_id and a matching identifier
        # Remove the most recently added annotation for this image as a fallback
        # if we can't find an exact match
        filtered_annotations = []
        removed_annotation = None
        
        for annotation in annotations:
            if annotation['image_id'] == image_id:
                # If this is the most recent annotation for this image, remove it
                if removed_annotation is None:
                    removed_annotation = annotation
                    continue
            filtered_annotations.append(annotation)
        
        if removed_annotation is None:
            raise HTTPException(status_code=404, detail="Annotation not found in virtual storage")
        
        # Update virtual storage
        virtual_storage[piece_label]['annotations'] = filtered_annotations
        
        # If image has no more annotations in virtual storage, remove it from the images set
        remaining_image_annotations = [ann for ann in filtered_annotations if ann['image_id'] == image_id]
        if not remaining_image_annotations:
            virtual_storage[piece_label]['images'].discard(image_id)
        
        # If no annotations remain for the piece, clean up virtual storage
        if not virtual_storage[piece_label]['annotations']:
            virtual_storage.pop(piece_label, None)
        
        print(f"Removed annotation from virtual storage for piece {piece_label}, image {image_id}")
        
        return {
            "status": "success",
            "message": "Annotation removed from virtual storage",
            "removed_annotation": removed_annotation,
            "remaining_count": len(filtered_annotations)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error removing annotation from virtual storage: {e}")
        raise HTTPException(status_code=500, detail=f"Error removing annotation from virtual storage: {str(e)}")

def get_virtual_annotations_service(piece_label: str, virtual_storage: Dict) -> Dict[str, Any]:
    """Get all annotations currently in virtual storage for a piece"""
    try:
        if piece_label not in virtual_storage:
            return {
                "piece_label": piece_label,
                "annotations": [],
                "images": [],
                "count": 0
            }
        
        return {
            "piece_label": piece_label,
            "annotations": virtual_storage[piece_label]['annotations'],
            "images": list(virtual_storage[piece_label]['images']),
            "count": len(virtual_storage[piece_label]['annotations'])
        }
        
    except Exception as e:
        print(f"Error getting virtual annotations: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting virtual annotations: {str(e)}")

def update_piece_annotation_status(piece_label: str, is_annotated: bool) -> bool:
    """
    Update piece annotation status via artifact_keeper API.
    This respects service boundaries and data ownership.
    """
    try:
        response = requests.patch(
            f"http://localhost/api/artifact_keeper/camera/pieces/{piece_label}/annotation-status",
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