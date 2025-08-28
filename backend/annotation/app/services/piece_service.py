import os
from datetime import datetime
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

# Configuration for API endpoints
def get_api_base_url():
    """Get the base URL for artifact_keeper API calls"""
    # In containerized environment, use the service name from docker-compose
    # In development, fall back to localhost
    api_host = os.getenv('ARTIFACT_KEEPER_HOST', 'artifact_keeper')
    api_port = os.getenv('ARTIFACT_KEEPER_PORT', '8000')
    
    # If running in containers, use service name; otherwise use localhost
    if api_host == 'artifact_keeper':
        return f"http://{api_host}:{api_port}"
    else:
        return f"http://localhost/api/artifact_keeper"

def _extract_piece_group(piece_label: str) -> str:
    """Extract the group prefix from the piece label (e.g., G123 from G123.12345.123.12)."""
    match = re.match(r'^([A-Z]\d{3})', piece_label)
    if not match:
        raise HTTPException(status_code=400, detail=f"Invalid piece_label format: {piece_label}")
    return match.group(1)

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
        
        # UPDATED: Use unified dataset structure for serving images
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
                    "is_annotated": image.is_annotated,
                    "image_path": image.image_path
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

    # UPDATED: Use unified dataset structure for serving images
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
        payload = {"is_annotated": is_annotated}
        api_base_url = get_api_base_url()
        
        response = requests.patch(
            f"{api_base_url}/captureImage/pieces/{piece_label}/annotation-status",
            json=payload,
            timeout=10,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Piece {piece_label} API Response Status: {response.status_code}")
        print(f"Piece {piece_label} API Response Text: {response.text}")
        
        if response.status_code == 200:
            print(f"Successfully updated piece {piece_label} annotation status to {is_annotated}")
            return True
        else:
            print(f"Failed to update piece annotation status: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Error calling artifact_keeper API: {str(e)}")
        return False

def create_yolo_directory_structure(base_path: str, piece_label: str):
    """Create YOLO directory structure for a piece in the unified dataset folder with group structure"""
    # Extract piece group (e.g., G123 from G123.12345.123.12)
    piece_group = _extract_piece_group(piece_label)
    
    # UPDATED: New structure - shared_data/dataset/piece/piece/{piece_group}/{piece_label}/
    piece_path = os.path.join(base_path, "piece", "piece", piece_group, piece_label)
    
    # Create the YOLO directories
    directories = [
        os.path.join(piece_path, "images", "valid"),
        os.path.join(piece_path, "images", "train"),
        os.path.join(piece_path, "labels", "valid"),
        os.path.join(piece_path, "labels", "train")
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")
    
    return piece_path

def copy_image_to_yolo_structure(source_image_path: str, piece_label: str, image_filename: str, dataset_base_path: str):
    """Copy image from captured location to YOLO structure within the same dataset folder"""
    try:
        # Extract piece group
        piece_group = _extract_piece_group(piece_label)
        
        # UPDATED: Destination is now in the group-based structure
        dest_image_path = os.path.join(dataset_base_path, "piece", "piece", piece_group, piece_label, "images", "valid", image_filename)

        # Check if source and destination are the same
        if os.path.abspath(source_image_path) == os.path.abspath(dest_image_path):
            print(f"Source and destination are the same: {source_image_path}")
            return True  # Consider this a success since the file is already in the right place
        
        # Copy the image file
        if os.path.exists(source_image_path):
            shutil.copy2(source_image_path, dest_image_path)
            print(f"Copied image from {source_image_path} to {dest_image_path}")
            return True
        else:
            print(f"Source image not found: {source_image_path}")
            return False
    except Exception as e:
        print(f"Error copying image: {e}")
        return False
    
def update_piece_image_annotation_status(image_id: int, is_annotated: bool):
    """
    Update piece image annotation status via artifact_keeper API.
    This respects service boundaries and data ownership.
    """
    try:
        payload = {"is_annotated": is_annotated}
        api_base_url = get_api_base_url()
        
        response = requests.patch(
            f"{api_base_url}/captureImage/piece-images/{image_id}/annotation-status",
            json=payload,
            timeout=10,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"Image {image_id} API Response Status: {response.status_code}")
        print(f"Image {image_id} API Response Text: {response.text}")
        
        if response.status_code == 200:
            print(f"Successfully updated piece image {image_id} annotation status to {is_annotated}")
            return True
        else:
            print(f"Failed to update piece image annotation status: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Error calling artifact_keeper API: {str(e)}")
        return False

def batch_update_piece_images_annotation_status(piece_label: str, image_ids: list, is_annotated: bool):
    """
    Batch update multiple piece images annotation status via artifact_keeper API.
    This is more efficient when updating many images at once.
    """
    try:
        # Fix: The API expects the request body to match PieceAnnotationStatusUpdate + image_ids
        payload = {
            "image_ids": image_ids,  # This is separate from the status update
            "is_annotated": is_annotated
        }
        
        api_base_url = get_api_base_url()
        
        response = requests.patch(
            f"{api_base_url}/captureImage/pieces/{piece_label}/batch-update-images",
            json=payload,
            timeout=30,
            headers={"Content-Type": "application/json"}
        )
        
        print(f"API Response Status: {response.status_code}")
        print(f"API Response Text: {response.text}")
        
        if response.status_code == 200:
            result = response.json()
            print(f"Successfully batch updated {result.get('updated_image_count', 0)} images for piece {piece_label}")
            return True
        else:
            print(f"Failed to batch update piece images: {response.status_code} - {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"Error calling artifact_keeper batch update API: {str(e)}")
        return False

def get_piece_annotation_status(piece_label: str):
    """
    Get piece annotation status from artifact_keeper API.
    """
    try:
        api_base_url = get_api_base_url()
        
        response = requests.get(
            f"{api_base_url}/captureImage/pieces/{piece_label}/annotation-status",
            timeout=10
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Failed to get piece annotation status: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.RequestException as e:
        print(f"Error calling artifact_keeper API: {str(e)}")
        return None

def update_group_data_yaml(dataset_base_path: str, piece_group: str, piece_label: str, class_id: int):
    """Update group-specific data.yaml file with new class data."""
    # Create group-specific data.yaml path
    group_data_yaml_path = os.path.join(dataset_base_path, "piece", "piece", piece_group, "data.yaml")
    
    # Load existing group data if it exists
    if os.path.exists(group_data_yaml_path):
        with open(group_data_yaml_path, 'r') as yaml_file:
            group_data_yaml = yaml.safe_load(yaml_file)
            print(f"Existing group data_yaml loaded for {piece_group}: {group_data_yaml}")
    else:
        # Create new group-specific data.yaml with relative paths for YOLO training
        group_data_yaml = {
            'names': {},
            'nc': 0,
            'path': os.path.join(dataset_base_path, "piece", "piece", piece_group),
            'train': '*/images/train',
            'val': '*/images/valid'
        }
        print(f"No existing group data_yaml found for {piece_group}. Creating new.")

    # Update the group data_yaml with new class data
    # For group-specific numbering: assign sequential IDs within the group
    if piece_label not in group_data_yaml['names'].values():
        # Find the next available ID within this group
        existing_ids = list(group_data_yaml['names'].keys()) if group_data_yaml['names'] else []
        next_group_id = 0
        while next_group_id in existing_ids:
            next_group_id += 1
        
        group_data_yaml['names'][next_group_id] = piece_label
        group_data_yaml['nc'] = len(group_data_yaml['names'])
        
        print(f"Added {piece_label} with ID {next_group_id} to group {piece_group}")

    # Create directories if needed
    os.makedirs(os.path.dirname(group_data_yaml_path), exist_ok=True)

    # Write to group-specific data.yaml
    try:
        with open(group_data_yaml_path, 'w') as yaml_file:
            yaml.dump(group_data_yaml, yaml_file, default_flow_style=False)
        print(f"Group data.yaml file updated at: {group_data_yaml_path}")
        print(f"Current group {piece_group} classes: {group_data_yaml['names']}")
    except IOError as e:
        print(f"Warning: Failed to write group data.yaml: {e}")

def save_annotations_to_db(db: Session, piece_label: str, save_folder: str):
    """Save all annotations from virtual storage to the database and create YOLO structure."""

    # Ensure the piece exists in the database (READ-ONLY check)
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

    # Extract piece group
    piece_group = _extract_piece_group(piece_label)

    # Use the unified dataset path
    dataset_base_path = os.getenv('DATASET_BASE_PATH', '/app/shared/dataset')
    
    # Create YOLO directory structure in the unified dataset folder with group structure
    piece_path = create_yolo_directory_structure(dataset_base_path, piece_label)
    
    # Set save folder for labels within the group-based structure
    labels_save_folder = os.path.join(piece_path, "labels", "valid")
    
    print(f"Created YOLO structure at: {piece_path}")
    print(f"Labels will be saved to: {labels_save_folder}")

    # FIRST: Update or create the group-specific data.yaml to get the correct class ID
    group_data_yaml_path = os.path.join(dataset_base_path, "piece", "piece", piece_group, "data.yaml")
    
    # Load existing group data if it exists
    if os.path.exists(group_data_yaml_path):
        with open(group_data_yaml_path, 'r') as yaml_file:
            group_data_yaml = yaml.safe_load(yaml_file)
            print(f"Existing group data_yaml loaded for {piece_group}: {group_data_yaml}")
    else:
        # Create new group-specific data.yaml with relative paths for YOLO training
        group_data_yaml = {
            'names': {},
            'nc': 0,
            'path': os.path.join(dataset_base_path, "piece", "piece", piece_group),
            'train': '*/images/train',
            'val': '*/images/valid'
        }
        print(f"No existing group data_yaml found for {piece_group}. Creating new.")

    # Find or assign the group-specific class ID for this piece
    group_class_id = None
    for gid, label in group_data_yaml.get('names', {}).items():
        if label == piece_label:
            group_class_id = gid
            break
    
    # If not found, assign a new ID
    if group_class_id is None:
        existing_ids = list(group_data_yaml.get('names', {}).keys())
        next_group_id = 0
        while next_group_id in existing_ids:
            next_group_id += 1
        
        group_class_id = next_group_id
        group_data_yaml['names'][group_class_id] = piece_label
        group_data_yaml['nc'] = len(group_data_yaml['names'])
        
        print(f"Assigned new class ID {group_class_id} to {piece_label} in group {piece_group}")
        
        # Create directories if needed and save the updated YAML
        os.makedirs(os.path.dirname(group_data_yaml_path), exist_ok=True)
        
        try:
            with open(group_data_yaml_path, 'w') as yaml_file:
                yaml.dump(group_data_yaml, yaml_file, default_flow_style=False)
            print(f"Updated group data.yaml file at: {group_data_yaml_path}")
            print(f"Current group {piece_group} classes: {group_data_yaml['names']}")
        except IOError as e:
            print(f"Warning: Failed to write group data.yaml: {e}")
    else:
        print(f"Found existing class ID {group_class_id} for {piece_label} in group {piece_group}")

    # Collect unique class IDs and labels from annotations
    processed_images = set()
    annotated_image_ids = []  # Track which images we've annotated

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

        # Fetch the image record for the current annotation (READ-ONLY)
        piece_image = db.query(PieceImage).filter(PieceImage.id == annotation_data['image_id']).first()

        if not piece_image:
            print("Image not found for image_id:", annotation_data['image_id'])
            raise HTTPException(status_code=404, detail="Image not found")

        print(f"Piece image found: {piece_image}")

        # Copy image to YOLO structure if not already done
        if annotation_data['image_id'] not in processed_images:
            copy_success = copy_image_to_yolo_structure(
                piece_image.image_path, 
                piece_label, 
                piece_image.file_name, 
                dataset_base_path
            )
            if copy_success:
                processed_images.add(annotation_data['image_id'])

        # Convert from percentage to YOLO format
        width_normalized = annotation_data['width'] / 100
        height_normalized = annotation_data['height'] / 100
        x_center_normalized = (annotation_data['x'] + annotation_data['width'] / 2) / 100
        y_center_normalized = (annotation_data['y'] + annotation_data['height'] / 2) / 100

        # Generate the annotationTXT_name from the file_name
        image_name_without_extension = os.path.splitext(piece_image.file_name)[0]
        annotationTXT_name = f"{image_name_without_extension}.txt"

        # Set the file path for saving the annotation in YOLO structure
        file_path = os.path.join(labels_save_folder, annotationTXT_name)

        # NOW use the correct group_class_id that was determined above
        print(f"Using class ID {group_class_id} for piece {piece_label}")

        # Prepare annotation in YOLO format with group-specific class ID
        annotationTXT = f"{group_class_id} {x_center_normalized} {y_center_normalized} {width_normalized} {height_normalized}\n"
        print(f"YOLO annotation line: {annotationTXT.strip()}")

        # Save the annotation inside a text file
        try:
            # If file exists, append to it (for multiple annotations per image)
            mode = 'a' if os.path.exists(file_path) else 'w'
            with open(file_path, mode) as file:
                file.write(annotationTXT)
            print(f"Annotation saved to text file: {file_path}")
        except IOError as e:
            print(f"Error writing annotation file {file_path}: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to write annotation file: {str(e)}")

        # Save the annotation in the database (annotation service owns this)
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
        
        # Track this image as annotated
        if annotation_data['image_id'] not in annotated_image_ids:
            annotated_image_ids.append(annotation_data['image_id'])

    # Commit annotation service's own data first
    try:
        db.commit()
        print("Successfully committed annotation data to database")
    except Exception as e:
        db.rollback()
        print(f"Failed to commit annotation data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to save annotations: {str(e)}")
    
    # Now update piece images annotation status via API
    print(f"Attempting to update {len(annotated_image_ids)} images: {annotated_image_ids}")
    
    # FIXED: Use individual updates for now as they're more reliable
    update_successes = 0
    for image_id in annotated_image_ids:
        if update_piece_image_annotation_status(image_id, True):
            update_successes += 1
        else:
            print(f"Warning: Failed to update piece image {image_id} annotation status via API")
    
    print(f"Successfully updated {update_successes} out of {len(annotated_image_ids)} images")
    
    # Only proceed with piece status update if at least some image updates succeeded
    if update_successes > 0:
        # Check if all images related to the piece are annotated
        piece_status = get_piece_annotation_status(piece_label)
        
        if piece_status:
            remaining_unannotated = piece_status.get('remaining_unannotated_images', 0)
            print(f"Remaining unannotated images: {remaining_unannotated}")
            
            # If all images are annotated, update piece status via API
            if remaining_unannotated == 0:
                print("All images annotated. Updating piece to annotated via API.")
                
                # Update piece status via artifact_keeper API
                if update_piece_annotation_status(piece_label, True):
                    print("Successfully updated piece annotation status")
                else:
                    print("Warning: Failed to update piece annotation status via API")
            else:
                print(f"Piece not fully annotated yet. {remaining_unannotated} images remaining.")
        else:
            print("Warning: Could not get piece annotation status from API")
    else:
        print("Warning: No image updates succeeded, skipping piece status update")
          
    # Clear virtual storage
    virtual_storage.pop(piece_label, None)
    print("Annotations saved successfully and virtual storage cleared.")
    
    return {
        "status": "Annotations saved successfully", 
        "save_folder": labels_save_folder, 
        "yolo_structure": piece_path,
        "piece_group": piece_group,
        "updated_images": update_successes,
        "total_images": len(annotated_image_ids),
        "class_id_used": group_class_id
    }

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
        piece_group = _extract_piece_group(piece_label)
        
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
            # UPDATED: Use the group-based dataset path for YOLO structure
            dataset_base_path = os.getenv('DATASET_BASE_PATH', '/app/shared/dataset')
            labels_folder = os.path.join(dataset_base_path, "piece", "piece", piece_group, piece_label, "labels", "valid")
            
            # Get the annotation file name
            annotation_file_path = os.path.join(labels_folder, annotation_txt_name)
            
            if os.path.exists(annotation_file_path):
                os.remove(annotation_file_path)
                print(f"Deleted annotation file: {annotation_file_path}")
            else:
                print(f"Annotation file not found: {annotation_file_path}")
                
        except Exception as file_error:
            print(f"Warning: Could not delete annotation file: {str(file_error)}")
        
        # Commit all changes
        db.commit()
        
        return {
            "status": "success",
            "message": f"Annotation {annotation_id} deleted successfully",
            "image_still_annotated": remaining_annotations > 0,
            "piece_label": piece_label,
            "piece_group": piece_group,
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

def get_group_pieces_info(db: Session, piece_group: str) -> Dict[str, Any]:
    """Get information about all pieces in a specific group"""
    try:
        # Query all pieces that belong to this group
        pieces = db.query(Piece).filter(
            Piece.piece_label.like(f"{piece_group}.%")
        ).all()
        
        if not pieces:
            return {
                "group": piece_group,
                "pieces": [],
                "total_pieces": 0,
                "annotated_pieces": 0
            }
        
        pieces_info = []
        annotated_count = 0
        
        for piece in pieces:
            piece_info = {
                "piece_label": piece.piece_label,
                "class_data_id": piece.class_data_id,
                "is_annotated": piece.is_annotated,
                "image_count": piece.nbre_img
            }
            pieces_info.append(piece_info)
            
            if piece.is_annotated:
                annotated_count += 1
        
        return {
            "group": piece_group,
            "pieces": pieces_info,
            "total_pieces": len(pieces),
            "annotated_pieces": annotated_count
        }
        
    except Exception as e:
        print(f"Error getting group pieces info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting group pieces info: {str(e)}")

def get_all_groups_info(db: Session) -> Dict[str, Any]:
    """Get information about all groups in the database"""
    try:
        # Get all pieces and extract unique groups
        all_pieces = db.query(Piece).all()
        
        groups_info = {}
        
        for piece in all_pieces:
            try:
                group = _extract_piece_group(piece.piece_label)
                
                if group not in groups_info:
                    groups_info[group] = {
                        "group": group,
                        "pieces": [],
                        "total_pieces": 0,
                        "annotated_pieces": 0
                    }
                
                piece_info = {
                    "piece_label": piece.piece_label,
                    "class_data_id": piece.class_data_id,
                    "is_annotated": piece.is_annotated,
                    "image_count": piece.nbre_img
                }
                
                groups_info[group]["pieces"].append(piece_info)
                groups_info[group]["total_pieces"] += 1
                
                if piece.is_annotated:
                    groups_info[group]["annotated_pieces"] += 1
                    
            except Exception as group_error:
                print(f"Error processing piece {piece.piece_label}: {group_error}")
                continue
        
        return {
            "groups": list(groups_info.values()),
            "total_groups": len(groups_info)
        }
        
    except Exception as e:
        print(f"Error getting all groups info: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting all groups info: {str(e)}")

def validate_group_data_yaml(dataset_base_path: str, piece_group: str) -> Dict[str, Any]:
    """Validate and repair group data.yaml file if needed"""
    try:
        group_data_yaml_path = os.path.join(dataset_base_path, "piece", "piece", piece_group, "data.yaml")
        
        validation_result = {
            "group": piece_group,
            "yaml_path": group_data_yaml_path,
            "exists": os.path.exists(group_data_yaml_path),
            "valid": False,
            "issues": [],
            "repaired": False
        }
        
        if not validation_result["exists"]:
            validation_result["issues"].append("data.yaml file does not exist")
            return validation_result
        
        # Load and validate the YAML
        try:
            with open(group_data_yaml_path, 'r') as yaml_file:
                yaml_data = yaml.safe_load(yaml_file)
                
            # Check required fields
            required_fields = ['names', 'nc', 'path', 'train', 'val']
            missing_fields = [field for field in required_fields if field not in yaml_data]
            
            if missing_fields:
                validation_result["issues"].extend([f"Missing field: {field}" for field in missing_fields])
            
            # Check if nc matches the number of names
            if 'names' in yaml_data and 'nc' in yaml_data:
                actual_count = len(yaml_data['names'])
                declared_count = yaml_data['nc']
                
                if actual_count != declared_count:
                    validation_result["issues"].append(f"Class count mismatch: declared={declared_count}, actual={actual_count}")
            
            if not validation_result["issues"]:
                validation_result["valid"] = True
            
            validation_result["content"] = yaml_data
            
        except Exception as yaml_error:
            validation_result["issues"].append(f"YAML parsing error: {str(yaml_error)}")
        
        return validation_result
        
    except Exception as e:
        print(f"Error validating group data.yaml: {e}")
        raise HTTPException(status_code=500, detail=f"Error validating group data.yaml: {str(e)}")