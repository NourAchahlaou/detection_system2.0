from datetime import datetime
import os
import re
from typing import Dict, Union
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
    """Fetch all images of a piece that are not annotated yet."""
    db_piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
    
    if db_piece:
        db_images = db.query(PieceImage).filter(
            PieceImage.piece_id == db_piece.id,
            PieceImage.is_annotated == False
        ).all()

        if not db_images:
            print("This piece has no unannotated images registered yet.")
            return []

        print("Piece images retrieved.")
        
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
                    "name": image.id
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

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Prepare annotation in YOLO format
        annotationTXT = f"{piece.class_data_id} {x_center_normalized} {y_center_normalized} {width_normalized} {height_normalized}\n"

        # Save the annotation inside a text file
        with open(file_path, 'w') as file:
            file.write(annotationTXT)

        print(f"Annotation saved to text file: {file_path}")

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

    # FIXED: Check if all images related to the piece are annotated
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
        
        # Update data.yaml
        data_yaml_path = os.path.join("dataset_custom", "data.yaml")

        # Load existing data if it exists
        if os.path.exists(data_yaml_path):
            with open(data_yaml_path, 'r') as yaml_file:
                data_yaml = yaml.safe_load(yaml_file)
                print(f"Existing data_yaml loaded: {data_yaml}")
        else:
            data_yaml = {
                'names': {},
                'nc': 0,
                'val': os.path.join("C:\\Users\\hp\\Desktop\\Airbus\\detectionSystemAirbus", "backend", "dataset_custom","images","valid"),
                'train': os.path.join("C:\\Users\\hp\\Desktop\\Airbus\\detectionSystemAirbus", "backend", "dataset_custom", "images", "train")
            }
            print("No existing data_yaml found. Creating new.")

        # Update the data_yaml with new class data
        class_id_to_label[piece.class_data_id] = piece.piece_label
        data_yaml['names'].update(class_id_to_label)
        data_yaml['nc'] = len(data_yaml['names'])

        # Create directories if needed
        os.makedirs(os.path.dirname(data_yaml_path), exist_ok=True)

        # Write to data.yaml
        with open(data_yaml_path, 'w') as yaml_file:
            yaml.dump(data_yaml, yaml_file, default_flow_style=False)

        print(f"data.yaml file updated at: {data_yaml_path}")

    # Commit changes to annotation service's data only
    db.commit()
    db.refresh(annotation)
    db.refresh(piece_image)
          
    # Clear virtual storage
    virtual_storage.pop(piece_label, None)
    print("Annotations saved successfully and virtual storage cleared.")
    
    # # Call the stop camera endpoint
    # try:
    #     # FIXED: Use the correct URL for your artifact keeper service
    #     stop_camera_response = requests.post("ttp://localhost/api/artifact_keeper/cameras/stop", timeout=10)

    #     if stop_camera_response.status_code != 200:
    #         print(f"Failed to stop camera: {stop_camera_response.json().get('detail')}")
    #     else:
    #         print("Camera stopped successfully.")
    # except Exception as e:
    #     print(f"Error stopping camera: {str(e)}")

    return {"status": "Annotations saved successfully"}

# http://localhost/api/artifact_keeper/camera/cameras/stop
