#database service
from fastapi import HTTPException, logger
import os
import re
import shutil
from sqlalchemy.orm import Session
from artifact_keeper.app.db.models.piece import Piece
from artifact_keeper.app.db.models.piece_image import PieceImage
from artifact_keeper.app.db.models.annotation import Annotation


def get_all_datasets(db: Session):
    """
    Fetch all datasets including pieces, images, and annotations from the database.
    """
    datasets = {}

    # Fetch all pieces
    pieces = db.query(Piece).all()

    for piece in pieces:
        piece_data = {
            "id": piece.id,
            "class_data_id" : piece.class_data_id,
            "label": piece.piece_label,
            "is_annotated": piece.is_annotated,
            "is_yolo_trained" : piece.is_yolo_trained,
            "nbre_img": piece.nbre_img,
            "images": []
        }

        # Fetch all images associated with the piece
        images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).all()

        for image in images:
            image_data = {
                "id": image.id,
                "file_name": image.file_name,
                "image_path": image.image_path,
                "is_annotated": image.is_annotated,
                "annotations": []
            }

            # Fetch all annotations associated with the image
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

# training microservice
def get_piece_labels_by_group(group_label: str, db: Session):
    try:
        # Query the database for pieces with piece_label starting with group_label
        pieces = db.query(Piece).filter(Piece.piece_label.like(f'{group_label}%')).all()
        
        # Extract piece labels
        piece_labels = [piece.piece_label for piece in pieces]
        
        if not piece_labels:
            logger.info(f"No pieces found for group '{group_label}'.")
        
        return piece_labels

    except Exception as e:
        logger.error(f"An error occurred while fetching piece labels: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred while fetching piece labels: {e}")
    

# ****************

# this will handle the deletion 
def delete_directory(directory_path: str):
    """Recursively delete a directory and its contents."""
    if os.path.exists(directory_path):
        shutil.rmtree(directory_path)
        print(f"Deleted directory: {directory_path}")
    else:
        print(f"Directory not found: {directory_path}")


def delete_piece_by_label(piece_label: str, db: Session):
    """Delete a piece, its images, annotations, and related folders."""
    # Fetch the piece from the database
    piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
    if not piece:
        raise HTTPException(status_code=404, detail="Piece not found")

    # Fetch all images related to the piece
    images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).all()
    
    # Delete all annotations related to each image
    for image in images:
        annotations = db.query(Annotation).filter(Annotation.piece_image_id == image.id).all()
        for annotation in annotations:
            db.delete(annotation)
    
    # Delete all images
    for image in images:
        db.delete(image)
    
    # Delete the piece
    db.delete(piece)
    db.commit()

    # Extract label for folder paths
    match = re.match(r'([A-Z]\d{3}\.\d{5})', piece_label)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid piece_label format.")
    
    extracted_label = match.group(1)
    
   
    folder_path_annotations_valid = os.path.join("dataset",'Pieces','Pieces','labels','valid',extracted_label,piece_label)
    folder_path_images_valid = os.path.join("dataset",'Pieces','Pieces','images','valid',extracted_label,piece_label)
    folder_path_annotations_train = os.path.join("dataset",'Pieces','Pieces','labels','train',extracted_label,piece_label)
    folder_path_images_train = os.path.join("dataset",'Pieces','Pieces','images','train',extracted_label,piece_label)
    
    # Delete the folders
    delete_directory(folder_path_annotations_valid)
    delete_directory(folder_path_images_valid)
    delete_directory(folder_path_annotations_train)
    delete_directory(folder_path_images_train)
    
    return {"status": "Piece and associated data deleted successfully"}

def delete_all_pieces(db: Session):
    """Delete all pieces, their images, annotations, and associated folders."""
    
    # Fetch all pieces
    pieces = db.query(Piece).all()
    
    if not pieces:
        raise HTTPException(status_code=404, detail="No pieces found to delete")

    for piece in pieces:
        # Fetch all images associated with the piece
        images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).all()

        # Delete annotations and images
        for image in images:
            # Fetch and delete annotations
            annotations = db.query(Annotation).filter(Annotation.piece_image_id == image.id).all()
            for annotation in annotations:
                db.delete(annotation)
            
            # Delete image record
            db.delete(image)
        
        # Delete the piece record
        db.delete(piece)
        
        # Determine the path of the folder to delete
        match = re.match(r'([A-Z]\d{3}\.\d{5})', piece.piece_label)
        if match:
            extracted_label = match.group(1)
            folder_path = os.path.join("dataset","Pieces")
            
            # Remove the folder and all its contents
            if os.path.exists(folder_path):
                shutil.rmtree(folder_path)
                print(f"Folder {folder_path} deleted successfully.")
            else:
                print(f"Folder {folder_path} does not exist.")

    # Commit the changes
    db.commit()
    
    return {"status": "All pieces and associated data deleted successfully"}