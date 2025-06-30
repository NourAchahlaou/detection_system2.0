
import re


from fastapi import HTTPException
from sqlalchemy.orm import Session

from training.app.services.advanced_rotation_service import rotate_and_save_images_and_annotations
from training.app.db.models.piece import Piece
from training.app.db.models.piece_image import PieceImage
from services.file_mover_with_hash_check import split_and_move_files

def rotate_and_update_images(piece_label: str, db: Session):
    # Retrieve the piece and its image record
    piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
    if not piece:
        raise HTTPException(status_code=404, detail="Piece not found.")
    
    piece_image = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).first()
    
    # Validate piece_label format and extract group label
    match = re.match(r'([A-Z]\d{3}\.\d{5})', piece_label)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid piece_label format.")
    group_label = match.group(1)
    
    # Define source folders (original + augmented images)
    source_image_folder = f'dataset_custom/images/valid/{piece_label}'
    source_annotation_folder = f'dataset_custom/labels/valid/{piece_label}'
    
    # Ensure that augmentation is applied first
    rotate_and_save_images_and_annotations(piece_label, rotation_angles=[45, 90, 135, 180, 270])
    
    # Define destination folders
    train_image_folder = f'dataset_custom/images/train/{piece_label}'
    train_annotation_folder = f'dataset_custom/labels/train/{piece_label}'
    valid_image_folder = f'dataset_custom/images/valid/{piece_label}'
    valid_annotation_folder = f'dataset_custom/labels/valid/{piece_label}'
    
    # Move all images (original + augmented) to training/validation sets

    
    # Update database path for the piece image
    new_image_url = f'Pieces/Pieces/images/train/{group_label}/{piece_label}'
    if piece_image:
        piece_image.image_url = new_image_url
        db.commit()
        db.refresh(piece_image)
        print(f"Database updated: {piece_image.image_url}")
    else:
        raise HTTPException(status_code=404, detail="PieceImage not found.")
    
    print("Files moved and database updated successfully.")

