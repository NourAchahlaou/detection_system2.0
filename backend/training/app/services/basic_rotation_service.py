
import re


from fastapi import HTTPException
from sqlalchemy.orm import Session

from training.app.services.advanced_rotation_service import rotate_and_save_images_and_annotations
from training.app.db.models.piece import Piece
from training.app.db.models.piece_image import PieceImage
from services.file_mover_with_hash_check import move_files_if_not_moved


def rotate_and_update_images(piece_label: str, db: Session):
    piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()

    if not piece:
        raise HTTPException(status_code=404, detail="Piece not found.")
    
    # Assuming PieceImage is a SQLAlchemy model and you have a piece_id or similar identifier
    piece_image = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).first()
    
    # Validate piece_label format
    match = re.match(r'([A-Z]\d{3}\.\d{5})', piece_label)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid piece_label format.")
        
    group_label = match.group(1)
    
    image_folder = f'dataset_custom/images/valid/{piece_label}'
    annotation_folder = f'dataset_custom/labels/valid/{piece_label}'
    save_image_folder = f'dataset_custom/images/train/{piece_label}'
    save_annotation_folder = f'dataset_custom/labels/train/{piece_label}'


    # Rotate and save images and annotations (assuming you have this function defined)
    rotate_and_save_images_and_annotations(piece_label, rotation_angles=[45,90,135,180,270])
        # Call the move_files_if_not_moved function to handle the file movement with hash checking
    move_files_if_not_moved(image_folder, annotation_folder, save_image_folder, save_annotation_folder,2,db)

    image_folder_1 = f'dataset/Pieces/Pieces/images/valid/{group_label}/{piece_label}'
    annotation_folder_1 = f'dataset/Pieces/Pieces/labels/valid/{group_label}/{piece_label}'
    save_image_folder_1 = f'dataset/Pieces/Pieces/images/train/{group_label}/{piece_label}'
    save_annotation_folder_1 = f'dataset/Pieces/Pieces/labels/train/{group_label}/{piece_label}'

    # Rotate and save images and annotations (assuming you have this function defined)
    rotate_and_save_images_and_annotations(piece_label, rotation_angles=[45,90,135,180,270])
        # Call the move_files_if_not_moved function to handle the file movement with hash checking
    move_files_if_not_moved(image_folder_1, annotation_folder_1, save_image_folder_1, save_annotation_folder_1,2,db)

    # Update piece_image URL in the database after successful movement
    new_image_url = f'Pieces/Pieces/images/train/{group_label}/{piece_label}'
    
    if piece_image:
        piece_image.image_url = new_image_url
        db.commit()
        db.refresh(piece_image)
        print(f"Database updated: {piece_image.image_url}")
    else:
        raise HTTPException(status_code=404, detail="PieceImage not found.")
    
    print("Files moved and database updated successfully.")
