import re
from fastapi import HTTPException
from sqlalchemy.orm import Session
from training.app.services.advanced_rotation_service import rotate_and_save_images_and_annotations
from training.app.db.models.piece import Piece

def rotate_and_update_images(piece_label: str, db: Session):
    # Retrieve the piece and its image record
    piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
    if not piece:
        raise HTTPException(status_code=404, detail="Piece not found.")
    rotate_and_save_images_and_annotations(piece_label, rotation_angles=[45, 90, 135, 180, 270])
    print("Files moved and database updated successfully.")

