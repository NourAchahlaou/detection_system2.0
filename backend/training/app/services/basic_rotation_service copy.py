import re
from typing import List
from fastapi import HTTPException
from sqlalchemy.orm import Session
from training.app.services.advanced_rotation_service import rotate_and_save_images_and_annotations
from training.app.db.models.piece import Piece

def rotate_and_update_images(piece_labels: List[str], db):
    """Updated function to handle group-based rotation with multiple pieces."""
    # Ensure piece_labels is a list
    if isinstance(piece_labels, str):
        piece_labels = [piece_labels]
    
    # Default rotation angles - you can modify these as needed
    rotation_angles = [45, 90, 135, 180, 270]
    
    try:
        result = rotate_and_save_images_and_annotations(piece_labels, rotation_angles)
        print(f"Successfully completed rotation for group with pieces: {piece_labels}")
        return result
    except Exception as e:
        print(f"Error during rotation for pieces {piece_labels}: {str(e)}")
        raise e