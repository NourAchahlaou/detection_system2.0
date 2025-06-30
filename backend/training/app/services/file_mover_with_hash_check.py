import hashlib
import os
import shutil
import random
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from training.app.db.models.piece_image import PieceImage

# Helper function to compute the file hash (MD5 in this case)
def compute_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def move_files_if_not_moved(image_folder, annotation_folder, save_image_folder, save_annotation_folder, num_valid_files_to_keep, db: Session):
    # Create source folders if they don't exist
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(annotation_folder, exist_ok=True)

    # Check if the image folder is empty
    if not os.path.exists(image_folder) or not os.listdir(image_folder):
        print(f"Source folder {image_folder} is empty or does not exist.")
        return

    # List source files
    image_files = sorted(os.listdir(image_folder))
    num_files = len(image_files)
    
    if num_files <= num_valid_files_to_keep:
        print(f"Not enough files to move. There are only {num_files} files, which is less than or equal to the number to keep ({num_valid_files_to_keep}).")
        return

    # Filter out images ending with _1 and _2
    files_to_move = [file for file in image_files if not (file.endswith('_1.jpg') or file.endswith('_2.jpg'))]

    # Ensure we don't move more files than we need
    if len(files_to_move) > num_files - num_valid_files_to_keep:
        files_to_move = random.sample(files_to_move, num_files - num_valid_files_to_keep)

    # Create destination folders if they don't exist
    os.makedirs(save_image_folder, exist_ok=True)
    os.makedirs(save_annotation_folder, exist_ok=True)

    # Iterate over the images and check hashes
    for image_file in files_to_move:
        src_image_path = os.path.join(image_folder, image_file)
        dest_image_path = os.path.join(save_image_folder, image_file)

        # Compute hashes if file exists in both source and destination
        if os.path.exists(dest_image_path):
            src_hash = compute_file_hash(src_image_path)
            dest_hash = compute_file_hash(dest_image_path)
            if src_hash == dest_hash:
                print(f"File {image_file} already moved.")
                continue  # Skip moving if files are identical
        
        # Move the image file
        shutil.move(src_image_path, dest_image_path)

        # Move corresponding annotation file
        annotation_file = image_file.replace('.jpg', '.txt')  # Assuming .jpg images and .txt annotations
        src_annotation_path = os.path.join(annotation_folder, annotation_file)
        dest_annotation_path = os.path.join(save_annotation_folder, annotation_file)
        
        if os.path.exists(src_annotation_path):
            if os.path.exists(dest_annotation_path):
                src_annotation_hash = compute_file_hash(src_annotation_path)
                dest_annotation_hash = compute_file_hash(dest_annotation_path)
                if src_annotation_hash == dest_annotation_hash:
                    print(f"Annotation {annotation_file} already moved.")
                    continue
        
            shutil.move(src_annotation_path, dest_annotation_path)

        # Update the database with new file paths and URL
        #update_piece_image_paths(db, image_file, dest_image_path, annotation_file, save_annotation_folder)

        print(f"Moved {image_file} and {annotation_file}.")

def update_piece_image_paths(db: Session, image_file: str, image_path: str, annotation_file: str, annotation_folder: str):
    try:
        # Update the path and URL in the database
        piece_image = db.query(PieceImage).filter(PieceImage.image_name == image_file).first()

        if piece_image:
            piece_image.piece_path = image_path
            piece_image.url = f"/static/images/{image_file}"  # Assuming static files served from this URL
            db.commit()
            print(f"Updated paths for {image_file} in database.")
        else:
            print(f"Piece image not found for {image_file}.")
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error updating database: {e}")

