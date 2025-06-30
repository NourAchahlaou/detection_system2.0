import hashlib
import os
import shutil
import random
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from training.app.db.models.piece_image import PieceImage

def compute_file_hash(filepath):
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

def update_piece_image_paths(db: Session, image_file: str, image_path: str, annotation_file: str, annotation_folder: str):
    try:
        piece_image = db.query(PieceImage).filter(PieceImage.image_name == image_file).first()
        if piece_image:
            piece_image.piece_path = image_path
            piece_image.url = f"/static/images/{image_file}"  # adjust as needed
            db.commit()
            print(f"Updated paths for {image_file} in database.")
        else:
            print(f"Piece image not found for {image_file}.")
    except SQLAlchemyError as e:
        db.rollback()
        print(f"Error updating database: {e}")

def split_and_move_files(
    image_folder, 
    annotation_folder, 
    train_image_folder, 
    train_annotation_folder, 
    valid_image_folder, 
    valid_annotation_folder,
    db: Session
):
    # Ensure source folders exist
    os.makedirs(image_folder, exist_ok=True)
    os.makedirs(annotation_folder, exist_ok=True)
    
    if not os.path.exists(image_folder) or not os.listdir(image_folder):
        print(f"Source folder {image_folder} is empty or does not exist.")
        return

    # Get all JPG images from source
    all_images = sorted([f for f in os.listdir(image_folder) if f.lower().endswith('.jpg')])
    total_images = len(all_images)
    if total_images == 0:
        print("No images found in the source folder.")
        return

    # Compute desired number of validation images as 15% of the total.
    # Always keep files ending with _1.jpg in the validation set.
    valid_always = [f for f in all_images if f.endswith('_1.jpg')]
    # The desired validation count is at least the number of _1 files,
    # but ideally should be 15% of total images.
    desired_valid_count = max(len(valid_always), int(round(total_images * 0.15)))
    # Prevent case where desired_valid_count equals total images.
    if desired_valid_count >= total_images:
        desired_valid_count = total_images - 1

    # The remaining images are candidates for either set.
    candidates = [f for f in all_images if not f.endswith('_1.jpg')]

    # Determine how many additional files we need in validation (beyond those ending in _1.jpg)
    additional_needed = max(0, desired_valid_count - len(valid_always))
    additional_valid = random.sample(candidates, additional_needed) if additional_needed > 0 and candidates else []

    # Final validation set is the _1 files plus any additional chosen ones.
    valid_set = set(valid_always + additional_valid)
    # Training set is whatever is left.
    train_set = [f for f in all_images if f not in valid_set]

    # Create destination folders if they don't exist
    for folder in [train_image_folder, train_annotation_folder, valid_image_folder, valid_annotation_folder]:
        os.makedirs(folder, exist_ok=True)

    def move_file(src, dest):
        if os.path.exists(dest):
            src_hash = compute_file_hash(src)
            dest_hash = compute_file_hash(dest)
            if src_hash == dest_hash:
                print(f"File {os.path.basename(src)} already moved.")
                return
        shutil.move(src, dest)
        print(f"Moved {os.path.basename(src)}.")

    # Move training images and annotations
    for file in train_set:
        src_image = os.path.join(image_folder, file)
        dest_image = os.path.join(train_image_folder, file)
        move_file(src_image, dest_image)

        # Process corresponding annotation file (assuming .jpg → .txt)
        annotation_file = file.replace('.jpg', '.txt')
        src_ann = os.path.join(annotation_folder, annotation_file)
        dest_ann = os.path.join(train_annotation_folder, annotation_file)
        if os.path.exists(src_ann):
            move_file(src_ann, dest_ann)
        else:
            print(f"Annotation {annotation_file} not found for training image {file}.")

        # Optionally update DB paths for training images
        # update_piece_image_paths(db, file, dest_image, annotation_file, train_annotation_folder)

    # Move validation images and annotations
    for file in valid_set:
        src_image = os.path.join(image_folder, file)
        dest_image = os.path.join(valid_image_folder, file)
        move_file(src_image, dest_image)

        annotation_file = file.replace('.jpg', '.txt')
        src_ann = os.path.join(annotation_folder, annotation_file)
        dest_ann = os.path.join(valid_annotation_folder, annotation_file)
        if os.path.exists(src_ann):
            move_file(src_ann, dest_ann)
        else:
            print(f"Annotation {annotation_file} not found for validation image {file}.")

        # Optionally update DB paths for validation images
        # update_piece_image_paths(db, file, dest_image, annotation_file, valid_annotation_folder)

    print(f"Splitting complete. Total images: {total_images}.")
    print(f"Training images: {len(train_set)} ({len(train_set)/total_images*100:.1f}%).")
    print(f"Validation images: {len(valid_set)} ({len(valid_set)/total_images*100:.1f}%).")
