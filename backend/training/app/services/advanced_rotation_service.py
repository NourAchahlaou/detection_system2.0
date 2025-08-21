import os
import re
import cv2
from fastapi import HTTPException
import numpy as np
import random
import shutil
import yaml
from typing import List

def rotate_and_save_images_and_annotations(piece_labels: List[str], rotation_angles: list):
    """Copy existing YOLO structure and apply augmentations for multiple piece labels in a group-based structure."""

    def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
        """Rotate the image by the given angle."""
        center = (image.shape[1] // 2, image.shape[0] // 2)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_image = cv2.warpAffine(image, matrix, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        return rotated_image

    def flip_image(image: np.ndarray, flip_code: int) -> np.ndarray:
        """Flip the image: 1 for horizontal, 0 for vertical."""
        return cv2.flip(image, flip_code)
    
    def rotate_annotation(annotation: list, angle: float, image_size: tuple) -> list:
        w, h = image_size
        x_center, y_center, width, height = annotation[1] * w, annotation[2] * h, annotation[3] * w, annotation[4] * h
        x_min, y_min = x_center - width / 2, y_center - height / 2
        x_max, y_max = x_center + width / 2, y_center + height / 2

        angle_rad = np.radians(-angle)
        cos_angle, sin_angle = np.cos(angle_rad), np.sin(angle_rad)
        cx, cy = w / 2, h / 2

        def rotate_point(x, y):
            return (
                cos_angle * (x - cx) - sin_angle * (y - cy) + cx,
                sin_angle * (x - cx) + cos_angle * (y - cy) + cy
            )

        corners = [rotate_point(*p) for p in [(x_min, y_min), (x_max, y_min), (x_max, y_max), (x_min, y_max)]]
        x_min_new, y_min_new = max(0, min(x for x, _ in corners)), max(0, min(y for _, y in corners))
        x_max_new, y_max_new = min(w, max(x for x, _ in corners)), min(h, max(y for _, y in corners))

        if x_max_new <= x_min_new or y_max_new <= y_min_new:
            return None  # Discard invalid bounding box

        new_x_center = (x_min_new + x_max_new) / 2 / w
        new_y_center = (y_min_new + y_max_new) / 2 / h
        new_width = (x_max_new - x_min_new) / w
        new_height = (y_max_new - y_min_new) / h

        if 0 <= new_x_center <= 1 and 0 <= new_y_center <= 1 and new_width > 0 and new_height > 0:
            return [annotation[0], new_x_center, new_y_center, new_width, new_height]
        return None

    def flip_annotation(annotation: list, flip_code: int, image_size: tuple) -> list:
        w, h = image_size
        x_center, y_center = annotation[1], annotation[2]
        if flip_code == 1:  # Horizontal flip
            x_center = 1 - x_center
        elif flip_code == 0:  # Vertical flip
            y_center = 1 - y_center

        return [annotation[0], x_center, y_center, annotation[3], annotation[4]]

    def save_annotations(annotation_file: str, annotations: list):
        """Save annotations to a file."""
        with open(annotation_file, 'w') as file:
            for annotation in annotations:
                line = f"{int(annotation[0])} {annotation[1]} {annotation[2]} {annotation[3]} {annotation[4]}\n"
                file.write(line)

    def apply_greyscale(image: np.ndarray, probability: float) -> np.ndarray:
        """Randomly convert image to greyscale with the given probability."""
        if random.random() < probability:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # Convert back to 3-channel
        return image

    def copy_directory_contents(src_dir: str, dst_dir: str):
        """Copy all contents from source directory to destination directory."""
        if os.path.exists(src_dir):
            os.makedirs(dst_dir, exist_ok=True)
            for item in os.listdir(src_dir):
                src_item = os.path.join(src_dir, item)
                dst_item = os.path.join(dst_dir, item)
                if os.path.isfile(src_item):
                    shutil.copy2(src_item, dst_item)
                elif os.path.isdir(src_item):
                    shutil.copytree(src_item, dst_item, dirs_exist_ok=True)
            print(f"Copied contents from {src_dir} to {dst_dir}")
        else:
            print(f"Warning: Source directory does not exist: {src_dir}")

    def copy_and_update_data_yaml(group_label: str, dataset_base_path: str, dataset_custom_path: str):
        """Copy group data.yaml and update the path to point to custom dataset."""
        # Source data.yaml path
        src_yaml_path = os.path.join(dataset_base_path, 'piece', 'piece', group_label, 'data.yaml')
        # Destination data.yaml path
        dst_yaml_path = os.path.join(dataset_custom_path, 'data.yaml')
        train = os.path.join(dataset_custom_path, 'images', 'train')
        valid_image = os.path.join(dataset_custom_path, 'images', 'valid')
        if os.path.exists(src_yaml_path):
            # Load the existing YAML
            with open(src_yaml_path, 'r') as yaml_file:
                yaml_data = yaml.safe_load(yaml_file)
            
            # Update the path to point to the custom dataset
            yaml_data.pop('path', None)
            # Keep the same relative paths for train and val
            yaml_data['train'] = train
            yaml_data['val'] = valid_image 
            
            # Write the updated YAML to the destination
            os.makedirs(os.path.dirname(dst_yaml_path), exist_ok=True)
            with open(dst_yaml_path, 'w') as yaml_file:
                yaml.dump(yaml_data, yaml_file, default_flow_style=False)
            
            print(f"Copied and updated data.yaml from {src_yaml_path} to {dst_yaml_path}")
            return True
        else:
            print(f"Warning: Source data.yaml not found at {src_yaml_path}")
            return False
    
    # Ensure piece_labels is a list
    if isinstance(piece_labels, str):
        piece_labels = [piece_labels]
    
    # Extract group from first piece label (assuming all pieces in the same group)
    match = re.match(r'([A-Z]\d{3})', piece_labels[0])
    if not match:
        raise HTTPException(status_code=400, detail="Invalid piece_label format.")
    
    group_label = match.group(1)
    
    # Get the dataset base path from environment variable
    dataset_base_path = os.getenv('DATASET_BASE_PATH', '/app/shared/dataset')
    
    # Custom dataset structure
    dataset_custom_path = os.path.join(dataset_base_path, 'dataset_custom', group_label)
    
    # Create base custom dataset directory
    os.makedirs(dataset_custom_path, exist_ok=True)
    
    print(f"Processing group: {group_label}")
    print(f"Custom dataset path: {dataset_custom_path}")
    
    # Copy and update the group's data.yaml first
    copy_and_update_data_yaml(group_label, dataset_base_path, dataset_custom_path)
    
    total_images_processed = 0
    total_augmented_images = 0
    
    # Process each piece in the group
    for piece_label in piece_labels:
        print(f"Processing piece: {piece_label}")
        
        # Validate piece_label format
        match = re.match(r'([A-Z]\d{3}\.\d{5})', piece_label)
        if not match:
            print(f"Warning: Invalid piece_label format for {piece_label}, skipping...")
            continue
        
        # Source paths for this piece (from the new structure)
        piece_source_path = os.path.join(dataset_base_path, 'piece', 'piece', group_label, piece_label)
        source_images_valid = os.path.join(piece_source_path, 'images', 'valid')
        source_labels_valid = os.path.join(piece_source_path, 'labels', 'valid')
        
        # Destination paths in custom dataset
        piece_custom_path = os.path.join(dataset_custom_path)
        dest_images_valid = os.path.join(piece_custom_path, 'images', 'valid',piece_label)
        dest_labels_valid = os.path.join(piece_custom_path, 'labels', 'valid',piece_label)
        dest_images_train = os.path.join(piece_custom_path, 'images', 'train',piece_label)
        dest_labels_train = os.path.join(piece_custom_path, 'labels', 'train',piece_label)
        
        # Create destination directories
        for dir_path in [dest_images_valid, dest_labels_valid, dest_images_train, dest_labels_train]:
            os.makedirs(dir_path, exist_ok=True)
        
        # Copy valid data as-is
        print(f"Copying valid data for piece {piece_label}")
        copy_directory_contents(source_images_valid, dest_images_valid)
        copy_directory_contents(source_labels_valid, dest_labels_valid)
        
        # Check if source directories exist for augmentation
        if not os.path.exists(source_images_valid):
            print(f"Warning: Source image folder not found: {source_images_valid}, skipping piece {piece_label}")
            continue
        
        if not os.path.exists(source_labels_valid):
            print(f"Warning: Source annotation folder not found: {source_labels_valid}, skipping piece {piece_label}")
            continue

        # Get all image files for augmentation (from valid folder)
        image_files = [f for f in os.listdir(source_images_valid) if f.endswith(('.jpg', '.png'))]
        
        if not image_files:
            print(f"Warning: No image files found in {source_images_valid} for piece {piece_label}")
            continue

        print(f"Found {len(image_files)} images for augmentation in piece {piece_label}")

        # Apply augmentations and save to train folder
        for image_file in image_files:
            image_path = os.path.join(source_images_valid, image_file)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"Warning: Could not load image {image_path}")
                continue

            # Apply the augmentations
            augmented_images = [
                apply_greyscale(image, 0.5),
            ]

            for angle in rotation_angles:
                for flip_code in [0, 1]:  # Flip vertically and horizontally
                    for i, augmented_image in enumerate(augmented_images):
                        # Rotate image
                        rotated_image = rotate_image(augmented_image, angle)
                        flipped_image = flip_image(rotated_image, flip_code)

                        # Save augmented images to train folder
                        new_image_file = f"{group_label}_{piece_label}_{angle}_{flip_code}_{i}_{image_file}"
                        new_image_path = os.path.join(dest_images_train, new_image_file)
                        cv2.imwrite(new_image_path, flipped_image)

                        # Read corresponding annotations
                        annotation_file = os.path.join(source_labels_valid, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
                        
                        if not os.path.exists(annotation_file):
                            print(f"Warning: Annotation file not found: {annotation_file}")
                            continue
                            
                        with open(annotation_file, 'r') as file:
                            annotations = [line.strip().split() for line in file.readlines()]
                            annotations = [list(map(float, annotation)) for annotation in annotations]

                        # Rotate and flip annotations
                        rotated_annotations = [rotate_annotation(annotation, angle, image.shape[:2]) for annotation in annotations]
                        flipped_annotations = [flip_annotation(annotation, flip_code, image.shape[:2]) for annotation in rotated_annotations if annotation]
                        valid_annotations = [annotation for annotation in flipped_annotations if annotation]

                        if valid_annotations:
                            # Save augmented annotations to train folder
                            new_annotation_file = f"{group_label}_{piece_label}_{angle}_{flip_code}_{i}_{image_file.replace('.jpg', '.txt').replace('.png', '.txt')}"
                            new_annotation_path = os.path.join(dest_labels_train, new_annotation_file)
                            save_annotations(new_annotation_path, valid_annotations)
                            
                            total_augmented_images += 1

            total_images_processed += len(image_files)
    
    print(f"Group {group_label} dataset creation completed!")
    print(f"Total original images processed: {total_images_processed}")
    print(f"Total augmented images created: {total_augmented_images}")
    print(f"Custom dataset saved to: {dataset_custom_path}")
    
    return {
        "message": "Custom dataset creation completed successfully",
        "group_label": group_label,
        "dataset_custom_path": dataset_custom_path,
        "total_pieces_processed": len(piece_labels),
        "total_original_images": total_images_processed,
        "total_augmented_images": total_augmented_images,
        "structure_copied": True,
        "data_yaml_updated": True
    }