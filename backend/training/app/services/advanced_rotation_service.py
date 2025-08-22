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
        """
        Rotate YOLO format annotation correctly.
        
        Args:
            annotation: [class_id, x_center, y_center, width, height] (normalized)
            angle: rotation angle in degrees
            image_size: (height, width) of the image
            
        Returns:
            Rotated annotation in YOLO format
        """
        h, w = image_size  # Note: cv2 image shape is (height, width, channels)
        
        # Extract normalized YOLO coordinates
        class_id = annotation[0]
        x_center_norm = annotation[1]
        y_center_norm = annotation[2] 
        width_norm = annotation[3]
        height_norm = annotation[4]
        
        # Convert normalized coordinates to pixel coordinates
        x_center = x_center_norm * w
        y_center = y_center_norm * h
        bbox_width = width_norm * w
        bbox_height = height_norm * h
        
        # Calculate the four corners of the bounding box
        x1 = x_center - bbox_width / 2
        y1 = y_center - bbox_height / 2
        x2 = x_center + bbox_width / 2
        y2 = y_center - bbox_height / 2
        x3 = x_center + bbox_width / 2
        y3 = y_center + bbox_height / 2
        x4 = x_center - bbox_width / 2
        y4 = y_center + bbox_height / 2
        
        # Create corners array
        corners = np.array([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        
        # Image center for rotation
        cx, cy = w / 2, h / 2
        
        # Convert angle to radians (negative because cv2 rotates clockwise)
        angle_rad = np.radians(-angle)
        
        # Rotation matrix
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        # Rotate each corner around image center
        rotated_corners = []
        for corner in corners:
            x, y = corner
            # Translate to origin
            x_translated = x - cx
            y_translated = y - cy
            
            # Apply rotation
            x_rotated = x_translated * cos_a - y_translated * sin_a
            y_rotated = x_translated * sin_a + y_translated * cos_a
            
            # Translate back
            x_final = x_rotated + cx
            y_final = y_rotated + cy
            
            rotated_corners.append([x_final, y_final])
        
        rotated_corners = np.array(rotated_corners)
        
        # Find the axis-aligned bounding box of the rotated corners
        x_min = np.min(rotated_corners[:, 0])
        y_min = np.min(rotated_corners[:, 1])
        x_max = np.max(rotated_corners[:, 0])
        y_max = np.max(rotated_corners[:, 1])
        
        # Calculate new center and dimensions
        new_x_center = (x_min + x_max) / 2
        new_y_center = (y_min + y_max) / 2
        new_width = x_max - x_min
        new_height = y_max - y_min
        
        # Clamp to image boundaries
        new_x_center = max(0, min(w, new_x_center))
        new_y_center = max(0, min(h, new_y_center))
        new_width = min(w, new_width)
        new_height = min(h, new_height)
        
        # Convert back to normalized coordinates
        new_x_center_norm = new_x_center / w
        new_y_center_norm = new_y_center / h
        new_width_norm = new_width / w
        new_height_norm = new_height / h
        
        # Ensure values are within valid range [0, 1]
        new_x_center_norm = max(0, min(1, new_x_center_norm))
        new_y_center_norm = max(0, min(1, new_y_center_norm))
        new_width_norm = max(0, min(1, new_width_norm))
        new_height_norm = max(0, min(1, new_height_norm))
        
        return [
            class_id,
            new_x_center_norm,
            new_y_center_norm, 
            new_width_norm,
            new_height_norm
        ]
    
    def flip_annotation(annotation: list, flip_code: int, image_size: tuple) -> list:
        """
        Flip YOLO format annotation.
        
        Args:
            annotation: [class_id, x_center, y_center, width, height] (normalized)
            flip_code: 1 for horizontal flip, 0 for vertical flip
            image_size: (height, width) of the image
            
        Returns:
            Flipped annotation in YOLO format
        """
        class_id = annotation[0]
        x_center = annotation[1]
        y_center = annotation[2]
        width = annotation[3]
        height = annotation[4]
        
        if flip_code == 1:  # Horizontal flip
            x_center = 1.0 - x_center
        elif flip_code == 0:  # Vertical flip
            y_center = 1.0 - y_center
        
        return [class_id, x_center, y_center, width, height]

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

                        # Rotate and flip annotations using the corrected functions
                        # Note: Pass image.shape (which is height, width, channels) correctly
                        rotated_annotations = []
                        for annotation in annotations:
                            rotated_ann = rotate_annotation(annotation, angle, image.shape[:2])  # Pass (height, width)
                            if rotated_ann:
                                rotated_annotations.append(rotated_ann)
                        
                        flipped_annotations = []
                        for annotation in rotated_annotations:
                            flipped_ann = flip_annotation(annotation, flip_code, image.shape[:2])  # Pass (height, width)
                            if flipped_ann:
                                flipped_annotations.append(flipped_ann)

                        if flipped_annotations:
                            # Save augmented annotations to train folder
                            new_annotation_file = f"{group_label}_{piece_label}_{angle}_{flip_code}_{i}_{image_file.replace('.jpg', '.txt').replace('.png', '.txt')}"
                            new_annotation_path = os.path.join(dest_labels_train, new_annotation_file)
                            save_annotations(new_annotation_path, flipped_annotations)
                            
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