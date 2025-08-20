import os
import re
import cv2
from fastapi import HTTPException
import numpy as np
import random

def rotate_and_save_images_and_annotations(piece_label: str, rotation_angles: list):
    """Rotate images and update annotations for the specified piece label."""

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
    
    # Get the dataset base path from environment variable
    dataset_base_path = os.getenv('DATASET_BASE_PATH', '/app/shared/dataset')
    
    # Validate piece_label format
    match = re.match(r'([A-Z]\d{3}\.\d{5})', piece_label)
    if not match:
        raise HTTPException(status_code=400, detail="Invalid piece_label format.")
        
    group_label = match.group(1)
    
    # Updated paths to use the Docker volume mount point
    image_folder = os.path.join(dataset_base_path,'piece','piece',piece_label,'images', 'valid')
    annotation_folder = os.path.join(dataset_base_path,'piece','piece',piece_label,'labels', 'valid')
    
    # Create dataset_custom directory structure
    dataset_custom_path = os.path.join(dataset_base_path, 'dataset_custom')
    save_image_folder = os.path.join(dataset_custom_path, 'images', 'train', piece_label)
    save_annotation_folder = os.path.join(dataset_custom_path, 'labels', 'train', piece_label)

    # Create directories if they don't exist
    os.makedirs(save_image_folder, exist_ok=True)
    os.makedirs(save_annotation_folder, exist_ok=True)
    
    # Also create the dataset_custom validation directories for completeness
    os.makedirs(os.path.join(dataset_custom_path, 'images', 'valid'), exist_ok=True)
    os.makedirs(os.path.join(dataset_custom_path, 'labels', 'valid'), exist_ok=True)

    # Check if source directories exist
    if not os.path.exists(image_folder):
        raise HTTPException(status_code=404, detail=f"Source image folder not found: {image_folder}")
    
    if not os.path.exists(annotation_folder):
        raise HTTPException(status_code=404, detail=f"Source annotation folder not found: {annotation_folder}")

    def apply_greyscale(image: np.ndarray, probability: float) -> np.ndarray:
        """Randomly convert image to greyscale with the given probability."""
        if random.random() < probability:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image


    # Process images
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png'))]
    
    if not image_files:
        raise HTTPException(status_code=404, detail=f"No image files found in {image_folder}")

    print(f"Processing {len(image_files)} images from {image_folder}")
    print(f"Saving augmented images to {save_image_folder}")
    print(f"Saving augmented annotations to {save_annotation_folder}")

    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)
        
        if image is None:
            print(f"Warning: Could not load image {image_path}")
            continue

        # Apply the augmentations (same for all transformations)
        augmented_images = [
            apply_greyscale(image, 0.5),
        ]

        for angle in rotation_angles:
            for flip_code in [0, 1]:  # Flip vertically and horizontally
                for i, augmented_image in enumerate(augmented_images):
                    # Rotate image
                    rotated_image = rotate_image(augmented_image, angle)
                    flipped_image = flip_image(rotated_image, flip_code)

                    # Save images
                    new_image_file = f"{group_label}_{angle}_{flip_code}_{i}_{image_file}"
                    new_image_path = os.path.join(save_image_folder, new_image_file)
                    cv2.imwrite(new_image_path, flipped_image)

                    # Read annotations
                    annotation_file = os.path.join(annotation_folder, image_file.replace('.jpg', '.txt').replace('.png', '.txt'))
                    
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
                        # Save annotations
                        new_annotation_file = f"{group_label}_{angle}_{flip_code}_{i}_{image_file.replace('.jpg', '.txt').replace('.png', '.txt')}"
                        new_annotation_path = os.path.join(save_annotation_folder, new_annotation_file)
                        save_annotations(new_annotation_path, valid_annotations)
    
    print(f"Dataset augmentation completed. Files saved to dataset_custom directory.")
    return {
        "message": "Dataset augmentation completed successfully",
        "dataset_custom_path": dataset_custom_path,
        "images_saved": save_image_folder,
        "annotations_saved": save_annotation_folder
    }