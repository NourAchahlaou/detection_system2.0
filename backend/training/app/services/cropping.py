import os
import cv2
import numpy as np
import yaml
import shutil
from typing import List, Tuple, Dict
import logging
from collections import defaultdict
import random
import json

logger = logging.getLogger(__name__)

class SmartCroppingService:
    def __init__(self, crop_size: int = 980, final_size: int = 512):
        """
        Enhanced smart cropping service with comprehensive NaN prevention.
        
        Args:
            crop_size: The consistent crop size to extract from original images (e.g., 980x980)
            final_size: The final size for training (e.g., 512x512)
        """
        self.crop_size = crop_size
        self.final_size = final_size
        
        # ULTRA-STRICT validation thresholds to prevent NaN
        self.min_bbox_size = 0.03  # Minimum 3% of image dimension (increased from 0.02)
        self.min_normalized_coord = 0.01  # Keep annotations well away from edges
        self.max_normalized_coord = 0.99  # Keep annotations well away from edges
        self.min_area_threshold = 0.001  # Minimum area 0.1% (increased)
        self.min_overlap_ratio = 0.7  # CRITICAL: 70% overlap required (was 0.5)
        self.max_aspect_ratio = 8.0  # Maximum aspect ratio (reduced from 10.0)
        self.min_pixel_size = 8  # Minimum bbox size in pixels after cropping
        
        # Additional validation parameters
        self.edge_safety_margin = 0.02  # 2% safety margin from edges
        self.min_validation_samples = 3  # Minimum samples needed for validation split
        
    def is_valid_normalized_annotation(self, x_center_norm: float, y_center_norm: float, 
                                     width_norm: float, height_norm: float) -> bool:
        """
        ULTRA-STRICT validation for normalized coordinates.
        """
        # Basic bounds check with safety margins
        if not (self.min_normalized_coord <= x_center_norm <= self.max_normalized_coord and
                self.min_normalized_coord <= y_center_norm <= self.max_normalized_coord and
                self.min_bbox_size <= width_norm <= (1.0 - 2 * self.edge_safety_margin) and
                self.min_bbox_size <= height_norm <= (1.0 - 2 * self.edge_safety_margin)):
            return False
        
        # Check bounding box doesn't exceed image boundaries with strict margins
        half_width = width_norm / 2
        half_height = height_norm / 2
        
        x_min = x_center_norm - half_width
        x_max = x_center_norm + half_width
        y_min = y_center_norm - half_height
        y_max = y_center_norm + half_height
        
        # ULTRA-STRICT boundary check
        if not (self.edge_safety_margin <= x_min and x_max <= (1.0 - self.edge_safety_margin) and 
                self.edge_safety_margin <= y_min and y_max <= (1.0 - self.edge_safety_margin)):
            return False
        
        # Enhanced area check
        area = width_norm * height_norm
        if area < self.min_area_threshold:
            return False
        
        # Enhanced aspect ratio check
        aspect_ratio = max(width_norm, height_norm) / min(width_norm, height_norm)
        if aspect_ratio > self.max_aspect_ratio:
            return False
        
        # Check for degenerate cases
        if width_norm <= 0 or height_norm <= 0:
            return False
            
        return True
    
    def validate_pixel_coordinates(self, x_center: float, y_center: float, 
                                 bbox_width: float, bbox_height: float,
                                 image_width: int, image_height: int) -> bool:
        """
        Validate pixel coordinates before processing.
        """
        # Check if coordinates are within image bounds
        if not (0 <= x_center <= image_width and 0 <= y_center <= image_height):
            return False
        
        # Check if bbox dimensions are reasonable
        if bbox_width < 1 or bbox_height < 1:
            return False
            
        if bbox_width > image_width or bbox_height > image_height:
            return False
        
        # Check bbox boundaries
        x_min = x_center - bbox_width / 2
        x_max = x_center + bbox_width / 2
        y_min = y_center - bbox_height / 2
        y_max = y_center + bbox_height / 2
        
        if x_min < 0 or x_max > image_width or y_min < 0 or y_max > image_height:
            return False
            
        return True
        
    def read_yolo_annotation(self, annotation_path: str, image_width: int, image_height: int) -> List[Dict]:
        """
        Enhanced YOLO annotation reading with comprehensive validation.
        """
        annotations = []
        
        if not os.path.exists(annotation_path):
            return annotations
        
        # Check file size first
        if os.path.getsize(annotation_path) == 0:
            logger.debug(f"Empty annotation file: {annotation_path}")
            return annotations
            
        with open(annotation_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line or line.startswith('#'):  # Skip empty lines and comments
                    continue
                    
                parts = line.split()
                if len(parts) < 5:
                    logger.warning(f"Malformed annotation at {annotation_path}:{line_num} - insufficient parts")
                    continue
                
                try:
                    class_id = int(parts[0])
                    if class_id < 0:
                        logger.warning(f"Invalid class_id {class_id} at {annotation_path}:{line_num}")
                        continue
                        
                    x_center_norm = float(parts[1])
                    y_center_norm = float(parts[2])
                    width_norm = float(parts[3])
                    height_norm = float(parts[4])
                    
                    # Check for NaN/inf values
                    values = [x_center_norm, y_center_norm, width_norm, height_norm]
                    if any(not np.isfinite(v) for v in values):
                        logger.warning(f"Non-finite values at {annotation_path}:{line_num}")
                        continue
                    
                    # Validate normalized coordinates BEFORE pixel conversion
                    if not self.is_valid_normalized_annotation(x_center_norm, y_center_norm, width_norm, height_norm):
                        logger.debug(f"Invalid normalized annotation at {annotation_path}:{line_num}")
                        continue
                    
                    # Convert to pixel coordinates with careful bounds checking
                    x_center = x_center_norm * image_width
                    y_center = y_center_norm * image_height
                    bbox_width = width_norm * image_width
                    bbox_height = height_norm * image_height
                    
                    # Validate pixel coordinates
                    if not self.validate_pixel_coordinates(x_center, y_center, bbox_width, bbox_height, 
                                                         image_width, image_height):
                        logger.debug(f"Invalid pixel coordinates at {annotation_path}:{line_num}")
                        continue
                    
                    annotations.append({
                        'class_id': class_id,
                        'x_center': x_center,
                        'y_center': y_center,
                        'bbox_width': bbox_width,
                        'bbox_height': bbox_height,
                        'x_center_norm': x_center_norm,
                        'y_center_norm': y_center_norm,
                        'width_norm': width_norm,
                        'height_norm': height_norm
                    })
                    
                except (ValueError, IndexError, OverflowError) as e:
                    logger.warning(f"Parse error at {annotation_path}:{line_num}: {e}")
                    continue
                
        logger.debug(f"Loaded {len(annotations)} valid annotations from {annotation_path}")
        return annotations
    
    def calculate_smart_crop_center(self, annotations: List[Dict], image_width: int, image_height: int) -> Tuple[int, int]:
        """
        Calculate optimal crop center with enhanced logic to maximize annotation retention.
        """
        if not annotations:
            return image_width // 2, image_height // 2
        
        # Calculate bounding box of all annotations
        min_x = min(ann['x_center'] - ann['bbox_width']/2 for ann in annotations)
        max_x = max(ann['x_center'] + ann['bbox_width']/2 for ann in annotations)
        min_y = min(ann['y_center'] - ann['bbox_height']/2 for ann in annotations)
        max_y = max(ann['y_center'] + ann['bbox_height']/2 for ann in annotations)
        
        # Calculate the center of the bounding box containing all annotations
        annotations_center_x = (min_x + max_x) / 2
        annotations_center_y = (min_y + max_y) / 2
        
        # Calculate the size needed to contain all annotations
        needed_width = max_x - min_x
        needed_height = max_y - min_y
        
        # If annotations fit comfortably in crop size, use their center
        crop_margin = self.crop_size * 0.1  # 10% margin
        if needed_width <= (self.crop_size - crop_margin) and needed_height <= (self.crop_size - crop_margin):
            center_x = annotations_center_x
            center_y = annotations_center_y
        else:
            # If annotations are spread out, use weighted centroid
            total_weight = 0
            weighted_x = 0
            weighted_y = 0
            
            for ann in annotations:
                # Weight by area and distance from edges
                area = ann['bbox_width'] * ann['bbox_height']
                edge_distance = min(ann['x_center'], image_width - ann['x_center'],
                                  ann['y_center'], image_height - ann['y_center'])
                weight = area * (1 + edge_distance / min(image_width, image_height))
                
                weighted_x += ann['x_center'] * weight
                weighted_y += ann['y_center'] * weight
                total_weight += weight
            
            center_x = weighted_x / total_weight if total_weight > 0 else image_width // 2
            center_y = weighted_y / total_weight if total_weight > 0 else image_height // 2
        
        # Ensure crop center allows for full crop within image bounds
        half_crop = self.crop_size // 2
        center_x = max(half_crop, min(image_width - half_crop, center_x))
        center_y = max(half_crop, min(image_height - half_crop, center_y))
        
        return int(center_x), int(center_y)
    
    def calculate_precise_crop_bounds(self, center_x: int, center_y: int, 
                                    image_width: int, image_height: int) -> Tuple[int, int, int, int]:
        """
        Calculate precise crop bounds with validation.
        """
        half_crop = self.crop_size // 2
        
        # Calculate ideal bounds
        x1 = center_x - half_crop
        y1 = center_y - half_crop
        x2 = center_x + half_crop
        y2 = center_y + half_crop
        
        # Adjust for image boundaries while preserving crop size as much as possible
        if x1 < 0:
            shift = -x1
            x1 = 0
            x2 = min(x2 + shift, image_width)
        elif x2 > image_width:
            shift = x2 - image_width
            x2 = image_width
            x1 = max(x1 - shift, 0)
            
        if y1 < 0:
            shift = -y1
            y1 = 0
            y2 = min(y2 + shift, image_height)
        elif y2 > image_height:
            shift = y2 - image_height
            y2 = image_height
            y1 = max(y1 - shift, 0)
        
        # Validate final crop dimensions
        crop_width = x2 - x1
        crop_height = y2 - y1
        min_crop_size = min(self.crop_size // 2, min(image_width, image_height))
        
        if crop_width < min_crop_size or crop_height < min_crop_size:
            logger.warning(f"Crop size too small ({crop_width}x{crop_height}), using maximum possible")
            max_possible = min(image_width, image_height)
            half_max = max_possible // 2
            center_x = image_width // 2
            center_y = image_height // 2
            
            x1 = max(0, center_x - half_max)
            y1 = max(0, center_y - half_max)
            x2 = min(image_width, x1 + max_possible)
            y2 = min(image_height, y1 + max_possible)
        
        return x1, y1, x2, y2
    
    def update_annotations_for_crop_enhanced(self, annotations: List[Dict], 
                                           x1: int, y1: int, crop_width: int, crop_height: int) -> List[Dict]:
        """
        ENHANCED annotation updating with strict validation and filtering.
        """
        updated_annotations = []
        
        for ann in annotations:
            # Transform coordinates to crop space
            crop_x_center = ann['x_center'] - x1
            crop_y_center = ann['y_center'] - y1
            
            # Original bbox dimensions remain the same
            bbox_width = ann['bbox_width']
            bbox_height = ann['bbox_height']
            
            # Calculate bbox boundaries in crop space
            bbox_x1 = crop_x_center - bbox_width / 2
            bbox_y1 = crop_y_center - bbox_height / 2
            bbox_x2 = crop_x_center + bbox_width / 2
            bbox_y2 = crop_y_center + bbox_height / 2
            
            # Calculate intersection with crop boundaries
            intersect_x1 = max(0, bbox_x1)
            intersect_y1 = max(0, bbox_y1)
            intersect_x2 = min(crop_width, bbox_x2)
            intersect_y2 = min(crop_height, bbox_y2)
            
            # Validate intersection
            intersect_width = max(0, intersect_x2 - intersect_x1)
            intersect_height = max(0, intersect_y2 - intersect_y1)
            
            if intersect_width <= 0 or intersect_height <= 0:
                continue
            
            # Check minimum pixel size after cropping
            if intersect_width < self.min_pixel_size or intersect_height < self.min_pixel_size:
                logger.debug(f"Annotation too small after cropping: {intersect_width}x{intersect_height}")
                continue
            
            # Calculate overlap ratio
            original_area = bbox_width * bbox_height
            intersect_area = intersect_width * intersect_height
            overlap_ratio = intersect_area / original_area if original_area > 0 else 0
            
            if overlap_ratio < self.min_overlap_ratio:
                logger.debug(f"Insufficient overlap: {overlap_ratio:.3f} < {self.min_overlap_ratio}")
                continue
            
            # Use intersection as new bbox
            new_center_x = (intersect_x1 + intersect_x2) / 2
            new_center_y = (intersect_y1 + intersect_y2) / 2
            new_width = intersect_width
            new_height = intersect_height
            
            # Convert to normalized coordinates
            new_x_center_norm = new_center_x / crop_width
            new_y_center_norm = new_center_y / crop_height
            new_width_norm = new_width / crop_width
            new_height_norm = new_height / crop_height
            
            # Apply strict bounds with clamping
            new_x_center_norm = np.clip(new_x_center_norm, self.min_normalized_coord, self.max_normalized_coord)
            new_y_center_norm = np.clip(new_y_center_norm, self.min_normalized_coord, self.max_normalized_coord)
            new_width_norm = np.clip(new_width_norm, self.min_bbox_size, 1.0 - 2*self.edge_safety_margin)
            new_height_norm = np.clip(new_height_norm, self.min_bbox_size, 1.0 - 2*self.edge_safety_margin)
            
            # Final validation
            if self.is_valid_normalized_annotation(new_x_center_norm, new_y_center_norm, 
                                                 new_width_norm, new_height_norm):
                updated_annotations.append({
                    'class_id': ann['class_id'],
                    'x_center_norm': new_x_center_norm,
                    'y_center_norm': new_y_center_norm,
                    'width_norm': new_width_norm,
                    'height_norm': new_height_norm,
                    'overlap_ratio': overlap_ratio
                })
            else:
                logger.debug(f"Final validation failed for normalized coordinates")
        
        return updated_annotations
    
    def save_validated_annotation(self, annotation_path: str, annotations: List[Dict]):
        """
        Save annotations with final validation and format consistency.
        """
        if not annotations:
            # Remove empty annotation files
            if os.path.exists(annotation_path):
                os.remove(annotation_path)
                logger.debug(f"Removed empty annotation file: {annotation_path}")
            return
        
        os.makedirs(os.path.dirname(annotation_path), exist_ok=True)
        
        # Final validation pass
        valid_annotations = []
        for ann in annotations:
            # Check all required fields
            required_fields = ['class_id', 'x_center_norm', 'y_center_norm', 'width_norm', 'height_norm']
            if not all(field in ann for field in required_fields):
                continue
            
            # Validate data types and ranges
            try:
                class_id = int(ann['class_id'])
                coords = [float(ann[field]) for field in required_fields[1:]]
                
                if class_id < 0 or any(not np.isfinite(coord) for coord in coords):
                    continue
                
                if self.is_valid_normalized_annotation(coords[0], coords[1], coords[2], coords[3]):
                    valid_annotations.append({
                        'class_id': class_id,
                        'x_center_norm': coords[0],
                        'y_center_norm': coords[1],
                        'width_norm': coords[2],
                        'height_norm': coords[3]
                    })
            except (ValueError, KeyError, TypeError):
                continue
        
        if not valid_annotations:
            if os.path.exists(annotation_path):
                os.remove(annotation_path)
            return
        
        # Write with high precision and consistent formatting
        with open(annotation_path, 'w') as f:
            for ann in valid_annotations:
                line = f"{ann['class_id']} {ann['x_center_norm']:.8f} {ann['y_center_norm']:.8f} {ann['width_norm']:.8f} {ann['height_norm']:.8f}\n"
                f.write(line)
        
        logger.debug(f"Saved {len(valid_annotations)} valid annotations to {annotation_path}")
    
    def fix_dataset_paths_in_yaml(self, yaml_path: str, dataset_path: str) -> bool:
        """
        CRITICAL FIX: Fix the paths in data.yaml to use absolute paths that actually exist.
        """
        try:
            # Create the actual directory structure
            train_images_dir = os.path.join(dataset_path, 'images', 'train')
            valid_images_dir = os.path.join(dataset_path, 'images', 'valid')
            
            os.makedirs(train_images_dir, exist_ok=True)
            os.makedirs(valid_images_dir, exist_ok=True)
            
            # Read existing yaml or create new one
            yaml_data = {}
            if os.path.exists(yaml_path):
                with open(yaml_path, 'r') as f:
                    yaml_data = yaml.safe_load(f) or {}
            
            # CRITICAL: Use absolute paths that exist
            yaml_data.update({
                'path': dataset_path,
                'train': os.path.join(dataset_path, 'images', 'train'),
                'val': os.path.join(dataset_path, 'images', 'valid'),
                'names': yaml_data.get('names', {0: 'object'})  # Default class if missing
            })
            
            # Write corrected yaml
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=False)
            
            # Verify paths exist
            train_exists = os.path.exists(yaml_data['train'])
            val_exists = os.path.exists(yaml_data['val'])
            
            logger.info(f"Updated data.yaml - Train path exists: {train_exists}, Val path exists: {val_exists}")
            return train_exists and val_exists
            
        except Exception as e:
            logger.error(f"Failed to fix dataset paths: {e}")
            return False
    
    def ensure_minimum_validation_split(self, dataset_path: str, min_ratio: float = 0.15) -> Dict:
        """
        ENHANCED: Ensure minimum validation split with better distribution.
        """
        result = {
            'action_taken': 'none',
            'moved_count': 0,
            'train_before': 0,
            'valid_before': 0,
            'train_after': 0,
            'valid_after': 0,
            'final_ratio': 0.0,
            'error': None
        }
        
        try:
            train_images_dir = os.path.join(dataset_path, 'images', 'train')
            valid_images_dir = os.path.join(dataset_path, 'images', 'valid')
            train_labels_dir = os.path.join(dataset_path, 'labels', 'train')
            valid_labels_dir = os.path.join(dataset_path, 'labels', 'valid')
            
            # Create directories
            for dir_path in [train_images_dir, valid_images_dir, train_labels_dir, valid_labels_dir]:
                os.makedirs(dir_path, exist_ok=True)
            
            # Get current valid pairs
            train_pairs = self.get_all_valid_pairs(train_images_dir, train_labels_dir)
            valid_pairs = self.get_all_valid_pairs(valid_images_dir, valid_labels_dir)
            
            result['train_before'] = len(train_pairs)
            result['valid_before'] = len(valid_pairs)
            
            total_pairs = len(train_pairs) + len(valid_pairs)
            if total_pairs == 0:
                result['error'] = 'No valid image-annotation pairs found'
                return result
            
            current_ratio = len(valid_pairs) / total_pairs
            result['final_ratio'] = current_ratio
            
            # Check if we need more validation samples
            min_valid_count = max(self.min_validation_samples, int(total_pairs * min_ratio))
            needed_valid = min_valid_count - len(valid_pairs)
            
            if needed_valid <= 0:
                result['action_taken'] = 'sufficient_validation'
                result['train_after'] = result['train_before']
                result['valid_after'] = result['valid_before']
                return result
            
            # Move pairs from train to validation
            pairs_to_move = min(needed_valid, len(train_pairs) // 2)  # Don't move more than half
            
            if pairs_to_move <= 0:
                result['action_taken'] = 'insufficient_training_data'
                result['train_after'] = result['train_before']
                result['valid_after'] = result['valid_before']
                return result
            
            # Use stratified sampling if possible (by class distribution)
            # For now, use random sampling with fixed seed
            random.seed(42)
            selected_pairs = random.sample(train_pairs, pairs_to_move)
            
            moved_count = 0
            for image_path, label_path in selected_pairs:
                try:
                    # Calculate destination paths
                    image_rel = os.path.relpath(image_path, train_images_dir)
                    label_rel = os.path.relpath(label_path, train_labels_dir)
                    
                    dst_image = os.path.join(valid_images_dir, image_rel)
                    dst_label = os.path.join(valid_labels_dir, label_rel)
                    
                    # Create destination directories
                    os.makedirs(os.path.dirname(dst_image), exist_ok=True)
                    os.makedirs(os.path.dirname(dst_label), exist_ok=True)
                    
                    # Move files
                    shutil.move(image_path, dst_image)
                    shutil.move(label_path, dst_label)
                    moved_count += 1
                    
                except Exception as e:
                    logger.warning(f"Failed to move pair: {e}")
            
            result['action_taken'] = 'moved_pairs'
            result['moved_count'] = moved_count
            result['train_after'] = result['train_before'] - moved_count
            result['valid_after'] = result['valid_before'] + moved_count
            result['final_ratio'] = result['valid_after'] / (result['train_after'] + result['valid_after'])
            
            logger.info(f"Moved {moved_count} pairs to validation. New ratio: {result['final_ratio']:.1%}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error ensuring validation split: {e}")
        
        return result
    
    def get_all_valid_pairs(self, images_dir: str, labels_dir: str) -> List[Tuple[str, str]]:
        """
        Get all valid image-annotation pairs with content validation.
        """
        pairs = []
        
        if not os.path.exists(images_dir):
            return pairs
        
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for root, dirs, files in os.walk(images_dir):
            for image_file in files:
                file_ext = os.path.splitext(image_file)[1].lower()
                if file_ext in image_extensions:
                    image_path = os.path.join(root, image_file)
                    
                    # Find corresponding label file
                    rel_path = os.path.relpath(root, images_dir)
                    label_root = os.path.join(labels_dir, rel_path) if rel_path != '.' else labels_dir
                    
                    base_name = os.path.splitext(image_file)[0]
                    label_file = f"{base_name}.txt"
                    label_path = os.path.join(label_root, label_file)
                    
                    # Validate both files exist and have content
                    if os.path.exists(label_path):
                        try:
                            # Check label file has valid content
                            with open(label_path, 'r') as f:
                                content = f.read().strip()
                            
                            if content and not content.startswith('#'):
                                # Validate image can be loaded
                                img = cv2.imread(image_path)
                                if img is not None:
                                    pairs.append((image_path, label_path))
                                    
                        except Exception as e:
                            logger.debug(f"Invalid pair {image_path}: {e}")
        
        return pairs
    
    def process_piece_with_enhanced_validation(self, piece_label: str, dataset_custom_path: str, 
                                             group_label: str) -> Dict:
        """
        Enhanced piece processing with comprehensive validation and NaN prevention.
        """
        stats = {
            'piece_label': piece_label,
            'images_processed': 0,
            'images_cropped': 0,
            'images_skipped': 0,
            'images_too_small': 0,
            'images_no_annotations': 0,
            'annotations_input': 0,
            'annotations_output': 0,
            'annotations_filtered': 0,
            'empty_files_removed': 0,
            'validation_issues': [],
            'errors': []
        }
        
        # Define all necessary paths
        piece_dirs = {
            'train_images': os.path.join(dataset_custom_path, 'images', 'train', piece_label),
            'valid_images': os.path.join(dataset_custom_path, 'images', 'valid', piece_label),
            'train_labels': os.path.join(dataset_custom_path, 'labels', 'train', piece_label),
            'valid_labels': os.path.join(dataset_custom_path, 'labels', 'valid', piece_label),
        }
        
        cropped_dirs = {
            'train_images': os.path.join(dataset_custom_path, 'images_cropped', 'train', piece_label),
            'valid_images': os.path.join(dataset_custom_path, 'images_cropped', 'valid', piece_label),
            'train_labels': os.path.join(dataset_custom_path, 'labels_cropped', 'train', piece_label),
            'valid_labels': os.path.join(dataset_custom_path, 'labels_cropped', 'valid', piece_label),
        }
        
        # Create cropped directories
        for dir_path in cropped_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Process both splits
        for split in ['train', 'valid']:
            images_dir = piece_dirs[f'{split}_images']
            labels_dir = piece_dirs[f'{split}_labels']
            cropped_images_dir = cropped_dirs[f'{split}_images']
            cropped_labels_dir = cropped_dirs[f'{split}_labels']
            
            if not os.path.exists(images_dir):
                continue
                
            # Get all valid pairs for this split
            valid_pairs = self.get_all_valid_pairs(images_dir, labels_dir)
            
            for image_path, label_path in valid_pairs:
                try:
                    stats['images_processed'] += 1
                    image_file = os.path.basename(image_path)
                    
                    # Load and validate image
                    image = cv2.imread(image_path)
                    if image is None:
                        stats['errors'].append(f"Could not load image: {image_path}")
                        stats['images_skipped'] += 1
                        continue
                    
                    height, width = image.shape[:2]
                    
                    # Load and validate annotations
                    annotations = self.read_yolo_annotation(label_path, width, height)
                    stats['annotations_input'] += len(annotations)
                    
                    if not annotations:
                        stats['images_no_annotations'] += 1
                        stats['images_skipped'] += 1
                        logger.debug(f"No valid annotations for {image_file}")
                        continue
                    
                    # Handle small images
                    if width < self.crop_size or height < self.crop_size:
                        stats['images_too_small'] += 1
                        logger.debug(f"Image {image_file} ({width}x{height}) smaller than crop size")
                        
                        # Resize image to final size
                        resized_image = cv2.resize(image, (self.final_size, self.final_size), 
                                                 interpolation=cv2.INTER_LANCZOS4)
                        
                        # Save resized image
                        cropped_image_path = os.path.join(cropped_images_dir, image_file)
                        cv2.imwrite(cropped_image_path, resized_image)
                        
                        # For small images, annotations can be used as-is after validation
                        valid_annotations = []
                        for ann in annotations:
                            if self.is_valid_normalized_annotation(ann['x_center_norm'], ann['y_center_norm'],
                                                                 ann['width_norm'], ann['height_norm']):
                                valid_annotations.append({
                                    'class_id': ann['class_id'],
                                    'x_center_norm': ann['x_center_norm'],
                                    'y_center_norm': ann['y_center_norm'],
                                    'width_norm': ann['width_norm'],
                                    'height_norm': ann['height_norm']
                                })
                        
                        # Save annotations
                        annotation_file = os.path.splitext(image_file)[0] + '.txt'
                        cropped_annotation_path = os.path.join(cropped_labels_dir, annotation_file)
                        
                        if valid_annotations:
                            self.save_validated_annotation(cropped_annotation_path, valid_annotations)
                            stats['annotations_output'] += len(valid_annotations)
                            stats['images_cropped'] += 1
                        else:
                            # Remove image if no valid annotations
                            if os.path.exists(cropped_image_path):
                                os.remove(cropped_image_path)
                            stats['empty_files_removed'] += 1
                            stats['images_skipped'] += 1
                        
                        stats['annotations_filtered'] += len(annotations) - len(valid_annotations)
                        continue
                    
                    # Calculate optimal crop for larger images
                    center_x, center_y = self.calculate_smart_crop_center(annotations, width, height)
                    x1, y1, x2, y2 = self.calculate_precise_crop_bounds(center_x, center_y, width, height)
                    
                    crop_width = x2 - x1
                    crop_height = y2 - y1
                    
                    # Perform crop
                    cropped_image = image[y1:y2, x1:x2]
                    
                    if cropped_image.size == 0:
                        stats['errors'].append(f"Empty crop for {image_file}")
                        stats['images_skipped'] += 1
                        continue
                    
                    # Resize to final size
                    final_image = cv2.resize(cropped_image, (self.final_size, self.final_size),
                                           interpolation=cv2.INTER_LANCZOS4)
                    
                    # Update annotations for crop
                    updated_annotations = self.update_annotations_for_crop_enhanced(
                        annotations, x1, y1, crop_width, crop_height)
                    
                    # Save results only if we have valid annotations
                    annotation_file = os.path.splitext(image_file)[0] + '.txt'
                    cropped_image_path = os.path.join(cropped_images_dir, image_file)
                    cropped_annotation_path = os.path.join(cropped_labels_dir, annotation_file)
                    
                    if updated_annotations:
                        cv2.imwrite(cropped_image_path, final_image)
                        self.save_validated_annotation(cropped_annotation_path, updated_annotations)
                        stats['annotations_output'] += len(updated_annotations)
                        stats['images_cropped'] += 1
                    else:
                        stats['empty_files_removed'] += 1
                        stats['images_skipped'] += 1
                        logger.debug(f"No valid annotations after cropping for {image_file}")
                    
                    stats['annotations_filtered'] += len(annotations) - len(updated_annotations)
                    
                except Exception as e:
                    stats['errors'].append(f"Error processing {image_file}: {str(e)}")
                    stats['images_skipped'] += 1
                    continue
        
        return stats
    
    def create_final_dataset_structure(self, dataset_custom_path: str, group_label: str) -> str:
        """
        Create final dataset structure with comprehensive validation and path fixing.
        """
        cropped_dataset_path = os.path.join(os.path.dirname(dataset_custom_path), f"{group_label}_cropped")
        
        logger.info(f"Creating final dataset structure at: {cropped_dataset_path}")
        
        # Create directory structure
        final_dirs = {
            'train_images': os.path.join(cropped_dataset_path, 'images', 'train'),
            'valid_images': os.path.join(cropped_dataset_path, 'images', 'valid'),
            'train_labels': os.path.join(cropped_dataset_path, 'labels', 'train'),
            'valid_labels': os.path.join(cropped_dataset_path, 'labels', 'valid')
        }
        
        for dir_path in final_dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        # Copy cropped data from intermediate directories
        source_dirs = {
            'train_images': os.path.join(dataset_custom_path, 'images_cropped', 'train'),
            'valid_images': os.path.join(dataset_custom_path, 'images_cropped', 'valid'),
            'train_labels': os.path.join(dataset_custom_path, 'labels_cropped', 'train'),
            'valid_labels': os.path.join(dataset_custom_path, 'labels_cropped', 'valid')
        }
        
        copy_stats = {'files_copied': 0, 'errors': []}
        
        for key in source_dirs:
            source_dir = source_dirs[key]
            dest_dir = final_dirs[key]
            
            if os.path.exists(source_dir):
                try:
                    # Copy all files recursively
                    for root, dirs, files in os.walk(source_dir):
                        for file in files:
                            src_file = os.path.join(root, file)
                            
                            # Calculate relative path to maintain structure
                            rel_path = os.path.relpath(src_file, source_dir)
                            dst_file = os.path.join(dest_dir, rel_path)
                            
                            # Create destination directory if needed
                            os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                            
                            # Copy file
                            shutil.copy2(src_file, dst_file)
                            copy_stats['files_copied'] += 1
                            
                except Exception as e:
                    copy_stats['errors'].append(f"Error copying {key}: {str(e)}")
        
        logger.info(f"Copied {copy_stats['files_copied']} files to final structure")
        
        # CRITICAL: Fix data.yaml with correct paths
        data_yaml_path = os.path.join(cropped_dataset_path, 'data.yaml')
        paths_fixed = self.fix_dataset_paths_in_yaml(data_yaml_path, cropped_dataset_path)
        
        if not paths_fixed:
            logger.error("CRITICAL: Failed to fix dataset paths in YAML - this will cause training failure!")
        
        # Ensure minimum validation split
        split_result = self.ensure_minimum_validation_split(cropped_dataset_path)
        logger.info(f"Validation split result: {split_result}")
        
        # Final comprehensive validation
        validation_report = self.comprehensive_dataset_validation(cropped_dataset_path)
        
        # Log critical issues
        if validation_report['critical_issues']:
            for issue in validation_report['critical_issues']:
                logger.error(f"CRITICAL ISSUE: {issue}")
        
        if validation_report['warnings']:
            for warning in validation_report['warnings'][:5]:  # Log first 5 warnings
                logger.warning(f"WARNING: {warning}")
        
        logger.info(f"Final dataset validation summary:")
        logger.info(f"  - Train pairs: {validation_report['train_pairs']}")
        logger.info(f"  - Valid pairs: {validation_report['valid_pairs']}")
        logger.info(f"  - Validation ratio: {validation_report['validation_ratio']:.1%}")
        logger.info(f"  - Total valid annotations: {validation_report['total_annotations']}")
        logger.info(f"  - Invalid annotations: {validation_report['invalid_annotations']}")
        logger.info(f"  - Critical issues: {len(validation_report['critical_issues'])}")
        
        # Clean up intermediate directories
        try:
            intermediate_dirs = [
                os.path.join(dataset_custom_path, 'images_cropped'),
                os.path.join(dataset_custom_path, 'labels_cropped')
            ]
            
            for intermediate_dir in intermediate_dirs:
                if os.path.exists(intermediate_dir):
                    shutil.rmtree(intermediate_dir)
                    logger.info(f"Cleaned up intermediate directory: {intermediate_dir}")
                    
        except Exception as e:
            logger.warning(f"Failed to clean up intermediate directories: {e}")
        
        return cropped_dataset_path
    
    def comprehensive_dataset_validation(self, dataset_path: str) -> Dict:
        """
        Comprehensive dataset validation with NaN prevention focus.
        """
        report = {
            'dataset_path': dataset_path,
            'train_pairs': 0,
            'valid_pairs': 0,
            'validation_ratio': 0.0,
            'total_annotations': 0,
            'invalid_annotations': 0,
            'class_distribution': defaultdict(int),
            'critical_issues': [],
            'warnings': [],
            'paths_validated': {},
            'annotation_stats': {
                'min_bbox_size': float('inf'),
                'max_bbox_size': 0,
                'avg_bbox_area': 0,
                'edge_annotations': 0,
                'very_small_annotations': 0
            }
        }
        
        # Validate critical paths exist
        critical_paths = {
            'train_images': os.path.join(dataset_path, 'images', 'train'),
            'valid_images': os.path.join(dataset_path, 'images', 'valid'),
            'train_labels': os.path.join(dataset_path, 'labels', 'train'),
            'valid_labels': os.path.join(dataset_path, 'labels', 'valid'),
            'data_yaml': os.path.join(dataset_path, 'data.yaml')
        }
        
        for path_name, path in critical_paths.items():
            exists = os.path.exists(path)
            report['paths_validated'][path_name] = exists
            if not exists:
                report['critical_issues'].append(f"Missing {path_name}: {path}")
        
        # Validate data.yaml content
        try:
            with open(critical_paths['data_yaml'], 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            required_keys = ['train', 'val', 'names']
            for key in required_keys:
                if key not in yaml_data:
                    report['critical_issues'].append(f"Missing '{key}' in data.yaml")
            
            # Check if paths in yaml actually exist
            if 'train' in yaml_data and not os.path.exists(yaml_data['train']):
                report['critical_issues'].append(f"Train path in yaml does not exist: {yaml_data['train']}")
            
            if 'val' in yaml_data and not os.path.exists(yaml_data['val']):
                report['critical_issues'].append(f"Validation path in yaml does not exist: {yaml_data['val']}")
                
        except Exception as e:
            report['critical_issues'].append(f"Error reading data.yaml: {str(e)}")
        
        # Get valid pairs for each split
        train_pairs = self.get_all_valid_pairs(critical_paths['train_images'], critical_paths['train_labels'])
        valid_pairs = self.get_all_valid_pairs(critical_paths['valid_images'], critical_paths['valid_labels'])
        
        report['train_pairs'] = len(train_pairs)
        report['valid_pairs'] = len(valid_pairs)
        
        total_pairs = report['train_pairs'] + report['valid_pairs']
        if total_pairs > 0:
            report['validation_ratio'] = report['valid_pairs'] / total_pairs
        
        # Critical checks
        if report['valid_pairs'] == 0:
            report['critical_issues'].append("No validation samples - WILL CAUSE NaN")
        elif report['valid_pairs'] < self.min_validation_samples:
            report['critical_issues'].append(f"Insufficient validation samples ({report['valid_pairs']} < {self.min_validation_samples}) - MAY CAUSE NaN")
        
        if report['validation_ratio'] < 0.1:
            report['critical_issues'].append(f"Validation ratio too low ({report['validation_ratio']:.1%}) - MAY CAUSE NaN")
        
        # Analyze annotations
        all_pairs = train_pairs + valid_pairs
        bbox_areas = []
        
        for image_path, label_path in all_pairs:
            try:
                # Get image dimensions
                img = cv2.imread(image_path)
                if img is None:
                    report['warnings'].append(f"Could not load image: {image_path}")
                    continue
                
                height, width = img.shape[:2]
                
                # Read annotations
                annotations = self.read_yolo_annotation(label_path, width, height)
                report['total_annotations'] += len(annotations)
                
                for ann in annotations:
                    # Update class distribution
                    report['class_distribution'][ann['class_id']] += 1
                    
                    # Calculate statistics
                    bbox_area = ann['width_norm'] * ann['height_norm']
                    bbox_areas.append(bbox_area)
                    
                    # Update min/max bbox size
                    bbox_size = min(ann['width_norm'], ann['height_norm'])
                    report['annotation_stats']['min_bbox_size'] = min(report['annotation_stats']['min_bbox_size'], bbox_size)
                    report['annotation_stats']['max_bbox_size'] = max(report['annotation_stats']['max_bbox_size'], bbox_size)
                    
                    # Check for edge annotations
                    edge_threshold = 0.05
                    if (ann['x_center_norm'] < edge_threshold or ann['x_center_norm'] > (1 - edge_threshold) or
                        ann['y_center_norm'] < edge_threshold or ann['y_center_norm'] > (1 - edge_threshold)):
                        report['annotation_stats']['edge_annotations'] += 1
                    
                    # Check for very small annotations
                    if bbox_area < 0.001:  # Less than 0.1% of image area
                        report['annotation_stats']['very_small_annotations'] += 1
                    
                    # Validate annotation
                    if not self.is_valid_normalized_annotation(ann['x_center_norm'], ann['y_center_norm'],
                                                             ann['width_norm'], ann['height_norm']):
                        report['invalid_annotations'] += 1
                
            except Exception as e:
                report['warnings'].append(f"Error analyzing {label_path}: {str(e)}")
        
        # Calculate average bbox area
        if bbox_areas:
            report['annotation_stats']['avg_bbox_area'] = sum(bbox_areas) / len(bbox_areas)
            
            # Fix infinite min_bbox_size
            if report['annotation_stats']['min_bbox_size'] == float('inf'):
                report['annotation_stats']['min_bbox_size'] = 0
        
        # Additional warnings
        if report['annotation_stats']['very_small_annotations'] > 0:
            report['warnings'].append(f"{report['annotation_stats']['very_small_annotations']} very small annotations found - may cause training instability")
        
        if report['annotation_stats']['edge_annotations'] > report['total_annotations'] * 0.3:
            report['warnings'].append(f"High proportion of edge annotations ({report['annotation_stats']['edge_annotations']}/{report['total_annotations']}) - may cause training issues")
        
        if len(report['class_distribution']) < 2:
            report['warnings'].append(f"Only {len(report['class_distribution'])} class(es) found - consider multi-class dataset")
        
        return report
    
    def apply_enhanced_smart_cropping(self, group_label: str, piece_labels: List[str],
                                    dataset_base_path: str = None) -> Dict:
        """
        Main entry point for enhanced smart cropping with comprehensive NaN prevention.
        """
        if dataset_base_path is None:
            dataset_base_path = os.getenv('DATASET_BASE_PATH', '/app/shared/dataset')
        
        dataset_custom_path = os.path.join(dataset_base_path, 'dataset_custom', group_label)
        
        if not os.path.exists(dataset_custom_path):
            raise ValueError(f"Dataset path does not exist: {dataset_custom_path}")
        
        logger.info(f"Starting enhanced smart cropping for group {group_label}")
        logger.info(f"Pieces to process: {piece_labels}")
        logger.info(f"Crop size: {self.crop_size}, Final size: {self.final_size}")
        
        # Initialize comprehensive statistics
        total_stats = {
            'group_label': group_label,
            'piece_count': len(piece_labels),
            'crop_size': self.crop_size,
            'final_size': self.final_size,
            'processing_summary': {
                'total_images_processed': 0,
                'total_images_cropped': 0,
                'total_images_skipped': 0,
                'total_annotations_input': 0,
                'total_annotations_output': 0,
                'total_annotations_filtered': 0,
                'pieces_processed': 0,
                'pieces_failed': 0
            },
            'piece_stats': {},
            'final_dataset_path': None,
            'validation_report': None,
            'errors': [],
            'warnings': []
        }
        
        # Process each piece
        for piece_label in piece_labels:
            logger.info(f"Processing piece: {piece_label}")
            
            try:
                piece_stats = self.process_piece_with_enhanced_validation(
                    piece_label, dataset_custom_path, group_label)
                
                total_stats['piece_stats'][piece_label] = piece_stats
                total_stats['processing_summary']['pieces_processed'] += 1
                
                # Aggregate statistics
                for key in ['images_processed', 'images_cropped', 'images_skipped']:
                    total_stats['processing_summary'][f'total_{key}'] += piece_stats.get(key, 0)
                
                for key in ['annotations_input', 'annotations_output', 'annotations_filtered']:
                    total_stats['processing_summary'][f'total_{key}'] += piece_stats.get(key, 0)
                
                total_stats['errors'].extend(piece_stats.get('errors', []))
                total_stats['warnings'].extend(piece_stats.get('validation_issues', []))
                
                logger.info(f"Completed piece {piece_label}: {piece_stats['images_cropped']} images cropped, "
                           f"{piece_stats['annotations_output']} annotations saved")
                
            except Exception as e:
                error_msg = f"Failed to process piece {piece_label}: {str(e)}"
                logger.error(error_msg)
                total_stats['errors'].append(error_msg)
                total_stats['processing_summary']['pieces_failed'] += 1
        
        # Create final dataset structure
        try:
            cropped_dataset_path = self.create_final_dataset_structure(dataset_custom_path, group_label)
            total_stats['final_dataset_path'] = cropped_dataset_path
            
            # Comprehensive final validation
            validation_report = self.comprehensive_dataset_validation(cropped_dataset_path)
            total_stats['validation_report'] = validation_report
            
            logger.info(f"Enhanced smart cropping completed for group {group_label}")
            logger.info(f"Final dataset: {cropped_dataset_path}")
            logger.info(f"Processing summary: {total_stats['processing_summary']}")
            
        except Exception as e:
            error_msg = f"Failed to create final dataset structure: {str(e)}"
            logger.error(error_msg)
            total_stats['errors'].append(error_msg)
        
        return total_stats