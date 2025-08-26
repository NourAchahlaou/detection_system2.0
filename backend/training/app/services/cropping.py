import os
import cv2
import numpy as np
import yaml
import shutil
from typing import List, Tuple, Dict
import logging
from collections import defaultdict
import random

logger = logging.getLogger(__name__)

class SmartCroppingService:
    def __init__(self, crop_size: int = 980, final_size: int = 512):
        """
        Initialize the smart cropping service with improved validation.
        
        Args:
            crop_size: The consistent crop size to extract from original images (e.g., 980x980)
            final_size: The final size for training (e.g., 512x512)
        """
        self.crop_size = crop_size
        self.final_size = final_size
        
        # CRITICAL FIX: More restrictive thresholds to prevent edge cases
        self.min_bbox_size = 0.02  # Minimum 2% of image dimension (was 0.01)
        self.min_normalized_coord = 0.005  # Keep annotations away from edges (was 0.0001)
        self.max_normalized_coord = 0.995  # Keep annotations away from edges (was 0.9999)
        self.min_area_threshold = 0.0004  # Minimum area 0.04% (was 0.0001)
        self.min_overlap_ratio = 0.5  # CRITICAL: Increase to 50% overlap (was 0.3)
        
    def is_valid_normalized_annotation(self, x_center_norm: float, y_center_norm: float, 
                                     width_norm: float, height_norm: float) -> bool:
        """
        ENHANCED validation for normalized coordinates with stricter checks.
        """
        # Check basic bounds with stricter margins
        if not (self.min_normalized_coord <= x_center_norm <= self.max_normalized_coord and
                self.min_normalized_coord <= y_center_norm <= self.max_normalized_coord and
                self.min_bbox_size <= width_norm <= 0.98 and  # Max 98% width
                self.min_bbox_size <= height_norm <= 0.98):   # Max 98% height
            return False
        
        # Check that bounding box doesn't exceed image boundaries with stricter margins
        half_width = width_norm / 2
        half_height = height_norm / 2
        
        x_min = x_center_norm - half_width
        x_max = x_center_norm + half_width
        y_min = y_center_norm - half_height
        y_max = y_center_norm + half_height
        
        # STRICTER boundary check - no overflow allowed
        if not (0.01 <= x_min and x_max <= 0.99 and 0.01 <= y_min and y_max <= 0.99):
            return False
        
        # Check minimum area with higher threshold
        area = width_norm * height_norm
        if area < self.min_area_threshold:
            return False
        
        # ADDITIONAL CHECK: Reject very thin rectangles that can cause training issues
        aspect_ratio = max(width_norm, height_norm) / min(width_norm, height_norm)
        if aspect_ratio > 10.0:  # Reject aspect ratios > 10:1
            logger.debug(f"Rejecting annotation with extreme aspect ratio: {aspect_ratio:.2f}")
            return False
            
        return True
        
    def read_yolo_annotation(self, annotation_path: str, image_width: int, image_height: int) -> List[Dict]:
        """
        Read YOLO annotation file with enhanced validation.
        """
        annotations = []
        
        if not os.path.exists(annotation_path):
            return annotations
            
        with open(annotation_path, 'r') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split()
                if len(parts) < 5:
                    logger.debug(f"Skipping malformed line {line_num} in {annotation_path}")
                    continue
                
                try:
                    class_id = int(parts[0])
                    x_center_norm = float(parts[1])
                    y_center_norm = float(parts[2])
                    width_norm = float(parts[3])
                    height_norm = float(parts[4])
                    
                    # Skip obviously invalid annotations BEFORE processing
                    if not self.is_valid_normalized_annotation(x_center_norm, y_center_norm, width_norm, height_norm):
                        logger.debug(f"Skipping invalid annotation at line {line_num}: center=({x_center_norm:.3f}, {y_center_norm:.3f}), size=({width_norm:.3f}, {height_norm:.3f})")
                        continue
                    
                    # Convert to pixel coordinates with bounds checking
                    x_center = max(0, min(image_width, x_center_norm * image_width))
                    y_center = max(0, min(image_height, y_center_norm * image_height))
                    bbox_width = max(1, min(image_width, width_norm * image_width))
                    bbox_height = max(1, min(image_height, height_norm * image_height))
                    
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
                    
                except (ValueError, IndexError) as e:
                    logger.debug(f"Could not parse annotation line {line_num} in {annotation_path}: {e}")
                    continue
                
        return annotations
    
    def calculate_optimal_crop_center(self, annotations: List[Dict], image_width: int, image_height: int) -> Tuple[int, int]:
        """
        Calculate the optimal center point for cropping with safety margins.
        """
        if not annotations:
            return image_width // 2, image_height // 2
        
        # Calculate weighted centroid based on bbox areas
        total_weight = 0
        weighted_x = 0
        weighted_y = 0
        
        for ann in annotations:
            area = ann['bbox_width'] * ann['bbox_height']
            weight = max(area, 1)  # Avoid zero weight
            
            weighted_x += ann['x_center'] * weight
            weighted_y += ann['y_center'] * weight
            total_weight += weight
        
        if total_weight > 0:
            centroid_x = weighted_x / total_weight
            centroid_y = weighted_y / total_weight
        else:
            centroid_x = image_width // 2
            centroid_y = image_height // 2
        
        # Ensure centroid allows for full crop within image bounds
        half_crop = self.crop_size // 2
        centroid_x = max(half_crop, min(image_width - half_crop, centroid_x))
        centroid_y = max(half_crop, min(image_height - half_crop, centroid_y))
        
        return int(centroid_x), int(centroid_y)
    
    def calculate_safe_crop_bounds(self, center_x: int, center_y: int, image_width: int, image_height: int) -> Tuple[int, int, int, int]:
        """
        Calculate crop bounds with enhanced safety checks.
        """
        half_crop = self.crop_size // 2
        
        # Calculate ideal crop bounds
        x1 = center_x - half_crop
        y1 = center_y - half_crop
        x2 = center_x + half_crop
        y2 = center_y + half_crop
        
        # Adjust for image boundaries with minimum crop size preservation
        if x1 < 0:
            offset = -x1
            x1 = 0
            x2 = min(x2 + offset, image_width)
        elif x2 > image_width:
            offset = x2 - image_width
            x2 = image_width
            x1 = max(x1 - offset, 0)
            
        if y1 < 0:
            offset = -y1
            y1 = 0
            y2 = min(y2 + offset, image_height)
        elif y2 > image_height:
            offset = y2 - image_height
            y2 = image_height
            y1 = max(y1 - offset, 0)
        
        # Ensure minimum viable crop size
        crop_width = x2 - x1
        crop_height = y2 - y1
        min_crop_size = min(self.crop_size // 2, min(image_width, image_height) // 2)
        
        if crop_width < min_crop_size or crop_height < min_crop_size:
            logger.warning(f"Crop too small ({crop_width}x{crop_height}), using center crop")
            # Fall back to maximum possible center crop
            max_crop_size = min(image_width, image_height, self.crop_size)
            half_max = max_crop_size // 2
            center_x = image_width // 2
            center_y = image_height // 2
            
            x1 = max(0, center_x - half_max)
            y1 = max(0, center_y - half_max)
            x2 = min(image_width, x1 + max_crop_size)
            y2 = min(image_height, y1 + max_crop_size)
        
        return x1, y1, x2, y2
    
    def crop_image(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """
        Crop image with validation and error handling.
        """
        # Validate crop bounds
        height, width = image.shape[:2]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        
        # Perform crop
        cropped = image[y1:y2, x1:x2]
        
        # Validate crop result
        if cropped.size == 0:
            logger.error(f"Empty crop with bounds ({x1}, {y1}, {x2}, {y2}) on image {width}x{height}")
            return np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
        
        # Resize to target crop size
        if cropped.shape[0] != self.crop_size or cropped.shape[1] != self.crop_size:
            cropped = cv2.resize(cropped, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LANCZOS4)
        
        return cropped
    
    def update_annotations_for_crop(self, annotations: List[Dict], x1: int, y1: int, 
                                  original_crop_width: int, original_crop_height: int) -> List[Dict]:
        """
        CRITICAL FIX: Update annotations with proper coordinate transformation and stricter filtering.
        """
        updated_annotations = []
        
        for ann in annotations:
            # Convert original pixel coordinates to crop space
            crop_x_center = ann['x_center'] - x1
            crop_y_center = ann['y_center'] - y1
            
            # Keep original bbox dimensions (they don't change during translation)
            crop_bbox_width = ann['bbox_width']
            crop_bbox_height = ann['bbox_height']
            
            # Calculate bounding box edges in crop space
            bbox_x1 = crop_x_center - crop_bbox_width / 2
            bbox_y1 = crop_y_center - crop_bbox_height / 2
            bbox_x2 = crop_x_center + crop_bbox_width / 2
            bbox_y2 = crop_y_center + crop_bbox_height / 2
            
            # Calculate intersection with crop boundaries
            intersect_x1 = max(0, bbox_x1)
            intersect_y1 = max(0, bbox_y1)
            intersect_x2 = min(original_crop_width, bbox_x2)
            intersect_y2 = min(original_crop_height, bbox_y2)
            
            # Check if intersection is valid
            intersect_width = max(0, intersect_x2 - intersect_x1)
            intersect_height = max(0, intersect_y2 - intersect_y1)
            intersect_area = intersect_width * intersect_height
            
            # Calculate original bbox area in crop space
            original_area = crop_bbox_width * crop_bbox_height
            
            # Only keep annotation if sufficient overlap
            overlap_ratio = intersect_area / original_area if original_area > 0 else 0
            
            if overlap_ratio < self.min_overlap_ratio:
                logger.debug(f"Skipping annotation with low overlap ratio: {overlap_ratio:.3f}")
                continue
            
            # CRITICAL CHECK: Ensure intersection has minimum viable size
            if intersect_width < 5 or intersect_height < 5:  # Minimum 5 pixels
                logger.debug(f"Skipping annotation with too small intersection: {intersect_width}x{intersect_height}")
                continue
            
            # Use intersection as the new bounding box
            new_center_x = (intersect_x1 + intersect_x2) / 2
            new_center_y = (intersect_y1 + intersect_y2) / 2
            new_width = intersect_width
            new_height = intersect_height
            
            # Convert to normalized coordinates using ORIGINAL crop dimensions
            new_x_center_norm = new_center_x / original_crop_width
            new_y_center_norm = new_center_y / original_crop_height
            new_width_norm = new_width / original_crop_width
            new_height_norm = new_height / original_crop_height
            
            # STRICT validation and clamping
            new_x_center_norm = max(self.min_normalized_coord, min(self.max_normalized_coord, new_x_center_norm))
            new_y_center_norm = max(self.min_normalized_coord, min(self.max_normalized_coord, new_y_center_norm))
            new_width_norm = max(self.min_bbox_size, min(0.98, new_width_norm))  # Max 98%
            new_height_norm = max(self.min_bbox_size, min(0.98, new_height_norm))  # Max 98%
            
            # Final validation with enhanced checks
            if self.is_valid_normalized_annotation(new_x_center_norm, new_y_center_norm, new_width_norm, new_height_norm):
                updated_annotations.append({
                    'class_id': ann['class_id'],
                    'x_center_norm': new_x_center_norm,
                    'y_center_norm': new_y_center_norm,
                    'width_norm': new_width_norm,
                    'height_norm': new_height_norm
                })
                
                logger.debug(f"Updated annotation: center=({new_x_center_norm:.3f}, {new_y_center_norm:.3f}), "
                           f"size=({new_width_norm:.3f}, {new_height_norm:.3f}), overlap={overlap_ratio:.3f}")
            else:
                logger.debug(f"Final validation failed for annotation: center=({new_x_center_norm:.3f}, {new_y_center_norm:.3f}), "
                           f"size=({new_width_norm:.3f}, {new_height_norm:.3f})")
        
        return updated_annotations
    
    def save_yolo_annotation(self, annotation_path: str, annotations: List[Dict]):
        """
        ENHANCED: Save annotations with comprehensive validation and format precision.
        """
        os.makedirs(os.path.dirname(annotation_path), exist_ok=True)
        
        # Filter out any remaining invalid annotations with extra checks
        valid_annotations = []
        for ann in annotations:
            if (isinstance(ann['class_id'], int) and ann['class_id'] >= 0 and
                self.is_valid_normalized_annotation(ann['x_center_norm'], ann['y_center_norm'], 
                                                  ann['width_norm'], ann['height_norm'])):
                
                # ADDITIONAL CHECK: Ensure all values are finite
                values = [ann['x_center_norm'], ann['y_center_norm'], ann['width_norm'], ann['height_norm']]
                if all(isinstance(v, (int, float)) and not np.isnan(v) and not np.isinf(v) for v in values):
                    valid_annotations.append(ann)
                else:
                    logger.debug(f"Final filter caught non-finite values: {ann}")
            else:
                logger.debug(f"Final filter caught invalid annotation: {ann}")
        
        # CRITICAL: Remove empty annotation files instead of creating them
        if not valid_annotations:
            if os.path.exists(annotation_path):
                os.remove(annotation_path)
                logger.info(f"Removed empty annotation file: {annotation_path}")
            else:
                logger.info(f"No annotations to save for {annotation_path} - no file created")
            return
        
        # Write valid annotations with consistent precision
        with open(annotation_path, 'w') as f:
            for ann in valid_annotations:
                f.write(f"{ann['class_id']} {ann['x_center_norm']:.6f} {ann['y_center_norm']:.6f} "
                       f"{ann['width_norm']:.6f} {ann['height_norm']:.6f}\n")
        
        logger.info(f"Saved {len(valid_annotations)} valid annotations to {annotation_path}")
    
    def resize_to_final_size(self, image: np.ndarray) -> np.ndarray:
        """Resize the cropped image to final training size."""
        if image.shape[:2] != (self.final_size, self.final_size):
            return cv2.resize(image, (self.final_size, self.final_size), interpolation=cv2.INTER_LANCZOS4)
        return image
    
    def get_all_image_annotation_pairs(self, images_dir: str, labels_dir: str) -> List[Tuple[str, str]]:
        """
        FIXED: Get all valid image-annotation pairs from a directory.
        Returns pairs of (image_file, label_file) that both exist and have content.
        """
        pairs = []
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            return pairs
        
        # Walk through all subdirectories to handle piece-based structure
        for root, dirs, files in os.walk(images_dir):
            for image_file in files:
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    # Get relative path from images_dir to maintain directory structure
                    rel_path = os.path.relpath(root, images_dir)
                    
                    # Construct corresponding label path
                    if rel_path == '.':
                        label_root = labels_dir
                    else:
                        label_root = os.path.join(labels_dir, rel_path)
                    
                    # Get corresponding annotation file
                    base_name = os.path.splitext(image_file)[0]
                    label_file = f"{base_name}.txt"
                    label_path = os.path.join(label_root, label_file)
                    
                    # Check if annotation file exists and has content
                    if os.path.exists(label_path):
                        try:
                            with open(label_path, 'r') as f:
                                content = f.read().strip()
                            if content:  # Has actual content
                                pairs.append((os.path.join(root, image_file), label_path))
                        except Exception as e:
                            logger.warning(f"Error reading {label_path}: {e}")
        
        return pairs
    
    def ensure_validation_split(self, dataset_path: str, target_ratio: float = 0.2) -> Dict:
        """
        FIXED: Ensure proper train/validation split to prevent NaN.
        Move images from train to valid if needed.
        """
        result = {
            'moved': 0,
            'reason': '',
            'train_before': 0,
            'valid_before': 0,
            'train_after': 0,
            'valid_after': 0
        }
        
        train_images_dir = os.path.join(dataset_path, 'images', 'train')
        valid_images_dir = os.path.join(dataset_path, 'images', 'valid')
        train_labels_dir = os.path.join(dataset_path, 'labels', 'train')
        valid_labels_dir = os.path.join(dataset_path, 'labels', 'valid')
        
        # Create validation directories if they don't exist
        os.makedirs(valid_images_dir, exist_ok=True)
        os.makedirs(valid_labels_dir, exist_ok=True)
        
        # Get all valid training and validation pairs
        train_pairs = self.get_all_image_annotation_pairs(train_images_dir, train_labels_dir)
        valid_pairs = self.get_all_image_annotation_pairs(valid_images_dir, valid_labels_dir)
        
        result['train_before'] = len(train_pairs)
        result['valid_before'] = len(valid_pairs)
        
        total_pairs = len(train_pairs) + len(valid_pairs)
        
        if total_pairs == 0:
            result['reason'] = 'No valid image-annotation pairs found'
            return result
        
        current_valid_ratio = len(valid_pairs) / total_pairs
        
        # If we already have sufficient validation data, don't move anything
        if current_valid_ratio >= 0.15:  # 15% minimum
            result['reason'] = f'Sufficient validation data ({current_valid_ratio:.1%})'
            result['train_after'] = result['train_before']
            result['valid_after'] = result['valid_before']
            return result
        
        # Calculate how many pairs to move to achieve target ratio
        target_valid_count = max(1, int(total_pairs * target_ratio))
        needed_valid = target_valid_count - len(valid_pairs)
        to_move = min(needed_valid, len(train_pairs) // 2)  # Don't move more than half
        
        if to_move <= 0:
            result['reason'] = 'No pairs need to be moved'
            result['train_after'] = result['train_before']
            result['valid_after'] = result['valid_before']
            return result
        
        # Randomly select pairs to move (deterministic for reproducibility)
        random.seed(42)
        pairs_to_move = random.sample(train_pairs, to_move)
        
        moved_count = 0
        for image_path, label_path in pairs_to_move:
            try:
                # Get relative path from train images directory
                image_rel_path = os.path.relpath(image_path, train_images_dir)
                label_rel_path = os.path.relpath(label_path, train_labels_dir)
                
                # Construct destination paths
                dst_image_path = os.path.join(valid_images_dir, image_rel_path)
                dst_label_path = os.path.join(valid_labels_dir, label_rel_path)
                
                # Create destination directories if needed
                os.makedirs(os.path.dirname(dst_image_path), exist_ok=True)
                os.makedirs(os.path.dirname(dst_label_path), exist_ok=True)
                
                # Move files
                shutil.move(image_path, dst_image_path)
                shutil.move(label_path, dst_label_path)
                
                moved_count += 1
                logger.info(f"Moved {os.path.basename(image_path)} from train to validation")
                
            except Exception as e:
                logger.error(f"Failed to move {image_path}: {e}")
        
        # Update counts
        result['moved'] = moved_count
        result['train_after'] = result['train_before'] - moved_count
        result['valid_after'] = result['valid_before'] + moved_count
        result['reason'] = f'Moved {moved_count} pairs to achieve {target_ratio:.0%} validation split'
        
        return result
    
    def cleanup_empty_annotation_files(self, labels_dir: str) -> Dict:
        """
        Clean up any remaining empty annotation files in the directory.
        """
        cleanup_stats = {
            'files_checked': 0,
            'empty_files_removed': 0,
            'files_with_content': 0
        }
        
        if not os.path.exists(labels_dir):
            return cleanup_stats
            
        for root, dirs, files in os.walk(labels_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_path = os.path.join(root, file)
                    cleanup_stats['files_checked'] += 1
                    
                    try:
                        # Check if file is empty or contains only whitespace
                        with open(file_path, 'r') as f:
                            content = f.read().strip()
                        
                        if not content:
                            os.remove(file_path)
                            cleanup_stats['empty_files_removed'] += 1
                            logger.info(f"Removed empty annotation file: {file_path}")
                        else:
                            cleanup_stats['files_with_content'] += 1
                            
                    except Exception as e:
                        logger.warning(f"Error checking file {file_path}: {e}")
        
        return cleanup_stats
    
    def remove_orphaned_images(self, images_dir: str, labels_dir: str) -> Dict:
        """
        Remove images that don't have corresponding annotation files.
        This prevents NaN values during training caused by images without labels.
        """
        removal_stats = {
            'images_checked': 0,
            'orphaned_images_removed': 0,
            'images_with_annotations': 0,
            'removed_image_paths': []
        }
        
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            return removal_stats
        
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        
        for root, dirs, files in os.walk(images_dir):
            for image_file in files:
                file_ext = os.path.splitext(image_file)[1].lower()
                if file_ext in image_extensions:
                    removal_stats['images_checked'] += 1
                    
                    # Construct corresponding annotation path
                    image_path = os.path.join(root, image_file)
                    
                    # Get relative path from images_dir to maintain directory structure
                    rel_path = os.path.relpath(root, images_dir)
                    
                    # Create corresponding label directory path
                    if rel_path == '.':
                        label_root = labels_dir
                    else:
                        label_root = os.path.join(labels_dir, rel_path)
                    
                    # Convert image filename to annotation filename
                    base_name = os.path.splitext(image_file)[0]
                    annotation_file = f"{base_name}.txt"
                    annotation_path = os.path.join(label_root, annotation_file)
                    
                    # Check if annotation file exists and has content
                    has_valid_annotation = False
                    if os.path.exists(annotation_path):
                        try:
                            with open(annotation_path, 'r') as f:
                                content = f.read().strip()
                            if content:  # File exists and has content
                                has_valid_annotation = True
                        except Exception as e:
                            logger.warning(f"Error reading annotation file {annotation_path}: {e}")
                    
                    if has_valid_annotation:
                        removal_stats['images_with_annotations'] += 1
                    else:
                        # Remove orphaned image
                        try:
                            os.remove(image_path)
                            removal_stats['orphaned_images_removed'] += 1
                            removal_stats['removed_image_paths'].append(image_path)
                            logger.info(f"Removed orphaned image (no annotation): {image_path}")
                        except Exception as e:
                            logger.error(f"Error removing orphaned image {image_path}: {e}")
        
        return removal_stats
    
    def validate_dataset_integrity(self, dataset_path: str) -> Dict:
        """
        ENHANCED dataset validation with comprehensive NaN prevention checks.
        """
        stats = {
            'total_images': 0,
            'total_annotations': 0,
            'empty_annotation_files': 0,
            'invalid_annotations': 0,
            'valid_annotations': 0,
            'edge_case_annotations': 0,
            'images_without_annotations': 0,
            'orphaned_images_removed': 0,
            'cleanup_stats': {},
            'orphaned_removal_stats': {},
            'validation_split_stats': {},
            'issues': []
        }
        
        # STEP 1: Clean up empty annotation files
        for split in ['train', 'valid']:
            labels_dir = os.path.join(dataset_path, 'labels', split)
            if os.path.exists(labels_dir):
                cleanup_stats = self.cleanup_empty_annotation_files(labels_dir)
                stats['cleanup_stats'][split] = cleanup_stats
        
        # STEP 2: Remove orphaned images (CRITICAL FOR PREVENTING NaN)
        for split in ['train', 'valid']:
            images_dir = os.path.join(dataset_path, 'images', split)
            labels_dir = os.path.join(dataset_path, 'labels', split)
            
            if os.path.exists(images_dir):
                orphaned_stats = self.remove_orphaned_images(images_dir, labels_dir)
                stats['orphaned_removal_stats'][split] = orphaned_stats
                stats['orphaned_images_removed'] += orphaned_stats['orphaned_images_removed']
                logger.info(f"Removed {orphaned_stats['orphaned_images_removed']} orphaned images from {split} split")
        
        # STEP 3: Ensure proper validation split (CRITICAL FOR PREVENTING NaN)
        balance_result = self.ensure_validation_split(dataset_path)
        stats['validation_split_stats'] = balance_result
        logger.info(f"Validation split adjustment: {balance_result}")
        
        # STEP 4: Final validation and statistics
        for split in ['train', 'valid']:
            images_dir = os.path.join(dataset_path, 'images', split)
            labels_dir = os.path.join(dataset_path, 'labels', split)
            
            if not os.path.exists(images_dir):
                continue
                
            for root, dirs, files in os.walk(images_dir):
                for image_file in files:
                    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        stats['total_images'] += 1
                        
                        # Check corresponding annotation
                        rel_path = os.path.relpath(root, images_dir)
                        label_root = os.path.join(labels_dir, rel_path) if rel_path != '.' else labels_dir
                        annotation_file = image_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
                        annotation_path = os.path.join(label_root, annotation_file)
                        
                        if os.path.exists(annotation_path):
                            with open(annotation_path, 'r') as f:
                                lines = f.readlines()
                                
                            if not lines or not any(line.strip() for line in lines):
                                # This shouldn't happen after cleanup, but just in case
                                os.remove(annotation_path)
                                stats['empty_annotation_files'] += 1
                                stats['images_without_annotations'] += 1
                                logger.warning(f"Removed empty annotation file during validation: {annotation_path}")
                            else:
                                for line in lines:
                                    line = line.strip()
                                    if line:
                                        stats['total_annotations'] += 1
                                        parts = line.split()
                                        if len(parts) == 5:
                                            try:
                                                class_id = int(parts[0])
                                                coords = [float(x) for x in parts[1:5]]
                                                
                                                # Enhanced validation
                                                if self.is_valid_normalized_annotation(coords[0], coords[1], coords[2], coords[3]):
                                                    stats['valid_annotations'] += 1
                                                    
                                                    # Check for edge cases that might cause training instability
                                                    if (coords[0] < 0.02 or coords[0] > 0.98 or 
                                                        coords[1] < 0.02 or coords[1] > 0.98 or
                                                        coords[2] < 0.05 or coords[3] < 0.05):
                                                        stats['edge_case_annotations'] += 1
                                                else:
                                                    stats['invalid_annotations'] += 1
                                                    stats['issues'].append(f"Invalid coordinates in {annotation_path}: {line}")
                                            except ValueError:
                                                stats['invalid_annotations'] += 1
                                                stats['issues'].append(f"Parse error in {annotation_path}: {line}")
                                        else:
                                            stats['invalid_annotations'] += 1
                                            stats['issues'].append(f"Wrong format in {annotation_path}: {line}")
                        else:
                            # This should NOT happen after orphaned image removal
                            stats['images_without_annotations'] += 1
                            logger.error(f"CRITICAL: Found image without annotation after cleanup: {os.path.join(root, image_file)}")
        
        # STEP 5: Final validation split check
        train_pairs = self.get_all_image_annotation_pairs(
            os.path.join(dataset_path, 'images', 'train'),
            os.path.join(dataset_path, 'labels', 'train')
        )
        valid_pairs = self.get_all_image_annotation_pairs(
            os.path.join(dataset_path, 'images', 'valid'),
            os.path.join(dataset_path, 'labels', 'valid')
        )
        
        total_pairs = len(train_pairs) + len(valid_pairs)
        valid_ratio = len(valid_pairs) / total_pairs if total_pairs > 0 else 0
        
        # Add critical warnings
        if len(valid_pairs) == 0:
            stats['issues'].append("CRITICAL: No validation images - WILL CAUSE NaN")
        
        if valid_ratio < 0.1:
            stats['issues'].append(f"CRITICAL: Insufficient validation split ({valid_ratio:.1%}) - WILL CAUSE NaN")
        
        stats['final_split_ratio'] = valid_ratio
        stats['train_pairs'] = len(train_pairs)
        stats['valid_pairs'] = len(valid_pairs)
        
        return stats
    
    def process_piece_images(self, piece_label: str, dataset_custom_path: str, group_label: str) -> Dict:
        """
        Process images with enhanced error handling, validation, and NaN prevention.
        """
        stats = {
            'images_processed': 0,
            'images_cropped': 0,
            'images_skipped': 0,
            'annotations_updated': 0,
            'annotations_filtered': 0,
            'empty_annotation_files_removed': 0,
            'orphaned_images_removed': 0,
            'errors': []
        }
        
        # Setup paths
        piece_images_valid = os.path.join(dataset_custom_path, 'images', 'valid', piece_label)
        piece_labels_valid = os.path.join(dataset_custom_path, 'labels', 'valid', piece_label)
        piece_images_train = os.path.join(dataset_custom_path, 'images', 'train', piece_label)
        piece_labels_train = os.path.join(dataset_custom_path, 'labels', 'train', piece_label)
        
        # Create cropped directories
        cropped_images_valid = os.path.join(dataset_custom_path, 'images_cropped', 'valid', piece_label)
        cropped_labels_valid = os.path.join(dataset_custom_path, 'labels_cropped', 'valid', piece_label)
        cropped_images_train = os.path.join(dataset_custom_path, 'images_cropped', 'train', piece_label)
        cropped_labels_train = os.path.join(dataset_custom_path, 'labels_cropped', 'train', piece_label)
        
        for cropped_dir in [cropped_images_valid, cropped_labels_valid, cropped_images_train, cropped_labels_train]:
            os.makedirs(cropped_dir, exist_ok=True)
        
        # Process both splits
        for split in ['valid', 'train']:
            if split == 'valid':
                images_dir = piece_images_valid
                labels_dir = piece_labels_valid
                cropped_images_dir = cropped_images_valid
                cropped_labels_dir = cropped_labels_valid
            else:
                images_dir = piece_images_train
                labels_dir = piece_labels_train
                cropped_images_dir = cropped_images_train
                cropped_labels_dir = cropped_labels_train
            
            if not os.path.exists(images_dir):
                continue
                
            image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            for image_file in image_files:
                try:
                    stats['images_processed'] += 1
                    
                    # Load and validate image
                    image_path = os.path.join(images_dir, image_file)
                    image = cv2.imread(image_path)
                    
                    if image is None:
                        stats['errors'].append(f"Could not load image: {image_path}")
                        continue
                    
                    height, width = image.shape[:2]
                    
                    # Handle small images
                    if width < self.crop_size or height < self.crop_size:
                        logger.info(f"Image {image_file} ({width}x{height}) smaller than crop size, resizing directly")
                        # Direct resize without cropping
                        resized_image = self.resize_to_final_size(image)
                        cropped_image_path = os.path.join(cropped_images_dir, image_file)
                        cv2.imwrite(cropped_image_path, resized_image)
                        
                        # Process annotation for direct resize
                        annotation_file = image_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
                        src_annotation = os.path.join(labels_dir, annotation_file)
                        dst_annotation = os.path.join(cropped_labels_dir, annotation_file)

                        if os.path.exists(src_annotation):
                            # Validate and filter original annotations
                            annotations = self.read_yolo_annotation(src_annotation, width, height)
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
                            
                            # This will automatically handle empty annotations (remove file)
                            self.save_yolo_annotation(dst_annotation, valid_annotations)
                            if valid_annotations:
                                stats['annotations_updated'] += len(valid_annotations)
                            else:
                                stats['empty_annotation_files_removed'] += 1
                        # If no source annotation exists, we don't create an empty one
                        
                        stats['images_cropped'] += 1
                        continue
                    
                    # Load and validate annotations
                    annotation_file = image_file.replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
                    annotation_path = os.path.join(labels_dir, annotation_file)
                    
                    annotations = self.read_yolo_annotation(annotation_path, width, height)
                    original_count = len(annotations)
                    
                    # Skip images with no valid annotations to prevent NaN
                    if not annotations:
                        logger.info(f"Skipping image {image_file} - no valid annotations found")
                        stats['images_skipped'] += 1
                        continue
                    
                    # Calculate crop parameters
                    center_x, center_y = self.calculate_optimal_crop_center(annotations, width, height)
                    
                    # Get safe crop bounds
                    x1, y1, x2, y2 = self.calculate_safe_crop_bounds(center_x, center_y, width, height)
                    original_crop_width = x2 - x1
                    original_crop_height = y2 - y1
                    
                    # Crop image
                    cropped_image = self.crop_image(image, x1, y1, x2, y2)
                    final_image = self.resize_to_final_size(cropped_image)
                    
                    # Save cropped image
                    cropped_image_path = os.path.join(cropped_images_dir, image_file)
                    cv2.imwrite(cropped_image_path, final_image)
                    
                    # Update annotations for crop
                    updated_annotations = self.update_annotations_for_crop(
                        annotations, x1, y1, original_crop_width, original_crop_height)
                    
                    # CRITICAL: Only save if we have valid annotations
                    cropped_annotation_path = os.path.join(cropped_labels_dir, annotation_file)
                    if updated_annotations:
                        self.save_yolo_annotation(cropped_annotation_path, updated_annotations)
                        stats['annotations_updated'] += len(updated_annotations)
                        stats['images_cropped'] += 1
                    else:
                        # Remove the cropped image too if no valid annotations
                        if os.path.exists(cropped_image_path):
                            os.remove(cropped_image_path)
                        stats['empty_annotation_files_removed'] += 1
                        stats['images_skipped'] += 1
                        logger.info(f"Removed cropped image {image_file} - no valid annotations after cropping")
                        
                    stats['annotations_filtered'] += original_count - len(updated_annotations)
                    
                except Exception as e:
                    stats['errors'].append(f"Error processing {image_file}: {str(e)}")
                    stats['images_skipped'] += 1
                    continue
        
        return stats
    
    def create_cropped_dataset_structure(self, dataset_custom_path: str, group_label: str) -> str:
        """
        Create final dataset structure with comprehensive validation and NaN prevention.
        """
        cropped_dataset_path = os.path.join(os.path.dirname(dataset_custom_path), f"{group_label}_cropped")
        
        # Create directory structure
        os.makedirs(cropped_dataset_path, exist_ok=True)
        
        # Copy cropped data to new structure
        source_dirs = {
            'train_images': os.path.join(dataset_custom_path, 'images_cropped', 'train'),
            'valid_images': os.path.join(dataset_custom_path, 'images_cropped', 'valid'),
            'train_labels': os.path.join(dataset_custom_path, 'labels_cropped', 'train'),
            'valid_labels': os.path.join(dataset_custom_path, 'labels_cropped', 'valid')
        }
        
        dest_dirs = {
            'train_images': os.path.join(cropped_dataset_path, 'images', 'train'),
            'valid_images': os.path.join(cropped_dataset_path, 'images', 'valid'),
            'train_labels': os.path.join(cropped_dataset_path, 'labels', 'train'),
            'valid_labels': os.path.join(cropped_dataset_path, 'labels', 'valid')
        }
        
        # Copy directories if they exist
        for key in source_dirs:
            if os.path.exists(source_dirs[key]):
                shutil.copytree(source_dirs[key], dest_dirs[key], dirs_exist_ok=True)
        
        # Create updated data.yaml with DYNAMIC paths (CRITICAL FIX)
        original_yaml = os.path.join(dataset_custom_path, 'data.yaml')
        cropped_yaml = os.path.join(cropped_dataset_path, 'data.yaml')
        
        if os.path.exists(original_yaml):
            with open(original_yaml, 'r') as f:
                yaml_data = yaml.safe_load(f)
            
            # CRITICAL FIX: Use dynamic paths instead of hardcoded ones
            yaml_data['path'] = cropped_dataset_path
            yaml_data['train'] = 'images/train'  # Relative paths
            yaml_data['val'] = 'images/valid'    # Relative paths
            
            with open(cropped_yaml, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=False)
        
        # CRITICAL: Comprehensive dataset validation with NaN prevention
        validation_stats = self.validate_dataset_integrity(cropped_dataset_path)
        logger.info(f"Dataset validation for {group_label}: {validation_stats}")
        
        # Log critical statistics
        logger.info(f"Dataset {group_label} final statistics:")
        logger.info(f"  - Total images: {validation_stats['total_images']}")
        logger.info(f"  - Images without annotations: {validation_stats['images_without_annotations']}")
        logger.info(f"  - Valid annotations: {validation_stats['valid_annotations']}")
        logger.info(f"  - Invalid annotations: {validation_stats['invalid_annotations']}")
        logger.info(f"  - Train pairs: {validation_stats.get('train_pairs', 0)}")
        logger.info(f"  - Valid pairs: {validation_stats.get('valid_pairs', 0)}")
        logger.info(f"  - Final validation split ratio: {validation_stats.get('final_split_ratio', 0):.1%}")
        
        # CRITICAL ERROR CHECKING
        if validation_stats['images_without_annotations'] > 0:
            logger.error(f"CRITICAL: {validation_stats['images_without_annotations']} images without annotations - WILL CAUSE NaN!")
        
        if validation_stats.get('final_split_ratio', 0) < 0.1:
            logger.error(f"CRITICAL: Insufficient validation split ({validation_stats.get('final_split_ratio', 0):.1%}) - WILL CAUSE NaN!")
        
        # Log validation issues
        for issue in validation_stats.get('issues', []):
            if 'CRITICAL' in issue:
                logger.error(issue)
            else:
                logger.warning(issue)
        
        if validation_stats['invalid_annotations'] > 0:
            logger.warning(f"Found {validation_stats['invalid_annotations']} invalid annotations in {group_label}")
            for issue in validation_stats['issues'][:5]:  # Log first 5 issues
                logger.warning(f"Dataset issue: {issue}")
        
        return cropped_dataset_path