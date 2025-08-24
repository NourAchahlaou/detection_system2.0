import os
import cv2
import numpy as np
import pandas as pd
import yaml
import torch
import math
from collections import defaultdict, Counter
import matplotlib.pyplot as plt
import json
from pathlib import Path
import logging
from typing import Dict, List, Tuple, Any
from ultralytics import YOLO
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DatasetNaNDiagnostic:
    """Comprehensive dataset diagnostic tool to identify NaN causes in YOLO training."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.issues = []
        self.stats = defaultdict(int)
        self.detailed_stats = defaultdict(list)
        
    def log_issue(self, severity: str, message: str, details: Any = None):
        """Log an issue with details."""
        issue = {
            'severity': severity,
            'message': message,
            'details': details
        }
        self.issues.append(issue)
        print(f"[{severity}] {message}")
        if details and isinstance(details, dict):
            for k, v in details.items():
                print(f"    {k}: {v}")
    
    def test_1_dataset_structure(self) -> Dict:
        """Test 1: Verify basic dataset structure."""
        print("\n" + "="*60)
        print("TEST 1: DATASET STRUCTURE VALIDATION")
        print("="*60)
        
        results = {
            'test_name': 'Dataset Structure',
            'status': 'PASS',
            'issues_found': []
        }
        
        # Check if dataset path exists
        if not self.dataset_path.exists():
            self.log_issue('CRITICAL', f'Dataset path does not exist: {self.dataset_path}')
            results['status'] = 'FAIL'
            return results
        
        # Check required directories
        required_dirs = [
            'images/train',
            'images/valid', 
            'labels/train',
            'labels/valid'
        ]
        
        for req_dir in required_dirs:
            dir_path = self.dataset_path / req_dir
            if not dir_path.exists():
                self.log_issue('CRITICAL', f'Missing required directory: {req_dir}')
                results['status'] = 'FAIL'
                results['issues_found'].append(f'Missing {req_dir}')
            else:
                # Count files in directory
                file_count = len(list(dir_path.glob('**/*')))
                print(f"✓ {req_dir}: {file_count} files")
                self.stats[f'{req_dir}_files'] = file_count
        
        # Check data.yaml
        yaml_path = self.dataset_path / 'data.yaml'
        if not yaml_path.exists():
            self.log_issue('CRITICAL', 'Missing data.yaml file')
            results['status'] = 'FAIL'
        else:
            try:
                with open(yaml_path, 'r') as f:
                    yaml_data = yaml.safe_load(f)
                    print(f"✓ data.yaml loaded successfully")
                    print(f"  Classes: {len(yaml_data.get('names', {}))}")
                    print(f"  Train path: {yaml_data.get('train', 'Not specified')}")
                    print(f"  Val path: {yaml_data.get('val', 'Not specified')}")
            except Exception as e:
                self.log_issue('CRITICAL', f'Invalid data.yaml: {str(e)}')
                results['status'] = 'FAIL'
        
        return results
    
    def test_2_image_label_matching(self) -> Dict:
        """Test 2: Check if every image has a corresponding label and vice versa."""
        print("\n" + "="*60)
        print("TEST 2: IMAGE-LABEL MATCHING")
        print("="*60)
        
        results = {
            'test_name': 'Image-Label Matching',
            'status': 'PASS',
            'orphaned_images': [],
            'orphaned_labels': [],
            'empty_labels': []
        }
        
        for split in ['train', 'valid']:
            images_dir = self.dataset_path / 'images' / split
            labels_dir = self.dataset_path / 'labels' / split
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
                
            print(f"\nChecking {split} split...")
            
            # Get all images and labels
            image_files = set()
            label_files = set()
            
            for img_path in images_dir.glob('**/*.jpg'):
                relative_path = img_path.relative_to(images_dir)
                image_files.add(str(relative_path.with_suffix('')))
            
            for img_path in images_dir.glob('**/*.jpeg'):
                relative_path = img_path.relative_to(images_dir)
                image_files.add(str(relative_path.with_suffix('')))
                
            for img_path in images_dir.glob('**/*.png'):
                relative_path = img_path.relative_to(images_dir)
                image_files.add(str(relative_path.with_suffix('')))
            
            for label_path in labels_dir.glob('**/*.txt'):
                relative_path = label_path.relative_to(labels_dir)
                label_files.add(str(relative_path.with_suffix('')))
            
            # Find orphaned images (images without labels)
            orphaned_images = image_files - label_files
            orphaned_labels = label_files - image_files
            
            print(f"  Images: {len(image_files)}")
            print(f"  Labels: {len(label_files)}")
            print(f"  Orphaned images: {len(orphaned_images)}")
            print(f"  Orphaned labels: {len(orphaned_labels)}")
            
            if orphaned_images:
                self.log_issue('CRITICAL', f'{split}: {len(orphaned_images)} images without labels (WILL CAUSE NaN)')
                results['status'] = 'FAIL'
                results['orphaned_images'].extend([f'{split}/{img}' for img in list(orphaned_images)[:5]])  # Show first 5
            
            if orphaned_labels:
                self.log_issue('WARNING', f'{split}: {len(orphaned_labels)} labels without images')
                results['orphaned_labels'].extend([f'{split}/{lbl}' for lbl in list(orphaned_labels)[:5]])
            
            # Check for empty label files
            empty_labels = []
            for label_file in label_files:
                label_path = labels_dir / f"{label_file}.txt"
                if label_path.exists():
                    try:
                        with open(label_path, 'r') as f:
                            content = f.read().strip()
                        if not content:
                            empty_labels.append(str(label_file))
                    except Exception:
                        pass
            
            if empty_labels:
                self.log_issue('CRITICAL', f'{split}: {len(empty_labels)} empty label files (WILL CAUSE NaN)')
                results['status'] = 'FAIL'
                results['empty_labels'].extend([f'{split}/{lbl}' for lbl in empty_labels[:5]])
            
            self.stats[f'{split}_images'] = len(image_files)
            self.stats[f'{split}_labels'] = len(label_files)
            self.stats[f'{split}_orphaned_images'] = len(orphaned_images)
            self.stats[f'{split}_empty_labels'] = len(empty_labels)
        
        return results
    
    def test_3_annotation_validity(self) -> Dict:
        """Test 3: Validate all annotation coordinates and values."""
        print("\n" + "="*60)
        print("TEST 3: ANNOTATION VALIDITY")
        print("="*60)
        
        results = {
            'test_name': 'Annotation Validity',
            'status': 'PASS',
            'invalid_coords': [],
            'out_of_bounds': [],
            'extreme_values': [],
            'malformed_lines': []
        }
        
        coord_stats = {
            'x_centers': [], 'y_centers': [], 'widths': [], 'heights': [],
            'x_mins': [], 'x_maxs': [], 'y_mins': [], 'y_maxs': []
        }
        
        total_annotations = 0
        invalid_count = 0
        
        for split in ['train', 'valid']:
            labels_dir = self.dataset_path / 'labels' / split
            if not labels_dir.exists():
                continue
                
            print(f"\nValidating {split} annotations...")
            
            for label_path in labels_dir.glob('**/*.txt'):
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    for line_num, line in enumerate(lines, 1):
                        line = line.strip()
                        if not line:
                            continue
                            
                        total_annotations += 1
                        parts = line.split()
                        
                        # Check format
                        if len(parts) != 5:
                            self.log_issue('ERROR', f'Malformed annotation line in {label_path}:{line_num}', 
                                         {'line': line, 'parts_count': len(parts)})
                            results['malformed_lines'].append(f'{label_path}:{line_num}')
                            results['status'] = 'FAIL'
                            invalid_count += 1
                            continue
                        
                        try:
                            class_id = int(parts[0])
                            x_center = float(parts[1])
                            y_center = float(parts[2])
                            width = float(parts[3])
                            height = float(parts[4])
                            
                            # Check for NaN or infinite values
                            values = [x_center, y_center, width, height]
                            if any(math.isnan(v) or math.isinf(v) for v in values):
                                self.log_issue('CRITICAL', f'NaN/Inf values in {label_path}:{line_num}', 
                                             {'values': values})
                                results['status'] = 'FAIL'
                                invalid_count += 1
                                continue
                            
                            # Check coordinate bounds (should be 0-1)
                            coord_issues = []
                            if not (0 <= x_center <= 1):
                                coord_issues.append(f'x_center: {x_center}')
                            if not (0 <= y_center <= 1):
                                coord_issues.append(f'y_center: {y_center}')
                            if not (0 < width <= 1):
                                coord_issues.append(f'width: {width}')
                            if not (0 < height <= 1):
                                coord_issues.append(f'height: {height}')
                            
                            if coord_issues:
                                self.log_issue('CRITICAL', f'Invalid coordinates in {label_path}:{line_num}', 
                                             {'issues': coord_issues})
                                results['invalid_coords'].append(f'{label_path}:{line_num}')
                                results['status'] = 'FAIL'
                                invalid_count += 1
                                continue
                            
                            # Check if bbox goes out of image bounds
                            x_min = x_center - width/2
                            x_max = x_center + width/2
                            y_min = y_center - height/2
                            y_max = y_center + height/2
                            
                            if x_min < 0 or x_max > 1 or y_min < 0 or y_max > 1:
                                self.log_issue('ERROR', f'Bounding box out of bounds in {label_path}:{line_num}',
                                             {'bbox': [x_min, y_min, x_max, y_max]})
                                results['out_of_bounds'].append(f'{label_path}:{line_num}')
                                results['status'] = 'FAIL'
                                invalid_count += 1
                            
                            # Check for extreme values that might cause training issues
                            if width < 0.01 or height < 0.01:
                                self.log_issue('WARNING', f'Very small bbox in {label_path}:{line_num}',
                                             {'width': width, 'height': height})
                                results['extreme_values'].append(f'{label_path}:{line_num}')
                            
                            if width > 0.9 or height > 0.9:
                                self.log_issue('WARNING', f'Very large bbox in {label_path}:{line_num}',
                                             {'width': width, 'height': height})
                                results['extreme_values'].append(f'{label_path}:{line_num}')
                            
                            # Collect statistics
                            coord_stats['x_centers'].append(x_center)
                            coord_stats['y_centers'].append(y_center)
                            coord_stats['widths'].append(width)
                            coord_stats['heights'].append(height)
                            coord_stats['x_mins'].append(x_min)
                            coord_stats['x_maxs'].append(x_max)
                            coord_stats['y_mins'].append(y_min)
                            coord_stats['y_maxs'].append(y_max)
                            
                        except ValueError as e:
                            self.log_issue('ERROR', f'Cannot parse values in {label_path}:{line_num}',
                                         {'line': line, 'error': str(e)})
                            results['malformed_lines'].append(f'{label_path}:{line_num}')
                            results['status'] = 'FAIL'
                            invalid_count += 1
                            
                except Exception as e:
                    self.log_issue('ERROR', f'Cannot read label file {label_path}', {'error': str(e)})
                    results['status'] = 'FAIL'
        
        print(f"\nAnnotation Statistics:")
        print(f"  Total annotations: {total_annotations}")
        print(f"  Invalid annotations: {invalid_count}")
        print(f"  Validity rate: {((total_annotations - invalid_count) / max(total_annotations, 1) * 100):.2f}%")
        
        if coord_stats['x_centers']:
            for coord_type, values in coord_stats.items():
                if values:
                    print(f"  {coord_type}: min={min(values):.4f}, max={max(values):.4f}, mean={np.mean(values):.4f}")
        
        self.stats['total_annotations'] = total_annotations
        self.stats['invalid_annotations'] = invalid_count
        
        return results
    
    def test_4_image_integrity(self) -> Dict:
        """Test 4: Check image files for corruption and valid dimensions."""
        print("\n" + "="*60)
        print("TEST 4: IMAGE INTEGRITY")
        print("="*60)
        
        results = {
            'test_name': 'Image Integrity',
            'status': 'PASS',
            'corrupted_images': [],
            'dimension_issues': [],
            'small_images': []
        }
        
        image_stats = {
            'widths': [], 'heights': [], 'aspects': [], 'channels': []
        }
        
        total_images = 0
        corrupted_count = 0
        
        for split in ['train', 'valid']:
            images_dir = self.dataset_path / 'images' / split
            if not images_dir.exists():
                continue
                
            print(f"\nChecking {split} images...")
            
            for img_path in images_dir.glob('**/*'):
                if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    total_images += 1
                    
                    try:
                        img = cv2.imread(str(img_path))
                        if img is None:
                            self.log_issue('CRITICAL', f'Cannot read image: {img_path}')
                            results['corrupted_images'].append(str(img_path))
                            results['status'] = 'FAIL'
                            corrupted_count += 1
                            continue
                        
                        height, width = img.shape[:2]
                        channels = img.shape[2] if len(img.shape) == 3 else 1
                        aspect_ratio = width / height
                        
                        # Check for valid dimensions
                        if width <= 0 or height <= 0:
                            self.log_issue('CRITICAL', f'Invalid dimensions in {img_path}',
                                         {'width': width, 'height': height})
                            results['dimension_issues'].append(str(img_path))
                            results['status'] = 'FAIL'
                            continue
                        
                        # Check for very small images
                        if width < 32 or height < 32:
                            self.log_issue('WARNING', f'Very small image: {img_path}',
                                         {'width': width, 'height': height})
                            results['small_images'].append(str(img_path))
                        
                        # Check for unusual aspect ratios
                        if aspect_ratio > 10 or aspect_ratio < 0.1:
                            self.log_issue('WARNING', f'Unusual aspect ratio in {img_path}',
                                         {'aspect_ratio': aspect_ratio})
                        
                        # Collect statistics
                        image_stats['widths'].append(width)
                        image_stats['heights'].append(height)
                        image_stats['aspects'].append(aspect_ratio)
                        image_stats['channels'].append(channels)
                        
                    except Exception as e:
                        self.log_issue('CRITICAL', f'Error processing image {img_path}', {'error': str(e)})
                        results['corrupted_images'].append(str(img_path))
                        results['status'] = 'FAIL'
                        corrupted_count += 1
        
        print(f"\nImage Statistics:")
        print(f"  Total images: {total_images}")
        print(f"  Corrupted images: {corrupted_count}")
        
        if image_stats['widths']:
            print(f"  Width range: {min(image_stats['widths'])}-{max(image_stats['widths'])}")
            print(f"  Height range: {min(image_stats['heights'])}-{max(image_stats['heights'])}")
            print(f"  Aspect ratio range: {min(image_stats['aspects']):.2f}-{max(image_stats['aspects']):.2f}")
            print(f"  Channels: {set(image_stats['channels'])}")
        
        self.stats['total_images'] = total_images
        self.stats['corrupted_images'] = corrupted_count
        
        return results
    
    def test_5_class_distribution(self) -> Dict:
        """Test 5: Analyze class distribution and balance."""
        print("\n" + "="*60)
        print("TEST 5: CLASS DISTRIBUTION")
        print("="*60)
        
        results = {
            'test_name': 'Class Distribution',
            'status': 'PASS',
            'class_counts': {},
            'imbalance_issues': []
        }
        
        class_counts = defaultdict(int)
        split_class_counts = {'train': defaultdict(int), 'valid': defaultdict(int)}
        
        for split in ['train', 'valid']:
            labels_dir = self.dataset_path / 'labels' / split
            if not labels_dir.exists():
                continue
                
            for label_path in labels_dir.glob('**/*.txt'):
                try:
                    with open(label_path, 'r') as f:
                        lines = f.readlines()
                    
                    for line in lines:
                        line = line.strip()
                        if line:
                            try:
                                class_id = int(line.split()[0])
                                class_counts[class_id] += 1
                                split_class_counts[split][class_id] += 1
                            except (ValueError, IndexError):
                                pass
                except Exception:
                    pass
        
        print(f"\nClass Distribution:")
        total_annotations = sum(class_counts.values())
        
        for class_id in sorted(class_counts.keys()):
            count = class_counts[class_id]
            percentage = (count / max(total_annotations, 1)) * 100
            train_count = split_class_counts['train'][class_id]
            valid_count = split_class_counts['valid'][class_id]
            
            print(f"  Class {class_id}: {count} annotations ({percentage:.1f}%) - Train: {train_count}, Valid: {valid_count}")
            
            # Check for severe class imbalance
            if percentage < 1.0:
                self.log_issue('WARNING', f'Class {class_id} has very few samples ({count})')
                results['imbalance_issues'].append(f'Class {class_id}: {count} samples')
            
            # Check if class exists in validation
            if valid_count == 0:
                self.log_issue('CRITICAL', f'Class {class_id} missing from validation set (WILL CAUSE NaN)')
                results['status'] = 'FAIL'
                results['imbalance_issues'].append(f'Class {class_id}: no validation samples')
        
        # Check for empty classes
        if not class_counts:
            self.log_issue('CRITICAL', 'No classes found in dataset (WILL CAUSE NaN)')
            results['status'] = 'FAIL'
        
        results['class_counts'] = dict(class_counts)
        self.stats['num_classes'] = len(class_counts)
        
        return results
    
    def test_6_validation_split(self) -> Dict:
        """Test 6: Validate train/validation split ratios."""
        print("\n" + "="*60)
        print("TEST 6: VALIDATION SPLIT")
        print("="*60)
        
        results = {
            'test_name': 'Validation Split',
            'status': 'PASS',
            'split_ratio': 0,
            'issues': []
        }
        
        train_images = 0
        valid_images = 0
        
        # Count valid image-label pairs
        for split in ['train', 'valid']:
            images_dir = self.dataset_path / 'images' / split
            labels_dir = self.dataset_path / 'labels' / split
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
                
            pairs = 0
            for img_path in images_dir.glob('**/*.jpg'):
                label_path = labels_dir / img_path.relative_to(images_dir).with_suffix('.txt')
                if label_path.exists():
                    try:
                        with open(label_path, 'r') as f:
                            content = f.read().strip()
                        if content:  # Has non-empty content
                            pairs += 1
                    except:
                        pass
            
            if split == 'train':
                train_images = pairs
            else:
                valid_images = pairs
        
        total_images = train_images + valid_images
        valid_ratio = valid_images / max(total_images, 1)
        
        print(f"\nSplit Analysis:")
        print(f"  Train images with labels: {train_images}")
        print(f"  Validation images with labels: {valid_images}")
        print(f"  Total valid pairs: {total_images}")
        print(f"  Validation ratio: {valid_ratio:.1%}")
        
        results['split_ratio'] = valid_ratio
        
        # Critical checks
        if valid_images == 0:
            self.log_issue('CRITICAL', 'No validation images (WILL CAUSE NaN)')
            results['status'] = 'FAIL'
            results['issues'].append('No validation images')
        
        if valid_ratio < 0.05:  # Less than 5%
            self.log_issue('CRITICAL', f'Insufficient validation split ({valid_ratio:.1%}) - Minimum 5% recommended')
            results['status'] = 'FAIL'
            results['issues'].append(f'Low validation ratio: {valid_ratio:.1%}')
        
        if valid_ratio > 0.5:  # More than 50%
            self.log_issue('WARNING', f'Very high validation split ({valid_ratio:.1%}) - Training data might be insufficient')
            results['issues'].append(f'High validation ratio: {valid_ratio:.1%}')
        
        self.stats['valid_ratio'] = valid_ratio
        self.stats['train_valid_pairs'] = train_images
        self.stats['valid_valid_pairs'] = valid_images
        
        return results
    
    def test_7_model_loading(self) -> Dict:
        """Test 7: Try loading dataset with YOLO to catch potential issues."""
        print("\n" + "="*60)
        print("TEST 7: MODEL LOADING TEST")
        print("="*60)
        
        results = {
            'test_name': 'Model Loading',
            'status': 'PASS',
            'loading_errors': []
        }
        
        try:
            # Test basic model creation
            model = YOLO('yolov8n.pt')
            print("✓ Base model loaded successfully")
            
            # Test dataset loading
            yaml_path = self.dataset_path / 'data.yaml'
            if yaml_path.exists():
                try:
                    # Try to create a training configuration (don't actually train)
                    with open(yaml_path, 'r') as f:
                        yaml_data = yaml.safe_load(f)
                    
                    # Check paths in yaml
                    train_path = yaml_data.get('train', '')
                    val_path = yaml_data.get('val', '')
                    
                    # Convert to absolute paths if relative
                    if not os.path.isabs(train_path):
                        train_path = self.dataset_path / train_path
                    if not os.path.isabs(val_path):
                        val_path = self.dataset_path / val_path
                    
                    if not Path(train_path).exists():
                        self.log_issue('CRITICAL', f'Train path in yaml does not exist: {train_path}')
                        results['status'] = 'FAIL'
                        results['loading_errors'].append(f'Missing train path: {train_path}')
                    
                    if not Path(val_path).exists():
                        self.log_issue('CRITICAL', f'Validation path in yaml does not exist: {val_path}')
                        results['status'] = 'FAIL'
                        results['loading_errors'].append(f'Missing val path: {val_path}')
                    
                    print("✓ Dataset paths validation passed")
                    
                except Exception as e:
                    self.log_issue('ERROR', f'Dataset configuration error: {str(e)}')
                    results['status'] = 'FAIL'
                    results['loading_errors'].append(str(e))
            else:
                self.log_issue('CRITICAL', 'No data.yaml found')
                results['status'] = 'FAIL'
                results['loading_errors'].append('Missing data.yaml')
                
        except Exception as e:
            self.log_issue('CRITICAL', f'Model loading failed: {str(e)}')
            results['status'] = 'FAIL'
            results['loading_errors'].append(str(e))
        
        return results
    
    def test_8_memory_and_batch_test(self) -> Dict:
        """Test 8: Test memory usage and potential batch processing issues."""
        print("\n" + "="*60)
        print("TEST 8: MEMORY AND BATCH PROCESSING")
        print("="*60)
        
        results = {
            'test_name': 'Memory and Batch Test',
            'status': 'PASS',
            'memory_issues': [],
            'batch_recommendations': {}
        }
        
        try:
            # Check image sizes to estimate memory usage
            total_pixels = 0
            image_count = 0
            max_image_size = 0
            
            for split in ['train', 'valid']:
                images_dir = self.dataset_path / 'images' / split
                if not images_dir.exists():
                    continue
                    
                for img_path in list(images_dir.glob('**/*.jpg'))[:50]:  # Sample first 50 images
                    try:
                        img = cv2.imread(str(img_path))
                        if img is not None:
                            height, width = img.shape[:2]
                            pixels = height * width
                            total_pixels += pixels
                            image_count += 1
                            max_image_size = max(max_image_size, max(height, width))
                    except:
                        pass
            
            if image_count > 0:
                avg_pixels = total_pixels / image_count
                estimated_memory_per_image = (avg_pixels * 3 * 4) / (1024 * 1024)  # RGB, float32, MB
                
                print(f"  Sampled images: {image_count}")
                print(f"  Average pixels per image: {avg_pixels:,.0f}")
                print(f"  Maximum image dimension: {max_image_size}")
                print(f"  Estimated memory per image: {estimated_memory_per_image:.2f} MB")
                
                # Memory-based batch size recommendations
                available_memory = 4000  # Assume 4GB available
                recommended_batch = max(1, int(available_memory / (estimated_memory_per_image * 10)))  # Conservative estimate
                
                results['batch_recommendations'] = {
                    'recommended_batch_size': min(recommended_batch, 8),
                    'memory_per_image_mb': estimated_memory_per_image,
                    'max_image_dimension': max_image_size
                }
                
                print(f"  Recommended batch size: {results['batch_recommendations']['recommended_batch_size']}")
                
                if estimated_memory_per_image > 50:
                    self.log_issue('WARNING', f'High memory usage per image: {estimated_memory_per_image:.2f} MB')
                    results['memory_issues'].append(f'High memory per image: {estimated_memory_per_image:.2f}MB')
            
        except Exception as e:
            self.log_issue('ERROR', f'Memory analysis failed: {str(e)}')
            results['memory_issues'].append(str(e))
        
        return results
    
    def generate_summary_report(self, test_results: List[Dict]) -> Dict:
        """Generate comprehensive summary report."""
        print("\n" + "="*60)
        print("DIAGNOSTIC SUMMARY REPORT")
        print("="*60)
        
        critical_issues = []
        warnings = []
        failed_tests = []
        passed_tests = []
        
        # Categorize issues
        for issue in self.issues:
            if issue['severity'] == 'CRITICAL':
                critical_issues.append(issue['message'])
            elif issue['severity'] in ['WARNING', 'ERROR']:
                warnings.append(issue['message'])
        
        # Categorize test results
        for result in test_results:
            if result['status'] == 'FAIL':
                failed_tests.append(result['test_name'])
            else:
                passed_tests.append(result['test_name'])
        
        # Determine overall status
        overall_status = 'FAIL' if critical_issues or failed_tests else 'PASS'
        nan_risk = 'HIGH' if critical_issues else ('MEDIUM' if warnings else 'LOW')
        
        summary = {
            'overall_status': overall_status,
            'nan_risk_level': nan_risk,
            'tests_passed': len(passed_tests),
            'tests_failed': len(failed_tests),
            'critical_issues_count': len(critical_issues),
            'warnings_count': len(warnings),
            'critical_issues': critical_issues,
            'warnings': warnings,
            'failed_tests': failed_tests,
            'dataset_stats': dict(self.stats),
            'recommendations': []
        }
        
        print(f"\nOVERALL STATUS: {overall_status}")
        print(f"NaN RISK LEVEL: {nan_risk}")
        print(f"Tests Passed: {len(passed_tests)}/{len(test_results)}")
        print(f"Critical Issues: {len(critical_issues)}")
        print(f"Warnings: {len(warnings)}")
        
        if critical_issues:
            print(f"\n🚨 CRITICAL ISSUES (WILL CAUSE NaN):")
            for i, issue in enumerate(critical_issues, 1):
                print(f"  {i}. {issue}")
        
        if warnings:
            print(f"\n⚠️  WARNINGS:")
            for i, warning in enumerate(warnings, 1):
                print(f"  {i}. {warning}")
        
        # Generate recommendations
        recommendations = []
        
        if self.stats.get('train_orphaned_images', 0) > 0:
            recommendations.append("Remove orphaned images (images without labels) from training set")
        
        if self.stats.get('valid_orphaned_images', 0) > 0:
            recommendations.append("Remove orphaned images from validation set")
        
        if self.stats.get('train_empty_labels', 0) > 0 or self.stats.get('valid_empty_labels', 0) > 0:
            recommendations.append("Remove empty label files or their corresponding images")
        
        if self.stats.get('valid_ratio', 0) < 0.1:
            recommendations.append("Increase validation split to at least 10-15% of total dataset")
        
        if self.stats.get('corrupted_images', 0) > 0:
            recommendations.append("Fix or remove corrupted images")
        
        if self.stats.get('invalid_annotations', 0) > 0:
            recommendations.append("Fix invalid annotation coordinates (must be 0-1 range)")
        
        if self.stats.get('num_classes', 0) > 0:
            recommendations.append("Ensure all classes have samples in both train and validation sets")
        
        recommendations.extend([
            "Use conservative hyperparameters (low learning rate, high patience)",
            "Start with batch_size=1 for GPU training",
            "Use gradient clipping to prevent exploding gradients",
            "Monitor training closely for early NaN detection"
        ])
        
        summary['recommendations'] = recommendations
        
        print(f"\n💡 RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        return summary
    
    def save_detailed_report(self, test_results: List[Dict], summary: Dict, output_file: str = None):
        """Save detailed report to file."""
        if output_file is None:
            output_file = self.dataset_path.parent / f"nan_diagnostic_report_{self.dataset_path.name}.json"
        
        report = {
            'dataset_path': str(self.dataset_path),
            'timestamp': str(pd.Timestamp.now()) if 'pd' in globals() else 'N/A',
            'summary': summary,
            'detailed_test_results': test_results,
            'all_issues': self.issues,
            'statistics': dict(self.stats)
        }
        
        try:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n📄 Detailed report saved to: {output_file}")
        except Exception as e:
            print(f"Failed to save report: {e}")
    
    def run_all_tests(self) -> Dict:
        """Run all diagnostic tests and generate comprehensive report."""
        print("🔍 Starting comprehensive dataset NaN diagnostic...")
        print(f"📁 Dataset: {self.dataset_path}")
        
        test_results = []
        
        # Run all tests
        test_results.append(self.test_1_dataset_structure())
        test_results.append(self.test_2_image_label_matching())
        test_results.append(self.test_3_annotation_validity())
        test_results.append(self.test_4_image_integrity())
        test_results.append(self.test_5_class_distribution())
        test_results.append(self.test_6_validation_split())
        test_results.append(self.test_7_model_loading())
        test_results.append(self.test_8_memory_and_batch_test())
        
        # Generate summary
        summary = self.generate_summary_report(test_results)
        
        # Save detailed report
        self.save_detailed_report(test_results, summary)
        
        return summary


class QuickNaNFixer:
    """Quick automatic fixer for common NaN-causing issues."""
    
    def __init__(self, dataset_path: str):
        self.dataset_path = Path(dataset_path)
        self.fixes_applied = []
    
    def fix_orphaned_images(self) -> int:
        """Remove images without corresponding labels."""
        fixed_count = 0
        
        for split in ['train', 'valid']:
            images_dir = self.dataset_path / 'images' / split
            labels_dir = self.dataset_path / 'labels' / split
            
            if not images_dir.exists() or not labels_dir.exists():
                continue
            
            for img_path in images_dir.glob('**/*.jpg'):
                label_path = labels_dir / img_path.relative_to(images_dir).with_suffix('.txt')
                
                has_valid_label = False
                if label_path.exists():
                    try:
                        with open(label_path, 'r') as f:
                            content = f.read().strip()
                        if content:
                            has_valid_label = True
                    except:
                        pass
                
                if not has_valid_label:
                    try:
                        img_path.unlink()
                        fixed_count += 1
                        print(f"Removed orphaned image: {img_path}")
                    except Exception as e:
                        print(f"Failed to remove {img_path}: {e}")
        
        if fixed_count > 0:
            self.fixes_applied.append(f"Removed {fixed_count} orphaned images")
        
        return fixed_count
    
    def fix_empty_labels(self) -> int:
        """Remove empty label files and their corresponding images."""
        fixed_count = 0
        
        for split in ['train', 'valid']:
            images_dir = self.dataset_path / 'images' / split
            labels_dir = self.dataset_path / 'labels' / split
            
            if not labels_dir.exists():
                continue
            
            for label_path in labels_dir.glob('**/*.txt'):
                try:
                    with open(label_path, 'r') as f:
                        content = f.read().strip()
                    
                    if not content:
                        # Remove empty label
                        label_path.unlink()
                        
                        # Remove corresponding image
                        img_path = images_dir / label_path.relative_to(labels_dir).with_suffix('.jpg')
                        if img_path.exists():
                            img_path.unlink()
                            print(f"Removed image with empty label: {img_path}")
                        
                        fixed_count += 1
                        
                except Exception as e:
                    print(f"Error processing {label_path}: {e}")
        
        if fixed_count > 0:
            self.fixes_applied.append(f"Removed {fixed_count} empty label files and images")
        
        return fixed_count
    
    def fix_validation_split(self, target_ratio: float = 0.15) -> int:
        """Move some training images to validation to achieve target ratio."""
        import random
        
        train_images_dir = self.dataset_path / 'images' / 'train'
        valid_images_dir = self.dataset_path / 'images' / 'valid'
        train_labels_dir = self.dataset_path / 'labels' / 'train'
        valid_labels_dir = self.dataset_path / 'labels' / 'valid'
        
        # Create validation directories if they don't exist
        valid_images_dir.mkdir(parents=True, exist_ok=True)
        valid_labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Get valid train pairs
        train_pairs = []
        for img_path in train_images_dir.glob('**/*.jpg'):
            label_path = train_labels_dir / img_path.relative_to(train_images_dir).with_suffix('.txt')
            if label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        if f.read().strip():
                            train_pairs.append((img_path, label_path))
                except:
                    pass
        
        # Get valid validation pairs
        valid_pairs = []
        for img_path in valid_images_dir.glob('**/*.jpg'):
            label_path = valid_labels_dir / img_path.relative_to(valid_images_dir).with_suffix('.txt')
            if label_path.exists():
                try:
                    with open(label_path, 'r') as f:
                        if f.read().strip():
                            valid_pairs.append((img_path, label_path))
                except:
                    pass
        
        total_pairs = len(train_pairs) + len(valid_pairs)
        current_valid_ratio = len(valid_pairs) / max(total_pairs, 1)
        
        if current_valid_ratio >= target_ratio:
            return 0  # Already sufficient
        
        target_valid_count = int(total_pairs * target_ratio)
        needed_valid = target_valid_count - len(valid_pairs)
        to_move = min(needed_valid, len(train_pairs) // 2)  # Don't move more than half
        
        if to_move <= 0:
            return 0
        
        # Randomly select pairs to move
        random.seed(42)
        pairs_to_move = random.sample(train_pairs, to_move)
        
        moved_count = 0
        for img_path, label_path in pairs_to_move:
            try:
                # Create destination paths
                img_rel_path = img_path.relative_to(train_images_dir)
                label_rel_path = label_path.relative_to(train_labels_dir)
                
                dst_img = valid_images_dir / img_rel_path
                dst_label = valid_labels_dir / label_rel_path
                
                # Create parent directories
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                dst_label.parent.mkdir(parents=True, exist_ok=True)
                
                # Move files
                img_path.rename(dst_img)
                label_path.rename(dst_label)
                
                moved_count += 1
                
            except Exception as e:
                print(f"Failed to move {img_path}: {e}")
        
        if moved_count > 0:
            self.fixes_applied.append(f"Moved {moved_count} pairs to validation (ratio: {target_ratio:.1%})")
        
        return moved_count
    
    def apply_all_fixes(self):
        """Apply all automatic fixes."""
        print("🔧 Applying automatic fixes for NaN prevention...")
        
        orphaned_fixed = self.fix_orphaned_images()
        empty_fixed = self.fix_empty_labels()
        split_fixed = self.fix_validation_split()
        
        print(f"\n✅ Fixes Applied:")
        for fix in self.fixes_applied:
            print(f"  - {fix}")
        
        if not self.fixes_applied:
            print("  - No fixes needed")
        
        return len(self.fixes_applied) > 0


def main():
    """Main function to run the diagnostic tool."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive Dataset NaN Diagnostic Tool')
    parser.add_argument('dataset_path', help='Path to the dataset directory')
    parser.add_argument('--fix', action='store_true', help='Apply automatic fixes for common issues')
    parser.add_argument('--output', help='Output file for detailed report')
    
    try:
        args = parser.parse_args()
    except:
        # Default values for testing
        dataset_path = input("Enter dataset path: ").strip()
        fix_issues = input("Apply automatic fixes? (y/n): ").lower().startswith('y')
        output_file = input("Output file (press Enter for default): ").strip() or None
        
        class Args:
            def __init__(self):
                self.dataset_path = dataset_path
                self.fix = fix_issues
                self.output = output_file
        
        args = Args()
    
    # Run diagnostic
    diagnostic = DatasetNaNDiagnostic(args.dataset_path)
    summary = diagnostic.run_all_tests()
    
    # Apply fixes if requested
    # if args.fix:
    #     fixer = QuickNaNFixer(args.dataset_path)
    #     fixes_applied = fixer.apply_all_fixes()
        
    #     if fixes_applied:
    #         print("\n🔄 Re-running diagnostic after fixes...")
    #         diagnostic = DatasetNaNDiagnostic(args.dataset_path)
    #         summary = diagnostic.run_all_tests()
    
    # Final recommendations
    print(f"\n📋 FINAL ASSESSMENT:")
    print(f"Dataset Status: {summary['overall_status']}")
    print(f"NaN Risk Level: {summary['nan_risk_level']}")
    
    if summary['nan_risk_level'] == 'HIGH':
        print("\n🚨 HIGH RISK: This dataset is likely to cause NaN during training!")
        print("   Please address critical issues before training.")
    elif summary['nan_risk_level'] == 'MEDIUM':
        print("\n⚠️  MEDIUM RISK: Some issues detected that might cause training problems.")
        print("   Consider addressing warnings before training.")
    else:
        print("\n✅ LOW RISK: Dataset appears to be well-prepared for training.")
    
    return summary


if __name__ == "__main__":
    main()