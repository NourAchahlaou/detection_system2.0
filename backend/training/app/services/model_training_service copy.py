import asyncio
import os
import logging
import shutil
from training.app.db.models.annotation import Annotation
import torch
from sqlalchemy.orm import Session
from ultralytics import YOLO
import yaml
from datetime import datetime
import time
import re
from collections import defaultdict
from typing import List, Dict, Tuple
import gc
import math

from training.app.db.models.piece_image import PieceImage
from training.app.services.basic_rotation_service import rotate_and_update_images
from training.app.db.models.piece import Piece
from training.app.db.models.training import TrainingSession
from training.app.db.session import create_new_session, safe_commit, safe_close

# Set up logging with dedicated log volume
log_dir = os.getenv("LOG_PATH", "/usr/srv/logs")
log_file = os.path.join(log_dir, "group_model_training_service.log")

# Ensure log directory exists and is writable
os.makedirs(log_dir, exist_ok=True)

# Configure logging with both console and file handlers
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler(log_file, mode='a')  # File output
    ]
)

logger = logging.getLogger(__name__)

# Global stop event
stop_event = asyncio.Event()
stop_sign = False

# Image size constants based on device - KEEP YOUR PREFERRED SIZES
GPU_IMAGE_SIZE = 512  # Your preferred size 512
CPU_IMAGE_SIZE = 640  # Your preferred size

async def stop_training():
    """Set the stop event to signal training to stop."""
    global stop_sign
    stop_event.set()
    stop_sign = True
    logger.info("Stop training signal sent.")

def extract_group_from_piece_label(piece_label: str) -> str:
    """Extract group identifier from piece label (e.g., 'E539.12345' -> 'E539')"""
    match = re.match(r'([A-Z]\d{3})', piece_label)
    if match:
        return match.group(1)
    return None

def group_pieces_by_prefix(piece_labels: List[str]) -> Dict[str, List[str]]:
    """Group piece labels by their prefix (e.g., E539, G053, etc.)"""
    groups = defaultdict(list)
    for piece_label in piece_labels:
        group = extract_group_from_piece_label(piece_label)
        if group:
            groups[group].append(piece_label)
        else:
            logger.warning(f"Could not extract group from piece label: {piece_label}")
    return dict(groups)

def select_device():
    """Select the best available device with conservative GPU settings."""
    
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    
    nvidia_visible_devices = os.environ.get('NVIDIA_VISIBLE_DEVICES', 'Not set')
    nvidia_driver_capabilities = os.environ.get('NVIDIA_DRIVER_CAPABILITIES', 'Not set')
    logger.info(f"NVIDIA_VISIBLE_DEVICES: {nvidia_visible_devices}")
    logger.info(f"NVIDIA_DRIVER_CAPABILITIES: {nvidia_driver_capabilities}")
    
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Number of CUDA devices: {device_count}")
        
        for i in range(device_count):
            device_props = torch.cuda.get_device_properties(i)
            logger.info(f"Device {i}: {device_props.name}")
            logger.info(f"  Memory: {device_props.total_memory / 1024**3:.1f} GB")
            logger.info(f"  Compute capability: {device_props.major}.{device_props.minor}")
        
        device = torch.device('cuda:0')
        logger.info(f"Selected device: {device}")
        
        # Set memory fraction for small GPUs
        total_memory = torch.cuda.get_device_properties(0).total_memory
        if total_memory <= 4 * 1024**3:  # 4GB or less
            torch.cuda.set_per_process_memory_fraction(0.6)  # Reduced from 0.7 for more stability
            logger.info("Set GPU memory fraction to 0.6 for small GPU")
        
        try:
            test_tensor = torch.randn(10, 10).to(device)
            del test_tensor
            torch.cuda.empty_cache()
            logger.info("GPU functionality test: PASSED")
        except Exception as e:
            logger.error(f"GPU functionality test failed: {e}")
            logger.info("Falling back to CPU")
            device = torch.device('cpu')
        
        return device
    else:
        logger.info("No GPU detected. Using CPU for training")
        return torch.device('cpu')

def adjust_batch_size(device, base_batch_size=8):
    """Ultra-conservative batch sizes for GPU stability - PREVENT NaN."""
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(0).total_memory
        if total_memory <= 4 * 1024**3:  # T600 4GB
            return 1  # Single batch for maximum stability
        elif total_memory <= 6 * 1024**3:
            return 1  # Also use 1 for 6GB cards to prevent NaN
        else:
            return 2  # Very conservative even for larger GPUs
    else:
        return max(1, base_batch_size // 4)  # More conservative CPU batch size

def get_training_image_size(device):
    """Return image size based on device type."""
    if device.type == 'cuda':
        image_size = GPU_IMAGE_SIZE
        logger.info(f"Using GPU image size: {image_size}")
    else:
        image_size = CPU_IMAGE_SIZE
        logger.info(f"Using CPU image size: {image_size}")
    
    return image_size

def force_cleanup_gpu_memory():
    """Aggressive GPU memory cleanup."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        gc.collect()
        
        # Log memory status
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU memory after cleanup - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")

def cleanup_old_datasets(dataset_base_path: str, keep_latest: int = 2):
    """Clean up old dataset_custom directories to save space."""
    try:
        dataset_custom_base = os.path.join(dataset_base_path, 'dataset_custom')
        if not os.path.exists(dataset_custom_base):
            return
        
        group_dirs = []
        for item in os.listdir(dataset_custom_base):
            item_path = os.path.join(dataset_custom_base, item)
            if os.path.isdir(item_path):
                creation_time = os.path.getctime(item_path)
                group_dirs.append((item, creation_time, item_path))
        
        group_dirs.sort(key=lambda x: x[1], reverse=True)
        
        if len(group_dirs) > keep_latest:
            for group_name, _, group_path in group_dirs[keep_latest:]:
                logger.info(f"Cleaning up old dataset directory: {group_path}")
                shutil.rmtree(group_path)
        
        logger.info(f"Dataset cleanup completed. Kept {min(len(group_dirs), keep_latest)} most recent group datasets.")
    except Exception as e:
        logger.warning(f"Dataset cleanup failed: {str(e)}")

def get_dataset_statistics(dataset_base_path: str) -> Dict:
    """Get statistics about dataset usage."""
    try:
        dataset_custom_base = os.path.join(dataset_base_path, 'dataset_custom')
        if not os.path.exists(dataset_custom_base):
            return {"error": "No dataset_custom directory found"}
        
        stats = {
            "total_groups": 0,
            "total_size_gb": 0,
            "groups": {}
        }
        
        for group_dir in os.listdir(dataset_custom_base):
            group_path = os.path.join(dataset_custom_base, group_dir)
            if os.path.isdir(group_path):
                stats["total_groups"] += 1
                
                size = 0
                for dirpath, dirnames, filenames in os.walk(group_path):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if os.path.exists(filepath):
                            size += os.path.getsize(filepath)
                
                stats["groups"][group_dir] = {
                    "size_gb": size / (1024**3),
                    "creation_time": datetime.fromtimestamp(os.path.getctime(group_path)).isoformat()
                }
                stats["total_size_gb"] += size / (1024**3)
        
        return stats
    except Exception as e:
        return {"error": f"Failed to get dataset statistics: {str(e)}"}

def check_existing_classes_in_model(model_path: str) -> List[str]:
    """Check what classes exist in an existing model."""
    try:
        if os.path.exists(model_path):
            model = YOLO(model_path)
            if hasattr(model.model, 'names') and model.model.names:
                return list(model.model.names.values())
            elif hasattr(model, 'names') and model.names:
                return list(model.names.values())
    except Exception as e:
        logger.warning(f"Could not load existing model classes: {e}")
    return []

def adjust_hyperparameters_for_incremental(existing_classes: List[str], new_classes: List[str], device=None) -> Dict:
    """FIXED hyperparameters to prevent NaN - much more conservative settings."""
    is_incremental = len(existing_classes) > 0
    total_classes = len(set(existing_classes + new_classes))
    is_gpu = device and device.type == 'cuda'
    
    logger.info(f"Training mode: {'Incremental' if is_incremental else 'New'}")
    logger.info(f"Device: {device}")
    logger.info(f"Existing classes: {len(existing_classes)}, New classes: {len(new_classes)}, Total: {total_classes}")
    
    if is_gpu:
        # GPU hyperparameters - ULTRA CONSERVATIVE to prevent NaN
        base_params = {
            "cos_lr": False,        # Disable cosine LR - can cause instability
            "momentum": 0.85,        # Reduced momentum for stability
            "warmup_epochs": 5.0,   # Longer warmup to prevent early instability
            "warmup_momentum": 0.6, # Lower warmup momentum
            "warmup_bias_lr": 0.001,# Much lower warmup bias LR
            "patience": 25,         # More patience
            "box": 7.0,             # Reduced box loss weight
            "cls": 1,             # Reduced cls loss weight
            "dfl": 1.5,             # Reduced dfl loss weight
            "label_smoothing": 0.0, # Disable label smoothing - can cause NaN
            "close_mosaic": 10,     # Close mosaic later
            "dropout": 0.0,         # Add dropout for regularization
            "weight_decay": 0.0005, # Lower weight decay
        }
        
        if is_incremental:
            gpu_params = {
                "lr0": 0.0001,      # Very conservative learning rate
                "lrf": 0.01,        # Conservative final LR ratio
            }
        else:
            gpu_params = {
                "lr0": 0.0005,      # Still conservative for new training
                "lrf": 0.01,        
            }
        
        base_params.update(gpu_params)
        
    else:
        # CPU hyperparameters - more aggressive but still stable
        base_params = {
            "cos_lr": True,
            "momentum": 0.9,        
            "warmup_epochs": 3.0,   
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "patience": 20,         
            "box": 7.5,
            "cls": 0.5,
            "dfl": 1.5,
            "label_smoothing": 0.05,  
            "close_mosaic": 8,        
            "dropout": 0.0,
            "weight_decay": 0.0003,
        }
        
        if is_incremental:
            cpu_params = {
                "lr0": 0.003,       # Moderate learning rate for CPU
                "lrf": 0.05,        
            }
        else:
            cpu_params = {
                "lr0": 0.01,        
                "lrf": 0.05,        
            }
        
        base_params.update(cpu_params)
    
    return base_params



def check_for_nan_in_results(results_dict: Dict) -> bool:
    """Check if results contain NaN values."""
    nan_keys = []
    for key, value in results_dict.items():
        if isinstance(value, (int, float)):
            if math.isnan(value) or math.isinf(value):
                nan_keys.append(key)
    
    if nan_keys:
        logger.error(f"NaN/Inf detected in: {nan_keys}")
        return True
    return False

def train_model_group_based(piece_labels: list, db: Session, session_id: int):
    """Train models using group-based approach with enhanced GPU stability."""
    
    if isinstance(piece_labels, str):
        piece_labels = [piece_labels]
    
    try:
        training_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if not training_session:
            logger.error(f"Training session {session_id} not found")
            return

        piece_groups = group_pieces_by_prefix(piece_labels)
        logger.info(f"Grouped {len(piece_labels)} pieces into {len(piece_groups)} groups: {list(piece_groups.keys())}")
        
        training_session.add_log("INFO", f"Starting group-based training for {len(piece_groups)} groups: {list(piece_groups.keys())}")
        safe_commit(db)

        device = select_device()
        training_session.device_used = str(device)
        
        # Get image size based on device
        adaptive_image_size = get_training_image_size(device)
        training_session.image_size = adaptive_image_size
        training_session.add_log("INFO", f"Using device-specific image size: {adaptive_image_size}x{adaptive_image_size} for {device.type.upper()}")
        safe_commit(db)

        if not training_session.total_images:
            total_images = 0
            for piece_label in piece_labels:
                piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
                if piece:
                    images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).all()
                    total_images += len(images)

            training_session.total_images = total_images
            safe_commit(db)

        for group_idx, (group_name, group_pieces) in enumerate(piece_groups.items()):
            if stop_event.is_set():
                training_session.add_log("INFO", "Stop event detected. Ending training.")
                safe_commit(db)
                break
                
            logger.info(f"Training group: {group_name} with {len(group_pieces)} pieces")
            training_session.add_log("INFO", f"Training group: {group_name} with pieces: {group_pieces}")
            safe_commit(db)
            
            group_progress = (group_idx / len(piece_groups)) * 100
            training_session.progress_percentage = group_progress
            safe_commit(db)
            
            train_single_group(group_name, group_pieces, db, session_id, device)
            
            # Force cleanup between groups
            force_cleanup_gpu_memory()
            
        dataset_base_path = os.getenv('DATASET_BASE_PATH', '/app/shared/dataset')
        cleanup_old_datasets(dataset_base_path, keep_latest=3)
        
        stats = get_dataset_statistics(dataset_base_path)
        training_session.add_log("INFO", f"Dataset statistics: {stats}")
        safe_commit(db)
            
    except Exception as e:
        logger.error(f"An error occurred during group-based training: {str(e)}", exc_info=True)
        try:
            db.refresh(training_session)
            training_session.add_log("ERROR", f"Training failed: {str(e)}")
            safe_commit(db)
        except Exception as update_error:
            logger.error(f"Failed to update session with error: {str(update_error)}")
    finally:
        stop_event.clear()
        force_cleanup_gpu_memory()
        logger.info("Group-based training process finished and GPU memory cleared.")

def train_single_group(group_name: str, piece_labels: List[str], db: Session, session_id: int, device=None):
    """Train a single group with NaN prevention and stability improvements."""
    model = None
    try:
        training_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if not training_session:
            logger.error(f"Training session {session_id} not found")
            return

        # Use passed device or select new one
        if device is None:
            device = select_device()

        dataset_base_path = os.getenv('DATASET_BASE_PATH', '/app/shared/dataset')
        models_base_path = os.getenv('MODELS_BASE_PATH', '/app/shared/models')
        cpu_base_path = os.getenv('MODELS_BASE_PATH', '/app/shared/models/cpu')
        gpu_base_path = os.getenv('MODELS_BASE_PATH', '/app/shared/models/gpu')
        os.makedirs(models_base_path, exist_ok=True)
        
        model_save_path = os.path.join(models_base_path, f"model_{group_name}.pt")
        training_path_cpu = os.path.join(cpu_base_path,group_name)
        training_path_gpu = os.path.join(gpu_base_path,group_name)
        valid_pieces = []
        total_images = 0
        class_mapping = {}
        
        for piece_label in piece_labels:
            piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
            if not piece:
                training_session.add_log("WARNING", f"Piece '{piece_label}' not found in database")
                continue
                
            if not piece.is_annotated:
                training_session.add_log("WARNING", f"Piece '{piece_label}' is not annotated, skipping")
                continue
                
            images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).all()
            if not images:
                training_session.add_log("WARNING", f"No images found for piece '{piece_label}', skipping")
                continue
                
            valid_pieces.append(piece)
            total_images += len(images)
            class_mapping[piece_label] = len(class_mapping)
        
        if not valid_pieces:
            training_session.add_log("ERROR", f"No valid pieces found for group {group_name}")
            safe_commit(db)
            return
        
        if len(valid_pieces) < 2:
            training_session.add_log("WARNING", f"Only {len(valid_pieces)} piece(s) in group {group_name}. Consider adding more data for better training stability.")
            
        dataset_custom_path = os.path.join(dataset_base_path, 'dataset_custom', group_name)
        
        # Create directory structure
        os.makedirs(dataset_custom_path, exist_ok=True)
        
        # Create required subdirectories
        train_images_path = os.path.join(dataset_custom_path, "images", "train")
        valid_images_path = os.path.join(dataset_custom_path, "images", "valid")
        train_labels_path = os.path.join(dataset_custom_path, "labels", "train")
        valid_labels_path = os.path.join(dataset_custom_path, "labels", "valid")
        
        for path in [train_images_path, valid_images_path, train_labels_path, valid_labels_path]:
            os.makedirs(path, exist_ok=True)

        data_yaml_path = os.path.join(dataset_custom_path, "data.yaml")
        
        # CRITICAL FIX: Use integer indices for class mapping, not the piece labels
        class_names = {i: label for i, label in enumerate(class_mapping.keys())}
        
        data_yaml_content = {
            "path": dataset_custom_path,
            "train": "images/train",
            "val": "images/valid", 
            "names": class_names  # Use integer-indexed names
        }
        
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml_content, f, default_flow_style=False)
        
        training_session.add_log("INFO", f"Created data.yaml for group {group_name} with {len(class_names)} classes")
        safe_commit(db)

        # Rotation and augmentation
        logger.info(f"Starting rotation and augmentation for group {group_name}")
        training_session.add_log("INFO", f"Starting rotation and augmentation for group {group_name}")
        safe_commit(db)
        
        try:
            rotate_and_update_images(piece_labels, db)
            training_session.add_log("INFO", f"Completed rotation and augmentation for group {group_name}")
        except Exception as rotation_error:
            training_session.add_log("WARNING", f"Rotation failed for group {group_name}: {str(rotation_error)}")
        safe_commit(db)


        # Check existing model
        existing_classes = check_existing_classes_in_model(model_save_path)
        new_classes = list(class_mapping.keys())
        
        if os.path.exists(model_save_path):
            training_session.add_log("INFO", f"Loading existing model for group {group_name} (incremental learning)")
            try:
                model = YOLO(model_save_path)
            except Exception as e:
                training_session.add_log("WARNING", f"Failed to load existing model, creating new one: {e}")
                model = YOLO("yolov8n.pt")  # Use nano model for better stability
        else:
            training_session.add_log("INFO", f"Creating new model for group {group_name}")
            model = YOLO("yolov8n.pt")  # Use nano model for better stability

        model.to(device)
        
        # Force memory cleanup before training
        force_cleanup_gpu_memory()
        
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated()/1024**3
            cached = torch.cuda.memory_reserved()/1024**3
            training_session.add_log("INFO", f"GPU memory before training - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
            safe_commit(db)
        
        # Get device-specific configuration
        imgsz = get_training_image_size(device)
        batch_size = adjust_batch_size(device)
        hyperparameters = adjust_hyperparameters_for_incremental(existing_classes, new_classes, device)
        
        training_session.batch_size = batch_size
        training_session.image_size = imgsz
        training_session.add_log("INFO", f"Group {group_name} config - Image: {imgsz}, Batch: {batch_size}, Device: {device}")
        safe_commit(db)

        # CONSERVATIVE augmentation settings to prevent NaN
        augmentations = {
            "hsv_h": 0.01,       # Very small color variation
            "hsv_s": 0.3,        # Reduced saturation changes
            "hsv_v": 0.2,        # Reduced brightness variation
            "degrees": 10.0,     # Reduced rotation
            "translate": 0.05,   # Very small translation
            "scale": 0.2,        # Reduced scale variation
            "shear": 0.0,        # Disable shear - can cause instability
            "perspective": 0.0,  # Disable perspective
            "flipud": 0.0,       # No vertical flip
            "fliplr": 0.3,       # Reduced horizontal flip probability
            "mosaic": 0.3,       # Reduced mosaic - can cause NaN with small datasets
            "mixup": 0.0,        # Disable mixup - can cause instability
            "copy_paste": 0.0,   
            "erasing": 0.0,
        }
        
        hyperparameters.update(augmentations)
        
        # Validate model with dummy input
        try:
            test_input = torch.randn(1, 3, imgsz, imgsz).to(device)
            with torch.no_grad():
                test_output = model.model(test_input)
                
            # Check for NaN in test output
            if any(torch.isnan(tensor).any() for tensor in test_output if torch.is_tensor(tensor)):
                training_session.add_log("ERROR", "Model produces NaN on test input")
                safe_commit(db)
                return
                
            del test_input, test_output
            force_cleanup_gpu_memory()
            training_session.add_log("INFO", f"Model validation passed for group {group_name}")
            safe_commit(db)
        except Exception as e:
            training_session.add_log("ERROR", f"Model validation failed: {e}")
            safe_commit(db)
            return
        
        training_session.add_log("INFO", f"Starting training for group {group_name} with {training_session.epochs} epochs")
        training_session.add_log("INFO", f"Training hyperparameters: {hyperparameters}")
        safe_commit(db)
        
        # Disable AMP completely for stability
        use_amp = False
        
        if device.type == 'cpu':
            # CPU: Train with fewer epochs first to test stability
            test_epochs = min(5, training_session.epochs)
            try:
                training_session.add_log("INFO", f"CPU Training: Starting with {test_epochs} test epochs")
                safe_commit(db)
                
                results = model.train(
                    data=data_yaml_path,
                    epochs=test_epochs,
                    imgsz=imgsz,
                    batch=batch_size,
                    device=device,
                    project=os.path.dirname(training_path_cpu),
                    name=f"{group_name}_test_training",
                    exist_ok=True,
                    amp=use_amp,
                    plots=True,
                    resume=False,
                    augment=True,
                    verbose=True,
                    workers=1,  # Single worker to prevent data loading issues
                    **hyperparameters
                )
                
                # Check results for NaN
                if hasattr(results, 'results_dict'):
                    if check_for_nan_in_results(results.results_dict):
                        training_session.add_log("ERROR", "NaN detected in test training, aborting")
                        safe_commit(db)
                        return
                
                # If test training successful, continue with full training
                if training_session.epochs > test_epochs:
                    remaining_epochs = training_session.epochs - test_epochs
                    training_session.add_log("INFO", f"Test training successful, continuing with {remaining_epochs} more epochs")
                    
                    results = model.train(
                        data=data_yaml_path,
                        epochs=remaining_epochs,
                        imgsz=imgsz,
                        batch=batch_size,
                        device=device,
                        project=os.path.dirname(training_path_cpu),
                        name=f"{group_name}_full_training",
                        exist_ok=True,
                        amp=use_amp,
                        plots=True,
                        resume=False,
                        augment=True,
                        verbose=True,
                        workers=1,
                        **hyperparameters
                    )
                
                # Update progress
                training_session.current_epoch = training_session.epochs
                training_session.progress_percentage = 100.0
                
                # Extract final results
                if hasattr(results, 'results_dict'):
                    results_dict = results.results_dict
                    
                    # Final NaN check
                    if check_for_nan_in_results(results_dict):
                        training_session.add_log("ERROR", "NaN detected in final results")
                        safe_commit(db)
                        return
                    
                    if 'train/box_loss' in results_dict:
                        training_session.update_losses(
                            box_loss=results_dict.get('train/box_loss'),
                            cls_loss=results_dict.get('train/cls_loss'),
                            dfl_loss=results_dict.get('train/dfl_loss')
                        )
                    
                    if 'lr/pg0' in results_dict:
                        training_session.update_metrics(
                            lr=results_dict.get('lr/pg0'),
                            momentum=hyperparameters.get('momentum', 0.9)
                        )
                safe_commit(db)
                
            except Exception as train_error:
                training_session.add_log("ERROR", f"CPU Training failed: {str(train_error)}")
                safe_commit(db)
                return
                
        else:
            # GPU: Train epoch by epoch with extensive NaN checking
            consecutive_nan_count = 0
            max_consecutive_nans = 3
            
            for epoch in range(1, training_session.epochs + 1):
                if stop_event.is_set():
                    training_session.add_log("INFO", "Stop event detected during group training")
                    safe_commit(db)
                    break
                
                training_session.current_epoch = epoch
                epoch_progress = (epoch / training_session.epochs) * 100
                training_session.progress_percentage = epoch_progress
                
                training_session.add_log("INFO", f"GPU Training - Epoch {epoch}/{training_session.epochs} (Progress: {epoch_progress:.1f}%)")
                safe_commit(db)
                
                try:
                    # Aggressive memory cleanup before each epoch
                    force_cleanup_gpu_memory()
                    
                    # Check GPU memory
                    if device.type == 'cuda':
                        allocated = torch.cuda.memory_allocated() / 1024**3
                        if allocated > 2.0:  # If using more than 2GB
                            training_session.add_log("WARNING", f"High GPU memory usage: {allocated:.2f}GB")
                            force_cleanup_gpu_memory()
                    
                    # Train single epoch with gradient clipping
                    results = model.train(
                        data=data_yaml_path,
                        epochs=1,
                        imgsz=imgsz,
                        batch=batch_size,
                        device=device,
                        project=os.path.dirname(training_path_gpu),
                        name=f"{group_name}_epoch_{epoch}",
                        exist_ok=True,
                        amp=use_amp,
                        plots=False,  # Disable plots to save memory
                        resume=False,
                        augment=False,  # Disable augment for stability
                        verbose=False,
                        workers=1,  # Single worker for GPU
                        **hyperparameters
                    )
                    
                    force_cleanup_gpu_memory()
                    
                    # Process results with extensive NaN checking
                    if hasattr(results, 'results_dict'):
                        results_dict = results.results_dict
                        
                        # Check for NaN values
                        if check_for_nan_in_results(results_dict):
                            consecutive_nan_count += 1
                            training_session.add_log("ERROR", f"NaN detected in epoch {epoch} (consecutive: {consecutive_nan_count})")
                            
                            if consecutive_nan_count >= max_consecutive_nans:
                                training_session.add_log("ERROR", f"Too many consecutive NaN epochs ({consecutive_nan_count}), stopping training")
                                safe_commit(db)
                                break
                            else:
                                # Try to continue with next epoch
                                safe_commit(db)
                                continue
                        else:
                            # Reset consecutive NaN count on successful epoch
                            consecutive_nan_count = 0
                        
                        # Update losses if valid
                        if 'train/box_loss' in results_dict:
                            box_loss = results_dict.get('train/box_loss')
                            cls_loss = results_dict.get('train/cls_loss')
                            dfl_loss = results_dict.get('train/dfl_loss')
                            
                            # Additional validation of loss values
                            if all(isinstance(loss, (int, float)) and not math.isnan(loss) and not math.isinf(loss) 
                                   for loss in [box_loss, cls_loss, dfl_loss] if loss is not None):
                                training_session.update_losses(
                                    box_loss=box_loss,
                                    cls_loss=cls_loss,
                                    dfl_loss=dfl_loss
                                )
                        
                        if 'lr/pg0' in results_dict:
                            lr = results_dict.get('lr/pg0')
                            if isinstance(lr, (int, float)) and not math.isnan(lr) and not math.isinf(lr):
                                training_session.update_metrics(
                                    lr=lr,
                                    momentum=hyperparameters.get('momentum', 0.7)
                                )
                        safe_commit(db)
                    
                except Exception as train_error:
                    error_str = str(train_error).lower()
                    training_session.add_log("ERROR", f"Training error in epoch {epoch}: {str(train_error)}")
                    safe_commit(db)
                    
                    # Handle specific error types
                    if any(keyword in error_str for keyword in ["out of memory", "cuda"]):
                        training_session.add_log("ERROR", "GPU memory error. Stopping training.")
                        safe_commit(db)
                        break
                    elif "nan" in error_str or "inf" in error_str:
                        consecutive_nan_count += 1
                        training_session.add_log("ERROR", f"NaN/Inf error in epoch {epoch} (consecutive: {consecutive_nan_count})")
                        if consecutive_nan_count >= max_consecutive_nans:
                            training_session.add_log("ERROR", "Too many NaN errors, stopping training")
                            safe_commit(db)
                            break
                    else:
                        continue
                
                # Save checkpoint every 10 epochs or if approaching end
                if epoch % 10 == 0 or epoch >= training_session.epochs - 2:
                    try:
                        model.save(model_save_path)
                        training_session.add_log("INFO", f"Checkpoint saved after epoch {epoch}")
                        safe_commit(db)
                    except Exception as save_error:
                        training_session.add_log("WARNING", f"Failed to save checkpoint: {str(save_error)}")
                        safe_commit(db)

        # Final model save with validation
        try:
            model.save(model_save_path)
            training_session.add_log("INFO", f"Training completed for group {group_name}. Model saved to {model_save_path}")
            
            # Verify model can be loaded and doesn't produce NaN
            test_model = YOLO(model_save_path)
            class_names = test_model.names
            
            # Test model with dummy input
            test_input = torch.randn(1, 3, imgsz, imgsz).to(device)
            with torch.no_grad():
                test_output = test_model.model(test_input)
                
            # Check for NaN in final model output
            if any(torch.isnan(tensor).any() for tensor in test_output if torch.is_tensor(tensor)):
                training_session.add_log("ERROR", "Final model produces NaN output!")
                safe_commit(db)
            else:
                training_session.add_log("INFO", f"Model verification successful. Classes: {class_names}")
            
            del test_model, test_input, test_output
            
        except Exception as save_error:
            training_session.add_log("ERROR", f"Failed to save final model: {str(save_error)}")
        
        # Mark pieces as trained
        for piece in valid_pieces:
            piece.is_yolo_trained = True
        safe_commit(db)
        
        training_session.add_log("INFO", f"Updated training status for all pieces in group {group_name}")
        safe_commit(db)

    except Exception as e:
        error_msg = f"Error during group {group_name} training: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        training_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if training_session:
            training_session.add_log("ERROR", error_msg)
            safe_commit(db)
    
    finally:
        # Cleanup model from memory
        if model:
            del model
        force_cleanup_gpu_memory()
        logger.info(f"Training completed for group {group_name}")

# Backward compatibility functions
def train_model(piece_labels: list, db: Session, session_id: int):
    """Backward compatibility wrapper - redirects to group-based training."""
    return train_model_group_based(piece_labels, db, session_id)

def train_single_piece(piece_label: str, db: Session, service_dir: str, session_id: int, is_resumed: bool = False):
    """Backward compatibility wrapper - redirects single piece to group-based training."""
    device = select_device()
    return train_single_group(
        group_name=extract_group_from_piece_label(piece_label),
        piece_labels=[piece_label],
        db=db,
        session_id=session_id,
        device=device
    )