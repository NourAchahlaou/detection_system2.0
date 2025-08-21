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

# FIXED: More conservative image size for GPU
DETECTION_IMAGE_SIZE = 416  # Reduced from 640 for better GPU compatibility

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
            torch.cuda.set_per_process_memory_fraction(0.7)  # Use only 70% of GPU memory
            logger.info("Set GPU memory fraction to 0.7 for small GPU")
        
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
    """Ultra-conservative batch sizes for GPU stability."""
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(0).total_memory
        if total_memory <= 4 * 1024**3:  # T600 4GB
            return 1  # Single batch for maximum stability
        elif total_memory <= 6 * 1024**3:
            return 2
        else:
            return 4
    else:
        return max(2, base_batch_size // 2)

def get_training_image_size():
    """Return conservative image size for GPU training."""
    return DETECTION_IMAGE_SIZE

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
    """Ultra-conservative hyperparameters for GPU stability."""
    is_incremental = len(existing_classes) > 0
    total_classes = len(set(existing_classes + new_classes))
    is_gpu = device and device.type == 'cuda'
    
    logger.info(f"Training mode: {'Incremental' if is_incremental else 'New'}")
    logger.info(f"Device: {device}")
    logger.info(f"Existing classes: {len(existing_classes)}, New classes: {len(new_classes)}, Total: {total_classes}")
    
    # Ultra-conservative base parameters for GPU
    base_params = {
        "cos_lr": False,
        "momentum": 0.8,  # Reduced momentum
        "warmup_epochs": 3.0,  # Shorter warmup
        "warmup_momentum": 0.7,
        "warmup_bias_lr": 0.01,
        "patience": 10,  # Shorter patience
        "box": 7.5,
        "cls": 0.5,
        "dfl": 1.5,
        "label_smoothing": 0.0,
        "close_mosaic": 5,  # Close mosaic earlier
        "dropout": 0.0,
    }
    
    if is_gpu:
        # Ultra-conservative GPU parameters
        if is_incremental:
            gpu_params = {
                "lr0": 0.000005,     # Very low learning rate
                "lrf": 0.0001,       # Very low final learning rate
                "weight_decay": 0.0001,
            }
        else:
            gpu_params = {
                "lr0": 0.00001,      # Very low learning rate
                "lrf": 0.001,        
                "weight_decay": 0.0002,
            }
    else:
        # CPU parameters
        if is_incremental:
            gpu_params = {
                "lr0": 0.00005,
                "lrf": 0.005,
                "weight_decay": 0.0008,
            }
        else:
            gpu_params = {
                "lr0": 0.0001,
                "lrf": 0.01,
                "weight_decay": 0.0005,
            }
    
    base_params.update(gpu_params)
    return base_params

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
        
        fixed_image_size = get_training_image_size()
        training_session.image_size = fixed_image_size
        training_session.add_log("INFO", f"Using conservative image size: {fixed_image_size}x{fixed_image_size} for GPU stability")
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
            
            train_single_group(group_name, group_pieces, db, session_id)
            
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

def train_single_group(group_name: str, piece_labels: List[str], db: Session, session_id: int):
    """Train a single group with maximum GPU stability measures."""
    model = None
    try:
        training_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if not training_session:
            logger.error(f"Training session {session_id} not found")
            return

        dataset_base_path = os.getenv('DATASET_BASE_PATH', '/app/shared/dataset')
        models_base_path = os.getenv('MODELS_BASE_PATH', '/app/shared/models')
        
        os.makedirs(models_base_path, exist_ok=True)
        
        model_save_path = os.path.join(models_base_path, f"model_{group_name}.pt")
        
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
            
        dataset_custom_path = os.path.join(dataset_base_path, 'dataset_custom', group_name)

        data_yaml_path = os.path.join(dataset_custom_path, "data.yaml")
        
        data_yaml_content = {
            "path": dataset_custom_path,
            "train": "images/train",
            "val": "images/valid", 
            "names": class_mapping
        }
        
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml_content, f, default_flow_style=False)
        
        training_session.add_log("INFO", f"Created data.yaml for group {group_name} with content: {data_yaml_content}")
        safe_commit(db)

        logger.info(f"Starting rotation and augmentation for group {group_name}")
        training_session.add_log("INFO", f"Starting rotation and augmentation for group {group_name}")
        safe_commit(db)
        
        try:
            rotate_and_update_images(piece_labels, db)
            training_session.add_log("INFO", f"Completed rotation and augmentation for group {group_name}")
        except Exception as rotation_error:
            training_session.add_log("WARNING", f"Rotation failed for group {group_name}: {str(rotation_error)}")
        safe_commit(db)

        existing_classes = check_existing_classes_in_model(model_save_path)
        new_classes = list(class_mapping.keys())
        
        device = select_device()
        
        if os.path.exists(model_save_path):
            training_session.add_log("INFO", f"Loading existing model for group {group_name} (incremental learning)")
            try:
                model = YOLO(model_save_path)
            except Exception as e:
                training_session.add_log("WARNING", f"Failed to load existing model, creating new one: {e}")
                model = YOLO("yolov8n.pt")  # Use nano model for GPU
        else:
            training_session.add_log("INFO", f"Creating new model for group {group_name}")
            model = YOLO("yolov8n.pt")  # Use nano model for better GPU compatibility

        model.to(device)
        
        # Force memory cleanup before training
        force_cleanup_gpu_memory()
        
        if device.type == 'cuda':
            allocated = torch.cuda.memory_allocated()/1024**3
            cached = torch.cuda.memory_reserved()/1024**3
            training_session.add_log("INFO", f"GPU memory before training - Allocated: {allocated:.2f}GB, Cached: {cached:.2f}GB")
            safe_commit(db)
        
        imgsz = get_training_image_size()
        batch_size = adjust_batch_size(device)
        hyperparameters = adjust_hyperparameters_for_incremental(existing_classes, new_classes, device)
        
        training_session.batch_size = batch_size
        training_session.image_size = imgsz
        training_session.add_log("INFO", f"Group {group_name} config - Image: {imgsz}, Batch: {batch_size}, Device: {device}")
        safe_commit(db)

        # Ultra-conservative augmentation for GPU
        augmentations = {
            "hsv_h": 0.005,      # Minimal augmentation
            "hsv_s": 0.2,        
            "hsv_v": 0.2,        
            "degrees": 5.0,      # Minimal rotation
            "translate": 0.05,   # Minimal translation
            "scale": 0.1,        # Minimal scaling
            "shear": 0.0,        
            "perspective": 0.0,  
            "flipud": 0.0,       
            "fliplr": 0.3,       # Reduced flip probability
            "mosaic": 0.0,       # Disable mosaic for stability
            "mixup": 0.0,        # Disable mixup
            "copy_paste": 0.0,   
            "erasing": 0.0,      # Disable erasing
        }
        
        hyperparameters.update(augmentations)
        
        # Validate model with minimal memory usage
        try:
            test_input = torch.randn(1, 3, imgsz, imgsz).to(device)
            with torch.no_grad():
                _ = model.model(test_input)
            del test_input
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
        
        # Disable AMP completely for GPU to prevent NaN
        use_amp = False
        
        # Modified training loop with enhanced stability
        for epoch in range(1, training_session.epochs + 1):
            if stop_event.is_set():
                training_session.add_log("INFO", "Stop event detected during group training")
                safe_commit(db)
                break
            
            training_session.current_epoch = epoch
            epoch_progress = (epoch / training_session.epochs) * 100
            training_session.progress_percentage = epoch_progress
            
            training_session.add_log("INFO", f"Group {group_name} - Epoch {epoch}/{training_session.epochs} (Progress: {epoch_progress:.1f}%)")
            safe_commit(db)
            
            try:
                # Aggressive memory cleanup before each epoch
                force_cleanup_gpu_memory()
                
                # Check GPU memory before training
                if device.type == 'cuda':
                    allocated = torch.cuda.memory_allocated() / 1024**3
                    if allocated > 2.0:  # If using more than 2GB, something's wrong
                        training_session.add_log("WARNING", f"High GPU memory usage before epoch: {allocated:.2f}GB")
                        force_cleanup_gpu_memory()
                
                # Train with minimal settings
                results = model.train(
                    data=data_yaml_path,
                    epochs=1,
                    imgsz=imgsz,
                    batch=batch_size,
                    device=device,
                    project=os.path.dirname(model_save_path),
                    name=f"{group_name}_epoch_{epoch}",
                    exist_ok=True,
                    amp=use_amp,
                    plots=False,  # Disable plots to save memory
                    resume=False,
                    augment=True,
                    verbose=False,  # Reduce verbose output
                    workers=1,  # Single worker to reduce memory usage
                    **hyperparameters
                )
                
                # Immediate cleanup after training step
                force_cleanup_gpu_memory()
                
                # Validate results and check for NaN
                if hasattr(results, 'results_dict'):
                    results_dict = results.results_dict
                    
                    has_nan = False
                    nan_keys = []
                    for key, value in results_dict.items():
                        if isinstance(value, (int, float)) and (value != value or value == float('inf') or value == float('-inf')):
                            has_nan = True
                            nan_keys.append(key)
                    
                    if has_nan:
                        training_session.add_log("ERROR", f"NaN detected in epoch {epoch} for keys: {nan_keys}")
                        safe_commit(db)
                        break
                    
                    # Update losses if valid
                    if 'train/box_loss' in results_dict and not has_nan:
                        training_session.update_losses(
                            box_loss=results_dict.get('train/box_loss'),
                            cls_loss=results_dict.get('train/cls_loss'),
                            dfl_loss=results_dict.get('train/dfl_loss')
                        )
                    
                    if 'lr/pg0' in results_dict:
                        training_session.update_metrics(
                            lr=results_dict.get('lr/pg0'),
                            momentum=hyperparameters.get('momentum', 0.8)
                        )
                    safe_commit(db)
                
            except Exception as train_error:
                error_str = str(train_error).lower()
                training_session.add_log("ERROR", f"Training error in epoch {epoch}: {str(train_error)}")
                safe_commit(db)
                
                # Handle specific GPU errors
                if any(keyword in error_str for keyword in ["out of memory", "cuda", "nan", "memory"]):
                    training_session.add_log("ERROR", "GPU memory or NaN error detected. Stopping training.")
                    safe_commit(db)
                    break
                else:
                    # For other errors, try to continue
                    continue
            
            # Skip validation to save memory - only save model periodically
            if epoch % 5 == 0:
                try:
                    model.save(model_save_path)
                    training_session.add_log("INFO", f"Checkpoint saved for group {group_name} after epoch {epoch}")
                    safe_commit(db)
                except Exception as save_error:
                    training_session.add_log("WARNING", f"Failed to save checkpoint: {str(save_error)}")
                    safe_commit(db)

        # Final model save
        try:
            model.save(model_save_path)
            training_session.add_log("INFO", f"Training completed for group {group_name}. Final model saved to {model_save_path}")
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
    return train_single_group(
        group_name=extract_group_from_piece_label(piece_label),
        piece_labels=[piece_label],
        db=db,
        session_id=session_id
    )