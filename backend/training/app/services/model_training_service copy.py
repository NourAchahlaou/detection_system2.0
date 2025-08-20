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

# FIXED: Consistent image size matching detection system
DETECTION_IMAGE_SIZE = 640  # Fixed size to match detection system

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
    """Select the best available device (GPU if available, else CPU) with detailed diagnostics."""
    
    # Check CUDA availability
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA version: {torch.version.cuda}")
    
    # Check for NVIDIA environment variables
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
        
        # Use first available GPU
        device = torch.device('cuda:0')
        logger.info(f"Selected device: {device}")
        
        # Test GPU functionality
        try:
            test_tensor = torch.randn(10, 10).to(device)
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
    """Adjust batch size based on the available device and fixed image size."""
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(0).total_memory
        # Adjust batch size based on memory, considering fixed 640x640 image size
        if total_memory > 12 * 1024**3:  # If GPU has more than 12GB of memory
            return base_batch_size * 2  # 16
        elif total_memory > 8 * 1024**3:  # If GPU has more than 8GB of memory
            return base_batch_size  # 8
        elif total_memory > 6 * 1024**3:  # If GPU has more than 6GB of memory
            return max(4, base_batch_size // 2)  # 4
        else:
            return max(2, base_batch_size // 4)  # 2 minimum
    else:
        return max(1, base_batch_size // 4)  # CPU with smaller batch

def get_training_image_size():
    """Return fixed image size for training to match detection system."""
    return DETECTION_IMAGE_SIZE

def cleanup_old_datasets(dataset_base_path: str, keep_latest: int = 2):
    """Clean up old dataset_custom directories to save space."""
    try:
        dataset_custom_base = os.path.join(dataset_base_path, 'dataset_custom')
        if not os.path.exists(dataset_custom_base):
            return
        
        # Get all group directories with their creation times
        group_dirs = []
        for item in os.listdir(dataset_custom_base):
            item_path = os.path.join(dataset_custom_base, item)
            if os.path.isdir(item_path):
                creation_time = os.path.getctime(item_path)
                group_dirs.append((item, creation_time, item_path))
        
        # Sort by creation time (newest first)
        group_dirs.sort(key=lambda x: x[1], reverse=True)
        
        # Remove old directories, keeping the latest ones
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
                
                # Calculate size
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

def adjust_hyperparameters_for_incremental(existing_classes: List[str], new_classes: List[str]) -> Dict:
    """Adjust hyperparameters based on whether this is incremental learning."""
    is_incremental = len(existing_classes) > 0
    total_classes = len(set(existing_classes + new_classes))
    
    logger.info(f"Training mode: {'Incremental' if is_incremental else 'New'}")
    logger.info(f"Existing classes: {len(existing_classes)}, New classes: {len(new_classes)}, Total: {total_classes}")
    
    if is_incremental:
        # More conservative hyperparameters for incremental learning
        return {
            "cos_lr": False,
            "lr0": 0.00005,  # Lower learning rate to preserve existing knowledge
            "lrf": 0.005,    # Lower final learning rate
            "momentum": 0.9,
            "weight_decay": 0.0008,  # Slightly higher weight decay
            "dropout": 0.15,         # Moderate dropout
            "warmup_epochs": 3.0,    # Shorter warmup
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.05,  # Lower bias learning rate
            "label_smoothing": 0.05, # Lower label smoothing
            "patience": 15,          # More patience for convergence
        }
    else:
        # Standard hyperparameters for new training
        return {
            "cos_lr": False,
            "lr0": 0.0001,
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "dropout": 0.2,
            "warmup_epochs": 5.0,
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "label_smoothing": 0.1,
            "patience": 10,
        }

def train_model_group_based(piece_labels: list, db: Session, session_id: int):
    """Train models using group-based approach with incremental learning."""
    
    # Ensure piece_labels is a list
    if isinstance(piece_labels, str):
        piece_labels = [piece_labels]
    
    try:
        # Get training session
        training_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if not training_session:
            logger.error(f"Training session {session_id} not found")
            return

        # Group pieces by their prefix
        piece_groups = group_pieces_by_prefix(piece_labels)
        logger.info(f"Grouped {len(piece_labels)} pieces into {len(piece_groups)} groups: {list(piece_groups.keys())}")
        
        training_session.add_log("INFO", f"Starting group-based training for {len(piece_groups)} groups: {list(piece_groups.keys())}")
        safe_commit(db)

        # Select device and update session
        device = select_device()
        training_session.device_used = str(device)
        
        # Set consistent image size
        fixed_image_size = get_training_image_size()
        training_session.image_size = fixed_image_size
        training_session.add_log("INFO", f"Using fixed image size: {fixed_image_size}x{fixed_image_size} to match detection system")
        safe_commit(db)

        # Count total images across all pieces
        if not training_session.total_images:
            total_images = 0
            for piece_label in piece_labels:
                piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
                if piece:
                    images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).all()
                    total_images += len(images)

            training_session.total_images = total_images
            safe_commit(db)

        # Train each group
        for group_idx, (group_name, group_pieces) in enumerate(piece_groups.items()):
            if stop_event.is_set():
                training_session.add_log("INFO", "Stop event detected. Ending training.")
                safe_commit(db)
                break
                
            logger.info(f"Training group: {group_name} with {len(group_pieces)} pieces")
            training_session.add_log("INFO", f"Training group: {group_name} with pieces: {group_pieces}")
            safe_commit(db)
            
            # Update progress based on group completion
            group_progress = (group_idx / len(piece_groups)) * 100
            training_session.progress_percentage = group_progress
            safe_commit(db)
            
            # Train the group
            train_single_group(group_name, group_pieces, db, session_id)
            
        # Cleanup old datasets after training
        dataset_base_path = os.getenv('DATASET_BASE_PATH', '/app/shared/dataset')
        cleanup_old_datasets(dataset_base_path, keep_latest=3)
        
        # Log dataset statistics
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
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Group-based training process finished and GPU memory cleared.")

def train_single_group(group_name: str, piece_labels: List[str], db: Session, session_id: int):
    """Train a single group of pieces together."""
    model = None
    try:
        # Get training session
        training_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if not training_session:
            logger.error(f"Training session {session_id} not found")
            return

        # Get dataset and models paths
        dataset_base_path = os.getenv('DATASET_BASE_PATH', '/app/shared/dataset')
        models_base_path = os.getenv('MODELS_BASE_PATH', '/app/shared/models')
        
        # Ensure models directory exists
        os.makedirs(models_base_path, exist_ok=True)
        
        # Group-specific model path
        model_save_path = os.path.join(models_base_path, f"model_{group_name}.pt")
        
        # Validate pieces and collect data
        valid_pieces = []
        total_images = 0
        class_mapping = {}  # Map piece labels to class IDs
        
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
            class_mapping[piece_label] = len(class_mapping)  # Assign incremental class IDs
        
        if not valid_pieces:
            training_session.add_log("ERROR", f"No valid pieces found for group {group_name}")
            safe_commit(db)
            return
            
        training_session.add_log("INFO", f"Group {group_name}: Processing {len(valid_pieces)} valid pieces with {total_images} total images")
        training_session.add_log("INFO", f"Class mapping for group {group_name}: {class_mapping}")
        safe_commit(db)

        # Create group-based dataset structure
        dataset_custom_path = os.path.join(dataset_base_path, 'dataset_custom', group_name)
        
        # Directory paths
        image_dir_train = os.path.join(dataset_custom_path, "images", "train")
        image_dir_val = os.path.join(dataset_custom_path, "images", "valid")
        label_dir_train = os.path.join(dataset_custom_path, "labels", "train")
        label_dir_val = os.path.join(dataset_custom_path, "labels", "valid")
        
        # Create all necessary directories
        for dir_path in [image_dir_train, image_dir_val, label_dir_train, label_dir_val]:
            os.makedirs(dir_path, exist_ok=True)

        training_session.add_log("INFO", f"Created dataset directories for group: {group_name}")
        safe_commit(db)

        # Copy images and create annotations for all pieces in the group
        for piece in valid_pieces:
            piece_label = piece.piece_label
            images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).all()
            
            # Split images for train/validation (80/20 split)
            train_split = int(len(images) * 0.8)
            train_images = images[:train_split]
            val_images = images[train_split:]
            
            # Process training images
            for image in train_images:
                # Copy image
                dest_path = os.path.join(image_dir_train, os.path.basename(image.image_path))
                shutil.copy(image.image_path, dest_path)
                
                # Create annotation
                annotations = db.query(Annotation).filter(Annotation.piece_image_id == image.id).all()
                if annotations:
                    label_path = os.path.join(label_dir_train, os.path.basename(image.image_path).replace('.jpg', '.txt').replace('.png', '.txt'))
                    with open(label_path, "w") as label_file:
                        for annotation in annotations:
                            class_id = class_mapping[piece_label]
                            label_file.write(f"{class_id} {annotation.x} {annotation.y} {annotation.width} {annotation.height}\n")
            
            # Process validation images
            for image in val_images:
                # Copy image
                dest_path = os.path.join(image_dir_val, os.path.basename(image.image_path))
                shutil.copy(image.image_path, dest_path)
                
                # Create annotation
                annotations = db.query(Annotation).filter(Annotation.piece_image_id == image.id).all()
                if annotations:
                    label_path = os.path.join(label_dir_val, os.path.basename(image.image_path).replace('.jpg', '.txt').replace('.png', '.txt'))
                    with open(label_path, "w") as label_file:
                        for annotation in annotations:
                            class_id = class_mapping[piece_label]
                            label_file.write(f"{class_id} {annotation.x} {annotation.y} {annotation.width} {annotation.height}\n")

        training_session.add_log("INFO", f"Copied images and annotations for {len(valid_pieces)} pieces in group {group_name}")
        safe_commit(db)

        # Create data.yaml for this group
        data_yaml_path = os.path.join(dataset_custom_path, "data.yaml")
        
        data_yaml_content = {
            'train': image_dir_train,
            'val': image_dir_val,
            'nc': len(class_mapping),
            'names': list(class_mapping.keys())
        }
        
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml_content, f)
        
        training_session.add_log("INFO", f"Created data.yaml for group {group_name} with {len(class_mapping)} classes")
        safe_commit(db)

        # Perform rotation and augmentation for the entire group
        logger.info(f"Starting rotation and augmentation for group {group_name}")
        training_session.add_log("INFO", f"Starting rotation and augmentation for group {group_name}")
        safe_commit(db)
        
        try:
            rotate_and_update_images(piece_labels, db)
            training_session.add_log("INFO", f"Completed rotation and augmentation for group {group_name}")
        except Exception as rotation_error:
            training_session.add_log("WARNING", f"Rotation failed for group {group_name}: {str(rotation_error)}")
        safe_commit(db)

        # Check for existing model and classes for incremental learning
        existing_classes = check_existing_classes_in_model(model_save_path)
        new_classes = list(class_mapping.keys())
        
        # Initialize model
        device = select_device()
        
        if os.path.exists(model_save_path):
            training_session.add_log("INFO", f"Loading existing model for group {group_name} (incremental learning)")
            model = YOLO(model_save_path)
        else:
            training_session.add_log("INFO", f"Creating new model for group {group_name}")
            base_model_path = os.path.join(models_base_path, "yolov8m.pt")
            if os.path.exists(base_model_path):
                model = YOLO(base_model_path)
            else:
                model = YOLO("yolov8m.pt")  # Download if not exists

        model.to(device)
        
        # Get optimized training parameters
        imgsz = get_training_image_size()
        batch_size = adjust_batch_size(device)
        hyperparameters = adjust_hyperparameters_for_incremental(existing_classes, new_classes)
        
        # Update training session
        training_session.batch_size = batch_size
        training_session.image_size = imgsz
        training_session.add_log("INFO", f"Group {group_name} training config - Image size: {imgsz}, Batch: {batch_size}, Device: {device}")
        safe_commit(db)

        # Augmentation parameters optimized for group training
        augmentations = {
            "hsv_h": 0.015,  
            "hsv_s": 0.7,  
            "hsv_v": 0.4,  
            "degrees": 15.0,
            "translate": 0.15,
            "scale": 0.2,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 0.8,
            "mixup": 0.15,
            "copy_paste": 0.1,
            "erasing": 0.4,
            "crop_fraction": 1.0,
        }
        
        hyperparameters.update(augmentations)
        
        # Training loop for the group
        training_session.add_log("INFO", f"Starting training for group {group_name} with {training_session.epochs} epochs")
        safe_commit(db)
        
        for epoch in range(1, training_session.epochs + 1):
            if stop_event.is_set():
                training_session.add_log("INFO", "Stop event detected during group training")
                safe_commit(db)
                break
            
            # Update progress
            training_session.current_epoch = epoch
            epoch_progress = (epoch / training_session.epochs) * 100
            training_session.progress_percentage = epoch_progress
            
            training_session.add_log("INFO", f"Group {group_name} - Epoch {epoch}/{training_session.epochs} (Progress: {epoch_progress:.1f}%)")
            safe_commit(db)
            
            # Train for one epoch
            results = model.train(
                data=data_yaml_path,
                epochs=1,
                imgsz=imgsz,
                batch=batch_size,
                device=device,
                project=os.path.dirname(model_save_path),
                name=f"{group_name}_epoch_{epoch}",
                exist_ok=True,
                amp=True,
                plots=False,
                resume=False,  # We handle resume logic manually
                augment=True,
                **hyperparameters
            )
            
            # Update losses if available
            if hasattr(results, 'results_dict'):
                results_dict = results.results_dict
                if 'train/box_loss' in results_dict:
                    training_session.update_losses(
                        box_loss=results_dict.get('train/box_loss'),
                        cls_loss=results_dict.get('train/cls_loss'),
                        dfl_loss=results_dict.get('train/dfl_loss')
                    )
                
                if 'lr/pg0' in results_dict:
                    training_session.update_metrics(
                        lr=results_dict.get('lr/pg0'),
                        momentum=hyperparameters.get('momentum', 0.937)
                    )
                safe_commit(db)
            
            # Validation every 5 epochs
            if epoch % 5 == 0:
                training_session.add_log("INFO", f"Running validation for group {group_name} after epoch {epoch}")
                safe_commit(db)
                
                validation_results = model.val(
                    data=data_yaml_path,
                    imgsz=imgsz,
                    batch=batch_size,
                    device=device,
                    plots=False
                )
                
                training_session.add_log("INFO", f"Validation completed for group {group_name} after epoch {epoch}")
                safe_commit(db)
            
            # Save checkpoint
            if epoch % 2 == 0:  # Save every 2 epochs
                model.save(model_save_path)
                training_session.add_log("INFO", f"Checkpoint saved for group {group_name} after epoch {epoch}")
                safe_commit(db)

        # Final model save
        model.save(model_save_path)
        training_session.add_log("INFO", f"Training completed for group {group_name}. Final model saved to {model_save_path}")
        
        # Mark all pieces in the group as trained
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
        
        if model:
            try:
                model.save(model_save_path)
                training_session.add_log("INFO", f"Model saved for group {group_name} after error")
                safe_commit(db)
            except Exception as save_error:
                logger.error(f"Failed to save model after error: {str(save_error)}")
    
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Training completed for group {group_name}")


# Backward compatibility function
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