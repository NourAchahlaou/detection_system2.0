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

from training.app.db.models.piece_image import PieceImage
from training.app.services.basic_rotation_service import rotate_and_update_images
from training.app.db.models.piece import Piece
from training.app.db.models.training import TrainingSession
from training.app.db.session import create_new_session, safe_commit, safe_close

# Set up logging with dedicated log volume
log_dir = os.getenv("LOG_PATH", "/usr/srv/logs")
log_file = os.path.join(log_dir, "model_training_service.log")

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

async def stop_training():
    """Set the stop event to signal training to stop."""
    global stop_sign
    stop_event.set()
    stop_sign = True
    logger.info("Stop training signal sent.")

def select_device():
    """Select the best available device (GPU if available, else CPU)."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"CUDA Device Detected: {torch.cuda.get_device_name(0)}")
        return device
    else:
        logger.info("No GPU detected. Using CPU.")
        return torch.device('cpu')

def adjust_batch_size(device, base_batch_size=8):
    """Adjust batch size based on the available device."""
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(0).total_memory
        if total_memory > 8 * 1024**3:  # If GPU has more than 8GB of memory
            return base_batch_size * 2
        elif total_memory > 4 * 1024**3:  # If GPU has more than 4GB of memory
            return base_batch_size
        else:
            return base_batch_size // 2
    else:
        return base_batch_size // 2

def adjust_imgsz(device):
    """Adjust image size based on available GPU memory."""
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory
        if total_memory > 8 * 1024**3:  # If GPU has more than 8GB of memory
            return 1024  # Larger image size
        elif total_memory >= 4 * 1024**3:  # If GPU has more than 4GB of memory
            return 640  # Medium image size
        else:
            return 416  # Smaller image size
    else:
        return 320  # Default for CPU

def validate_dataset(data_yaml_path):
    """Validate dataset for label consistency and data split ratio."""
    logger.info("Validating dataset...")
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    # Check split ratio
    train_images = len(os.listdir(data['train']))
    val_images = len(os.listdir(data['val']))
    
    total_images = train_images + val_images 

    if not (0.75 <= train_images / total_images <= 0.85):
        logger.warning("Train dataset split ratio is outside the recommended range (75-85%).")
    if not (0.05 <= val_images / total_images <= 0.15):
        logger.warning("Validation dataset split ratio is outside the recommended range (5-15%).")

    logger.info("Dataset validation complete.")

def add_background_images(data_yaml_path):
    """Add background images to the dataset to reduce false positives."""
    logger.info("Adding background images to the dataset...")
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    background_dir = data['background']  # Assuming a separate directory for background images
    if os.path.exists(background_dir):
        logger.info(f"Adding {len(os.listdir(background_dir))} background images to the training set.")
    else:
        logger.warning("No background images found.")

def update_training_progress(session_id: int, epoch: int, total_epochs: int, losses: dict = None, metrics: dict = None):
    """Update training progress in database with real-time data."""
    db = create_new_session()
    try:
        training_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if training_session:
            # Update epoch and progress
            training_session.current_epoch = epoch
            training_session.progress_percentage = (epoch / total_epochs) * 100
            
            # Update losses if provided
            if losses:
                training_session.update_losses(
                    box_loss=losses.get('box_loss'),
                    cls_loss=losses.get('cls_loss'),
                    dfl_loss=losses.get('dfl_loss')
                )
            
            # Update metrics if provided
            if metrics:
                training_session.update_metrics(
                    instances=metrics.get('instances'),
                    lr=metrics.get('lr'),
                    momentum=metrics.get('momentum')
                )
            
            # Update timestamp
            training_session.last_updated = datetime.utcnow()
            
            if safe_commit(db):
                logger.info(f"Updated training progress - Epoch: {epoch}/{total_epochs}, Progress: {training_session.progress_percentage:.1f}%")
                if losses:
                    logger.info(f"Current losses - Box: {losses.get('box_loss', 'N/A')}, Cls: {losses.get('cls_loss', 'N/A')}, DFL: {losses.get('dfl_loss', 'N/A')}")
    except Exception as e:
        logger.error(f"Failed to update training progress: {str(e)}")
    finally:
        safe_close(db)

class TrainingCallback:
    """Custom callback to capture training metrics in real-time."""
    
    def __init__(self, session_id: int, total_epochs: int):
        self.session_id = session_id
        self.total_epochs = total_epochs
        self.current_epoch = 0
        self.batch_count = 0
        
    def on_train_epoch_start(self, trainer):
        """Called at the start of each training epoch."""
        self.current_epoch = trainer.epoch + 1
        self.batch_count = 0
        logger.info(f"Starting epoch {self.current_epoch}/{self.total_epochs}")
        
    def on_train_batch_end(self, trainer):
        """Called at the end of each training batch."""
        try:
            self.batch_count += 1
            
            # Get current losses from trainer
            if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                # YOLO typically has [box_loss, cls_loss, dfl_loss]
                losses = {
                    'box_loss': float(trainer.loss_items[0]) if len(trainer.loss_items) > 0 else 0,
                    'cls_loss': float(trainer.loss_items[1]) if len(trainer.loss_items) > 1 else 0,
                    'dfl_loss': float(trainer.loss_items[2]) if len(trainer.loss_items) > 2 else 0
                }
                
                # Get current learning rate
                current_lr = trainer.optimizer.param_groups[0]['lr'] if trainer.optimizer else 0
                
                metrics = {
                    'lr': current_lr,
                    'momentum': trainer.optimizer.param_groups[0].get('momentum', 0.9) if trainer.optimizer else 0.9,
                    'batch_count': self.batch_count
                }
                
                # Update every 10 batches to avoid too frequent database updates
                if self.batch_count % 10 == 0:
                    update_training_progress(
                        session_id=self.session_id,
                        epoch=self.current_epoch,
                        total_epochs=self.total_epochs,
                        losses=losses,
                        metrics=metrics
                    )
        except Exception as e:
            logger.error(f"Error in training callback: {str(e)}")
    
    def on_train_epoch_end(self, trainer):
        """Called at the end of each training epoch."""
        try:
            # Get final epoch losses
            if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                losses = {
                    'box_loss': float(trainer.loss_items[0]) if len(trainer.loss_items) > 0 else 0,
                    'cls_loss': float(trainer.loss_items[1]) if len(trainer.loss_items) > 1 else 0,
                    'dfl_loss': float(trainer.loss_items[2]) if len(trainer.loss_items) > 2 else 0
                }
                
                current_lr = trainer.optimizer.param_groups[0]['lr'] if trainer.optimizer else 0
                metrics = {
                    'lr': current_lr,
                    'momentum': trainer.optimizer.param_groups[0].get('momentum', 0.9) if trainer.optimizer else 0.9,
                    'batch_count': self.batch_count
                }
                
                # Final update for the epoch
                update_training_progress(
                    session_id=self.session_id,
                    epoch=self.current_epoch,
                    total_epochs=self.total_epochs,
                    losses=losses,
                    metrics=metrics
                )
                
                logger.info(f"Completed epoch {self.current_epoch}/{self.total_epochs}")
                
        except Exception as e:
            logger.error(f"Error in epoch end callback: {str(e)}")
    
    def on_val_end(self, trainer):
        """Called at the end of validation."""
        try:
            # Get validation metrics if available
            if hasattr(trainer, 'metrics') and trainer.metrics is not None:
                # Log validation metrics
                logger.info(f"Validation completed for epoch {self.current_epoch}")
                
                # You can add validation metrics update here if needed
                # For example: mAP, precision, recall, etc.
                
        except Exception as e:
            logger.error(f"Error in validation callback: {str(e)}")
    
    def on_fit_epoch_end(self, trainer):
        """Called at the end of each fit epoch (after validation)."""
        try:
            # This is called after both training and validation for the epoch
            # You can use this for final epoch statistics
            pass
        except Exception as e:
            logger.error(f"Error in fit epoch end callback: {str(e)}")

def TrainingCallback(session_id: int, total_epochs: int):
    """Create custom callback functions for YOLO training."""
    
    def on_train_epoch_start(trainer):
        """Called at the start of each training epoch."""
        current_epoch = trainer.epoch + 1
        logger.info(f"Starting epoch {current_epoch}/{total_epochs}")
        
        # Update database with epoch start
        update_training_progress(
            session_id=session_id,
            epoch=current_epoch,
            total_epochs=total_epochs
        )
    
    def on_train_batch_end(trainer):
        """Called at the end of each training batch."""
        try:
            # Get current epoch and batch info
            current_epoch = trainer.epoch + 1
            
            # Get batch number from trainer
            batch_num = getattr(trainer, 'batch_i', 0)
            
            # Get current losses from trainer
            if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                losses = {
                    'box_loss': float(trainer.loss_items[0]) if len(trainer.loss_items) > 0 else 0,
                    'cls_loss': float(trainer.loss_items[1]) if len(trainer.loss_items) > 1 else 0,
                    'dfl_loss': float(trainer.loss_items[2]) if len(trainer.loss_items) > 2 else 0
                }
                
                # Get current learning rate
                current_lr = trainer.optimizer.param_groups[0]['lr'] if trainer.optimizer else 0
                
                metrics = {
                    'lr': current_lr,
                    'momentum': trainer.optimizer.param_groups[0].get('momentum', 0.9) if trainer.optimizer else 0.9,
                    'batch_num': batch_num
                }
                
                # Update every 10 batches to avoid too frequent database updates
                if batch_num % 10 == 0:
                    update_training_progress(
                        session_id=session_id,
                        epoch=current_epoch,
                        total_epochs=total_epochs,
                        losses=losses,
                        metrics=metrics
                    )
        except Exception as e:
            logger.error(f"Error in batch end callback: {str(e)}")
    
    def on_train_epoch_end(trainer):
        """Called at the end of each training epoch."""
        try:
            current_epoch = trainer.epoch + 1
            
            # Get final epoch losses
            if hasattr(trainer, 'loss_items') and trainer.loss_items is not None:
                losses = {
                    'box_loss': float(trainer.loss_items[0]) if len(trainer.loss_items) > 0 else 0,
                    'cls_loss': float(trainer.loss_items[1]) if len(trainer.loss_items) > 1 else 0,
                    'dfl_loss': float(trainer.loss_items[2]) if len(trainer.loss_items) > 2 else 0
                }
                
                current_lr = trainer.optimizer.param_groups[0]['lr'] if trainer.optimizer else 0
                metrics = {
                    'lr': current_lr,
                    'momentum': trainer.optimizer.param_groups[0].get('momentum', 0.9) if trainer.optimizer else 0.9
                }
                
                # Final update for the epoch
                update_training_progress(
                    session_id=session_id,
                    epoch=current_epoch,
                    total_epochs=total_epochs,
                    losses=losses,
                    metrics=metrics
                )
                
                logger.info(f"Completed epoch {current_epoch}/{total_epochs}")
                
        except Exception as e:
            logger.error(f"Error in epoch end callback: {str(e)}")
    
    def on_val_end(trainer):
        """Called at the end of validation."""
        try:
            current_epoch = trainer.epoch + 1
            
            # Get validation metrics if available
            if hasattr(trainer, 'metrics') and trainer.metrics is not None:
                logger.info(f"Validation completed for epoch {current_epoch}")
                
                # Extract validation metrics
                val_metrics = {}
                if hasattr(trainer.metrics, 'box_map'):
                    val_metrics['mAP_0.5'] = trainer.metrics.box_map
                if hasattr(trainer.metrics, 'box_map_50'):
                    val_metrics['mAP_0.5'] = trainer.metrics.box_map_50
                if hasattr(trainer.metrics, 'box_map_75'):
                    val_metrics['mAP_0.75'] = trainer.metrics.box_map_75
                
                # Update with validation metrics
                if val_metrics:
                    update_training_progress(
                        session_id=session_id,
                        epoch=current_epoch,
                        total_epochs=total_epochs,
                        metrics=val_metrics
                    )
                
        except Exception as e:
            logger.error(f"Error in validation callback: {str(e)}")
    
    return {
        'on_train_epoch_start': on_train_epoch_start,
        'on_train_batch_end': on_train_batch_end,
        'on_train_epoch_end': on_train_epoch_end,
        'on_val_end': on_val_end
    }
def train_model(piece_labels: list, db: Session, session_id: int):

    # Train models for multiple pieces.
    
   
    
    # Ensure piece_labels is a list
    if isinstance(piece_labels, str):
        piece_labels = [piece_labels]
    
    try:
        # Get training session
        training_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if not training_session:
            logger.error(f"Training session {session_id} not found")
            return

        # Set service directory
        service_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Service directory: {service_dir}")
        
        # Log training start
        training_session.add_log("INFO", f"Starting training process for piece labels: {piece_labels}")
        safe_commit(db)

        # Select device and update session
        device = select_device()
        training_session.device_used = str(device)
        safe_commit(db)

        # Count total images across all pieces
        total_images = 0
        for piece_label in piece_labels:
            piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
            if piece:
                images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).all()
                total_images += len(images)

        # Update session with total images
        training_session.total_images = total_images
        safe_commit(db)

        # Process each piece
        for i, piece_label in enumerate(piece_labels):
            if stop_event.is_set():
                training_session.add_log("INFO", "Stop event detected. Ending training.")
                safe_commit(db)
                break
                
            logger.info(f"Training piece: {piece_label}")
            training_session.add_log("INFO", f"Training piece: {piece_label}")
            safe_commit(db)
            
            # Update progress percentage based on piece completion
            piece_progress = (i / len(piece_labels)) * 100
            training_session.progress_percentage = piece_progress
            safe_commit(db)
            
            train_single_piece(piece_label, db, service_dir, session_id)
            
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}", exc_info=True)
        # Update session with error
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
        logger.info("Training process finished and GPU memory cleared.")

def train_single_piece(piece_label: str, db: Session, service_dir: str, session_id: int):
    """
    Train a single piece model with real-time loss updates.
    
    Args:
        piece_label: Label of the piece to train
        db: Database session
        service_dir: Service directory path
        session_id: Training session ID
    """
    model = None
    try:
        # Get training session
        training_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if not training_session:
            logger.error(f"Training session {session_id} not found")
            return

        # Fetch the specific piece from the database
        piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
        if not piece:
            error_msg = f"Piece with label '{piece_label}' not found."
            logger.error(error_msg)
            training_session.add_log("ERROR", error_msg)
            db.commit()
            return

        if not piece.is_annotated:
            error_msg = f"Piece with label '{piece_label}' is not annotated. Training cannot proceed."
            logger.error(error_msg)
            training_session.add_log("ERROR", error_msg)
            db.commit()
            return

        training_session.add_log("INFO", f"Found annotated piece: {piece_label}")
        db.commit()

        # Retrieve all images for the piece
        images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).all()
        if not images:
            error_msg = f"No images found for piece '{piece_label}'. Training cannot proceed."
            logger.error(error_msg)
            training_session.add_log("ERROR", error_msg)
            db.commit()
            return

        training_session.add_log("INFO", f"Found {len(images)} images for piece: {piece_label}")
        db.commit()

        # Get the dataset base path from environment variable
        dataset_base_path = os.getenv('DATASET_BASE_PATH', '/app/shared/dataset')
        
        # Get the models base path from environment variable
        models_base_path = os.getenv('MODELS_BASE_PATH', '/app/shared/models')
        
        # Ensure models directory exists
        os.makedirs(models_base_path, exist_ok=True)
        
        # Create dataset_custom directory structure
        dataset_custom_path = os.path.join(dataset_base_path, 'dataset_custom')
        
        # Directory paths
        image_dir = os.path.join(dataset_custom_path, "images", "valid", piece_label)
        image_dir_train = os.path.join(dataset_custom_path, "images", "train", piece_label)
        label_dir = os.path.join(dataset_custom_path, "labels", "valid", piece_label)
        label_dir_train = os.path.join(dataset_custom_path, "labels", "train", piece_label)
        
        # Create all necessary directories
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(image_dir_train, exist_ok=True)
        os.makedirs(label_dir_train, exist_ok=True)

        training_session.add_log("INFO", f"Created dataset directories for piece: {piece_label}")
        db.commit()

        # Copy images and create annotation files
        for image in images:
            # Copy to validation directory
            shutil.copy(image.image_path, os.path.join(image_dir, os.path.basename(image.image_path)))
            
            # Query annotations directly
            annotations = db.query(Annotation).filter(Annotation.piece_image_id == image.id).all()
            
            # Create annotations for validation directory
            for annotation in annotations:
                label_path = os.path.join(label_dir, annotation.annotationTXT_name)
                with open(label_path, "w") as label_file:
                    label_file.write(f"{piece.class_data_id} {annotation.x} {annotation.y} {annotation.width} {annotation.height}\n")

        training_session.add_log("INFO", f"Copied {len(images)} images and their annotations to dataset directory")
        db.commit()

        # Create a custom data.yaml for this piece
        data_yaml_path = os.path.join(dataset_custom_path, "data.yaml")
        
        # Create the data.yaml file with correct paths
        data_yaml_content = {
            'train': os.path.join(dataset_custom_path, 'images', 'train'),
            'val': os.path.join(dataset_custom_path, 'images', 'valid'),
            'nc': 1,  # Number of classes (assuming single class per piece)
            'names': [piece_label]
        }
        
        with open(data_yaml_path, 'w') as f:
            yaml.dump(data_yaml_content, f)
        
        training_session.add_log("INFO", f"Created data.yaml at: {data_yaml_path}")
        db.commit()
        
        # Model save path using shared models volume
        model_save_path = os.path.join(models_base_path, f"yolo8x_model_{piece_label}.pt")
        
        # Validate dataset for issues
        validate_dataset(data_yaml_path)

        # Rotate and update images for the specific piece
        training_session.add_log("INFO", "Starting image rotation and augmentation process")
        db.commit()
        rotate_and_update_images(piece_label, db)

        # Initialize the model
        device = select_device()
        if os.path.exists(model_save_path):
            training_session.add_log("INFO", f"Loading existing model for piece {piece_label}")
            model = YOLO(model_save_path)  # Load the pre-trained model for fine-tuning
        else:
            training_session.add_log("INFO", "No pre-existing model found. Starting training from scratch.")
            model_path = os.path.join(models_base_path, "yolov8m.pt")
            if os.path.exists(model_path):
                model = YOLO(model_path)
            else:
                # Fallback to default YOLO model download location
                model = YOLO("yolov8m.pt")  # Load a base YOLO model

        model.to(device)
        batch_size = adjust_batch_size(device)
        imgsz = adjust_imgsz(device)
        
        # Update training session with configuration
        training_session.batch_size = batch_size
        training_session.image_size = imgsz
        db.commit()

        # Hyperparameters setup
        hyperparameters = {
            "cos_lr": False,
            "lr0": 0.0001,  # Decreased learning rate for finer updates
            "lrf": 0.01,
            "momentum": 0.937,
            "weight_decay": 0.0005,
            "dropout": 0.2, 
            "warmup_epochs": 10.0,  # Increased warmup for fine-tuning
            "warmup_momentum": 0.8,
            "warmup_bias_lr": 0.1,
            "label_smoothing": 0.1,
        }

        # Augmentation parameters
        augmentations = {
            "hsv_h": 0.015,  
            "hsv_s": 0.7,  
            "hsv_v": 0.4,  
            "degrees": 10.0,  # Increase rotation degree for better variance
            "translate": 0.2,  # Increase translation range
            "scale": 0.3,  # Slightly higher scaling to improve generalization
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.7,  # Increase horizontal flip probability
            "mosaic": 0.7,  # Increase mosaic strength
            "mixup": 0.1,  # Consider adding mixup for even better generalization
            "copy_paste": 0.0,
            "erasing": 0.5,  # Increase image erasing to reduce overfitting
            "crop_fraction": 1.0,
        }

        # Merge augmentations into hyperparameters
        hyperparameters.update(augmentations)

        training_session.add_log("INFO", f"Starting training for piece: {piece_label} with {training_session.epochs} epochs")
        db.commit()

        # Create training callback for real-time updates
        callback = TrainingCallback(session_id, training_session.epochs)

        # Add the callback to the model
        for callback_name, callback_func in callback.items():
            model.add_callback(callback_name, callback_func)

        # Start the training process with all epochs at once
        training_session.add_log("INFO", f"Starting training with {training_session.epochs} epochs")
        db.commit()
        
        results = model.train(
            data=data_yaml_path,
            epochs=training_session.epochs,  # Train for all epochs at once
            imgsz=imgsz,
            batch=batch_size,
            device=device,
            project=os.path.dirname(model_save_path),
            name=f"{piece_label}_training",
            exist_ok=True,
            amp=True,
            patience=10,
            augment=True,
            save_period=1,  # Save every epoch
            **hyperparameters
        )

        # Final model save and completion
        model.save(model_save_path)
        training_session.model_path = model_save_path
        training_session.add_log("SUCCESS", f"Model training complete for piece: {piece_label}. Final model saved to {model_save_path}")
        db.commit()

        # Update the is_yolo_trained field for the piece
        piece.is_yolo_trained = True
        db.commit()
        
        training_session.add_log("INFO", f"Updated piece {piece_label} is_yolo_trained status to True")
        db.commit()

    except Exception as e:
        error_msg = f"An error occurred during training piece {piece_label}: {str(e)}"
        logger.error(error_msg, exc_info=True)
        
        # Update session with error
        training_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if training_session:
            training_session.add_log("ERROR", error_msg)
            db.commit()
        
        if model:
            try:
                model.save(model_save_path)
                training_session.add_log("INFO", f"Model saved at {model_save_path} after encountering an error.")
                db.commit()
            except Exception as save_error:
                save_error_msg = f"Failed to save model after error: {str(save_error)}"
                logger.error(save_error_msg)
                training_session.add_log("ERROR", save_error_msg)
                db.commit()
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Training process finished for piece {piece_label} and GPU memory cleared.")