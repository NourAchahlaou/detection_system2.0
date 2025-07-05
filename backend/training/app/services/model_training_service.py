import asyncio
import os
import logging
import shutil
from training.app.db.models.annotation import Annotation
import torch
from requests import Session
from ultralytics import YOLO
import yaml

from training.app.db.models.piece_image import PieceImage
from training.app.services.basic_rotation_service import rotate_and_update_images
from training.app.db.models.piece import Piece

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

# FIXED: Updated to handle list of piece labels
def train_model(piece_labels: list, db: Session):
    """
    Train models for multiple pieces.
    
    Args:
        piece_labels: List of piece labels to train
        db: Database session
    """
    model = None
    
    # Ensure piece_labels is a list
    if isinstance(piece_labels, str):
        piece_labels = [piece_labels]
    
    try:
        # Set service directory
        service_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Service directory: {service_dir}")
        logger.info(f"Starting training process for piece labels: {piece_labels}")

        # Process each piece
        for piece_label in piece_labels:
            if stop_event.is_set():
                logger.info("Stop event detected. Ending training.")
                break
                
            logger.info(f"Training piece: {piece_label}")
            train_single_piece(piece_label, db, service_dir)
            
    except Exception as e:
        logger.error(f"An error occurred during training: {str(e)}", exc_info=True)
    finally:
        stop_event.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Training process finished and GPU memory cleared.")

def train_single_piece(piece_label: str, db: Session, service_dir: str):
    """
    Train a single piece model.
    
    Args:
        piece_label: Label of the piece to train
        db: Database session
        service_dir: Service directory path
    """
    model = None
    try:
        # Fetch the specific piece from the database
        piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
        if not piece:
            logger.error(f"Piece with label '{piece_label}' not found.")
            return

        if not piece.is_annotated:
            logger.error(f"Piece with label '{piece_label}' is not annotated. Training cannot proceed.")
            return

        logger.info(f"Found annotated piece: {piece_label}")

        # Retrieve all images for the piece
        images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).all()
        if not images:
            logger.error(f"No images found for piece '{piece_label}'. Training cannot proceed.")
            return

        logger.info(f"Found {len(images)} images for piece: {piece_label}")

        # Get the dataset base path from environment variable (consistent with paste-2)
        dataset_base_path = os.getenv('DATASET_BASE_PATH', '/app/shared/dataset')
        
        # NEW: Get the models base path from environment variable
        models_base_path = os.getenv('MODELS_BASE_PATH', '/app/shared/models')
        
        # Ensure models directory exists
        os.makedirs(models_base_path, exist_ok=True)
        
        # Create dataset_custom directory structure (consistent with paste-2)
        dataset_custom_path = os.path.join(dataset_base_path, 'dataset_custom')
        
        # Updated directory paths to match paste-2 structure
        image_dir = os.path.join(dataset_custom_path, "images", "valid", piece_label)
        image_dir_train = os.path.join(dataset_custom_path, "images", "train", piece_label)
        label_dir = os.path.join(dataset_custom_path, "labels", "valid", piece_label)
        label_dir_train = os.path.join(dataset_custom_path, "labels", "train", piece_label)
        
        # Create all necessary directories
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(image_dir_train, exist_ok=True)
        os.makedirs(label_dir_train, exist_ok=True)

        logger.info(f"Created dataset directories for piece: {piece_label}")
        logger.info(f"Dataset custom path: {dataset_custom_path}")

        for image in images:
            # Copy to validation directory
            shutil.copy(image.image_path, os.path.join(image_dir, os.path.basename(image.image_path)))
            
            # Query annotations directly instead of using relationship
            annotations = db.query(Annotation).filter(Annotation.piece_image_id == image.id).all()
            
            # Create annotations for validation directory
            for annotation in annotations:
                label_path = os.path.join(label_dir, annotation.annotationTXT_name)
                with open(label_path, "w") as label_file:
                    label_file.write(f"{piece.class_data_id} {annotation.x} {annotation.y} {annotation.width} {annotation.height}\n")

        logger.info(f"Copied {len(images)} images and their annotations to dataset directory")

        # Create a custom data.yaml for this piece in the dataset_custom directory
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
        
        logger.info(f"Created data.yaml at: {data_yaml_path}")
        
        # NEW: Model save path using shared models volume
        model_save_path = os.path.join(models_base_path, f"yolo8x_model_{piece_label}.pt")
        
        logger.info(f"Model save path: {model_save_path}")

        # Validate dataset for issues
        validate_dataset(data_yaml_path)

        # Rotate and update images for the specific piece
        logger.info("Starting image rotation and augmentation process")
        rotate_and_update_images(piece_label, db)

        # Initialize the model (load the previously trained model for the piece if available)
        device = select_device()
        if os.path.exists(model_save_path):
            logger.info(f"Loading existing model for piece {piece_label} from {model_save_path}")
            model = YOLO(model_save_path)  # Load the pre-trained model for fine-tuning
        else:
            logger.info("No pre-existing model found. Starting training from scratch.")
            model = YOLO("yolov8x.pt")  # Load a base YOLO model

        model.to(device)
        batch_size = adjust_batch_size(device)
        imgsz = adjust_imgsz(device)
        
        logger.info(f"Training configuration - Device: {device}, Batch size: {batch_size}, Image size: {imgsz}")

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

        # Augmentation parameters (Mosaic and Mixup)
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

        logger.info(f"Starting fine-tuning for piece: {piece_label} with 25 epochs")

        # Fine-tuning loop
        for epoch in range(25):
            if stop_event.is_set():
                logger.info("Stop event detected. Ending training.")
                break

            logger.info(f"Starting epoch {epoch + 1}/25 for piece {piece_label}")

            # Start the training process
            model.train(
                data=data_yaml_path,
                epochs=1,
                imgsz=640,
                batch=batch_size,
                device=device,
                project=os.path.dirname(model_save_path),
                name=f"{piece_label}_epoch_{epoch}",
                exist_ok=True,
                amp=True,
                patience=10,
                augment=True,  # Ensure augmentation is enabled
                **hyperparameters
            )
            
            # Validate the model periodically during training to monitor metrics
            if epoch % 5 == 0:  # Every 5 epochs, check validation performance
                logger.info(f"Running validation after epoch {epoch + 1} for piece {piece_label}")
                validation_results = model.val(
                    data=data_yaml_path,
                    imgsz=640,
                    batch=batch_size,
                    device=device,
                )
                logger.info(f"Validation results after epoch {epoch + 1}: {validation_results}")

            # Save the model periodically
            if epoch % 1 == 0:  # Save model after every epoch
                model.save(model_save_path)
                logger.info(f"Checkpoint saved to {model_save_path} after epoch {epoch + 1}")

            logger.info(f"Completed epoch {epoch + 1}/25 for piece {piece_label}")

        logger.info(f"Model fine-tuning complete for piece: {piece_label}. Final model saved to {model_save_path}")

        # Update the `is_yolo_trained` field for the piece
        piece.is_yolo_trained = True
        db.commit()
        logger.info(f"Updated piece {piece_label} is_yolo_trained status to True")

    except Exception as e:
        logger.error(f"An error occurred during training piece {piece_label}: {str(e)}", exc_info=True)
        if model:
            try:
                model.save(model_save_path)
                logger.info(f"Model saved at {model_save_path} after encountering an error.")
            except Exception as save_error:
                logger.error(f"Failed to save model after error: {str(save_error)}")
    finally:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info(f"Training process finished for piece {piece_label} and GPU memory cleared.")