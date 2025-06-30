import asyncio
import os
import logging
import shutil
import torch
from requests import Session
from ultralytics import YOLO
import yaml
from collections import Counter
import matplotlib.pyplot as plt
from training.app.db.models.piece_image import PieceImage
from training.app.services.basic_rotation_service import rotate_and_update_images
from training.app.db.models.piece import Piece
# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(),
                              logging.FileHandler("training_logs.log", mode='a')])

logger = logging.getLogger(__name__)

# Global stop event
stop_event = asyncio.Event()
stop_sign = False

async def stop_training():
    stop_event.set()
    stop_sign == True
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

def analyze_class_distribution(data_yaml_path):
    """Analyze and plot class distribution in the dataset."""
    with open(data_yaml_path, 'r') as f:
        data = yaml.safe_load(f)

    class_counts = Counter()
    train_labels_dir = data['train'].replace('images', 'labels')

    label_files = [os.path.join(train_labels_dir, f) for f in os.listdir(train_labels_dir) if f.endswith('.txt')]

    for label_file in label_files:
        with open(label_file, 'r') as lf:
            for line in lf:
                try:
                    class_idx = int(line.split()[0])
                    class_counts[class_idx] += 1
                except ValueError:
                    logger.error(f"Error parsing line in {label_file}: {line}")

    class_names = data['names']
    class_labels = [class_names[i] for i in sorted(class_counts.keys())]
    counts = [class_counts[i] for i in sorted(class_counts.keys())]

    plt.figure(figsize=(10, 6))
    plt.bar(class_labels, counts, color='skyblue')
    plt.xlabel('Class')
    plt.ylabel('Number of Samples')
    plt.title('Class Distribution')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig("class_distribution.png")
    plt.show()

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



def train_model(piece_label: str, db: Session):
    model = None
    try:
        # Set service directory
        service_dir = os.path.dirname(os.path.abspath(__file__))
        logger.info(f"Service directory: {service_dir}")

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

        # Generate a custom dataset for the specific piece
        piece_data_dir = os.path.join(service_dir, "..", "..", "dataset_custom")
        os.makedirs(piece_data_dir, exist_ok=True)

        image_dir = os.path.join(piece_data_dir, "images", "valid",piece_label)
        image_dir_train = os.path.join(piece_data_dir, "images", "train",piece_label)
        label_dir = os.path.join(piece_data_dir, "labels", "valid",piece_label)
        label_dir_train = os.path.join(piece_data_dir, "labels", "train",piece_label)
        os.makedirs(image_dir, exist_ok=True)
        os.makedirs(label_dir, exist_ok=True)
        os.makedirs(image_dir_train, exist_ok=True)
        os.makedirs(label_dir_train, exist_ok=True)

        # Copy images and labels to the custom dataset directory
        for image in images:
            shutil.copy(image.piece_path, os.path.join(image_dir, os.path.basename(image.piece_path)))
            for annotation in image.annotations:
                label_path = os.path.join(label_dir, annotation.annotationTXT_name)
                with open(label_path, "w") as label_file:
                    label_file.write(f"{piece.class_data_id} {annotation.x} {annotation.y} {annotation.width} {annotation.height}\n")

        # Create a custom data.yaml for this piece
        data_yaml_path = os.path.join(piece_data_dir, "data.yaml")


        model_save_path = os.path.join(service_dir, '..', '..', 'detection', 'models', f"yolo8x_model.pt")
        logger.info(f"Resolved data.yaml path: {data_yaml_path}")
        logger.info(f"Model save path: {model_save_path}")

        if not os.path.isfile(data_yaml_path):
            logger.error(f"data.yaml file not found at {data_yaml_path}")
            return

        # Validate dataset for issues
        validate_dataset(data_yaml_path)

        # Ensure the model directory exists
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

        # Rotate and update images for the specific piece
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
        logger.info(f"Using image size: {imgsz}")

        logger.info(f"Starting fine-tuning for piece: {piece_label}")
        logger.info(f"Using device: {device}, Batch size: {batch_size}")

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
            "label_smoothing" : 0.1,
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

        # Set the optimizer object

        # Fine-tuning loop
        for epoch in range(25):
            if stop_event.is_set():
                logger.info("Stop event detected. Ending training.")
                break

            # Start the training process
            model.train(
                data=data_yaml_path,
                epochs=1,
                imgsz=640,
                batch=batch_size,
                device=device,
                project=os.path.dirname(model_save_path),
                name=piece_label,
                exist_ok=True,
                amp=True,
                patience=10,
                augment=True,  # Ensure augmentation is enabled
                **hyperparameters
            )
            
            # Validate the model periodically during training to monitor metrics
            if epoch % 5 == 0:  # Every 5 epochs, check validation performance
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

            logger.info(f"Results after epoch {epoch + 1}")

        logger.info(f"Model fine-tuning complete for piece: {piece_label}. Final model saved to {model_save_path}")

        # Update the `is_yolo_trained` field for the piece
        piece.is_yolo_trained = True
        db.commit()

    except Exception as e:
        logger.error(f"An error occurred: {e}")
        if model:
            try:
                model.save(model_save_path)
                logger.info(f"Model saved at {model_save_path} after encountering an error.")
            except Exception as save_error:
                logger.error(f"Failed to save model after error: {save_error}")
    finally:
        stop_event.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Fine-tuning process finished.")

