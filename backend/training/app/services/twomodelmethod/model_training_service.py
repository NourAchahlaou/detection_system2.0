import asyncio
import os
import logging
import shutil
import json
import numpy as np
from pathlib import Path
from training.app.db.models.annotation import Annotation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b0
from sqlalchemy.orm import Session
from ultralytics import YOLO
import yaml
from datetime import datetime
import time
from PIL import Image
import faiss

from training.app.db.models.piece_image import PieceImage
from training.app.services.basic_rotation_service import rotate_and_update_images
from training.app.db.models.piece import Piece
from training.app.db.models.training import TrainingSession
from training.app.db.session import create_new_session, safe_commit, safe_close

# Set up logging with dedicated log volume
log_dir = os.getenv("LOG_PATH", "/usr/srv/logs")
log_file = os.path.join(log_dir, "hybrid_training_service.log")

os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, mode='a')
    ]
)

logger = logging.getLogger(__name__)


# Global stop event
stop_event = asyncio.Event()
stop_sign = False

# Configuration constants
DETECTION_IMAGE_SIZE = 640  # For YOLO detection
EMBEDDING_IMAGE_SIZE = 224  # For embedding extraction
EMBEDDING_DIM = 512  # Embedding vector dimension

class AirplanePartDataset(Dataset):
    """Custom dataset for airplane part embedding training."""

    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        label = self.labels[idx]

        try:
            image = Image.open(image_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            logger.error(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            if self.transform:
                fallback = self.transform(Image.new('RGB', (224, 224), (0, 0, 0)))
            else:
                fallback = torch.zeros(3, 224, 224)
            return fallback, label

class EmbeddingNetwork(nn.Module):
    """Deep embedding network for airplane part identification."""

    def __init__(self, num_classes, embedding_dim=512, backbone='resnet50', pretrained=True):
        super(EmbeddingNetwork, self).init()
        self.embedding_dim = embedding_dim

        if backbone == 'resnet50':
            self.backbone = resnet50(pretrained=pretrained)
            # Remove the final classification layer
            self.backbone.fc = nn.Identity()
            backbone_dim = 2048
        elif backbone == 'efficientnet_b0':
            self.backbone = efficientnet_b0(pretrained=pretrained)
            self.backbone.classifier = nn.Identity()
            backbone_dim = 1280
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Embedding layer
        self.embedding_layer = nn.Sequential(
            nn.Linear(backbone_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )

        # Classification head (for training with CrossEntropy + ArcFace)
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, x, return_embedding=False):
        features = self.backbone(x)
        embeddings = self.embedding_layer(features)

        # L2 normalize embeddings for metric learning
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        if return_embedding:
            return embeddings

        logits = self.classifier(embeddings)
        return logits, embeddings

class ArcFaceLoss(nn.Module):
    """ArcFace loss for better embedding learning."""

    def __init__(self, embedding_dim, num_classes, margin=0.5, scale=64):
        super(ArcFaceLoss, self).init()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings, labels):
        # Normalize weights and embeddings
        normalized_weights = nn.functional.normalize(self.weight, p=2, dim=1)
        normalized_embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        # Compute cosine similarity
        cosine = torch.mm(normalized_embeddings, normalized_weights.t())

        # Apply ArcFace margin
        theta = torch.acos(torch.clamp(cosine, -1.0 + 1e-7, 1.0 - 1e-7))
        target_theta = theta + self.margin
        target_cosine = torch.cos(target_theta)

        # Create one-hot labels
        one_hot = torch.zeros_like(cosine)
        one_hot.scatter(1, labels.view(-1, 1), 1)

        # Apply margin only to target class
        output = (one_hot * target_cosine) + ((1.0 - one_hot) * cosine)
        output = output * self.scale

        return output

class VectorDatabase:
    """Vector database for part embeddings using FAISS."""

    def __init__(self, embedding_dim=512, index_type='IndexFlatIP'):
        self.embedding_dim = embedding_dim
        self.index = None
        self.part_labels = []
        self.part_metadata = {}
        self.index_type = index_type

    def build_index(self, embeddings, labels, metadata=None):
        """Build FAISS index from embeddings."""
        logger.info(f"Building FAISS index with {len(embeddings)} embeddings")

        # Create FAISS index
        if self.index_type == 'IndexFlatIP':
            self.index = faiss.IndexFlatIP(self.embedding_dim)  # Inner product (cosine similarity)
        elif self.index_type == 'IndexIVFFlat':
            quantizer = faiss.IndexFlatIP(self.embedding_dim)
            self.index = faiss.IndexIVFFlat(quantizer, self.embedding_dim, min(100, len(embeddings) // 10))
            self.index.train(embeddings.astype(np.float32))

        # Add embeddings to index
        self.index.add(embeddings.astype(np.float32))
        self.part_labels = labels
        self.part_metadata = metadata or {}

        logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors")

    def search(self, query_embedding, k=5, threshold=0.5):
        """Search for similar parts in the database."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index first.")

        query_embedding = query_embedding.astype(np.float32)
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)

        scores, indices = self.index.search(query_embedding, k)

        results = []
        for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
            if score > threshold:
                result = {
                    'rank': i + 1,
                    'part_label': self.part_labels[idx],
                    'similarity_score': float(score),
                    'metadata': self.part_metadata.get(idx, {})
                }
                results.append(result)

        return results

    def save_index(self, save_path):
        """Save FAISS index and metadata to disk."""
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        # Save FAISS index
        faiss.write_index(self.index, str(save_path / "faiss_index.bin"))

        # Save metadata
        metadata = {
            'part_labels': self.part_labels,
            'part_metadata': self.part_metadata,
            'embedding_dim': self.embedding_dim,
            'index_type': self.index_type
        }

        with open(save_path / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Vector database saved to {save_path}")

    def load_index(self, load_path):
        """Load FAISS index and metadata from disk."""
        load_path = Path(load_path)

        # Load FAISS index
        self.index = faiss.read_index(str(load_path / "faiss_index.bin"))

        # Load metadata
        with open(load_path / "metadata.json", 'r') as f:
            metadata = json.load(f)

        self.part_labels = metadata['part_labels']
        self.part_metadata = metadata['part_metadata']
        self.embedding_dim = metadata['embedding_dim']
        self.index_type = metadata['index_type']

        logger.info(f"Vector database loaded from {load_path}")

async def stop_training():
    """Set the stop event to signal training to stop."""
    global stop_sign
    stop_event.set()
    stop_sign = True
    logger.info("Stop training signal sent.")

def select_device():
    """Select the best available device with T600 optimizations."""
    logger.info(f"PyTorch version: {torch.__version__}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        logger.info(f"Number of CUDA devices: {device_count}")

        for i in range(device_count):
            device_props = torch.cuda.get_device_properties(i)
            logger.info(f"Device {i}: {device_props.name}")
            logger.info(f"  Memory: {device_props.total_memory / 1024**3:.1f} GB")

        device = torch.device('cuda:0')
        logger.info(f"Selected device: {device}")

        # Test GPU functionality
        try:
            test_tensor = torch.randn(10, 10).to(device)
            logger.info("GPU functionality test: PASSED")
        except Exception as e:
            logger.error(f"GPU functionality test failed: {e}")
            device = torch.device('cpu')

        return device
    else:
        logger.info("No GPU detected. Using CPU for training")
        return torch.device('cpu')

def adjust_batch_size_for_stage(device, stage='detection', base_batch_size=8):
    """Adjust batch size based on device and training stage."""
    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(0).total_memory
        memory_gb = total_memory / 1024**3

        if stage == 'detection':
            # YOLO detection - more memory efficient
            if memory_gb > 6:
                return min(16, base_batch_size * 2)
            elif memory_gb > 4:
                return base_batch_size
            else:
                return max(4, base_batch_size // 2)
        else:  # embedding stage
            # Embedding training - more memory intensive
            if memory_gb > 6:
                return base_batch_size
            elif memory_gb > 4:
                return max(4, base_batch_size // 2)
            else:
                return max(2, base_batch_size // 4)
    else:
        return max(2, base_batch_size // 4)

def train_detection_stage(piece_labels: list, db: Session, session_id: int):
    """Train YOLO detector for generic 'part' detection (Stage 1)."""
    try:
        training_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if not training_session:
            logger.error(f"Training session {session_id} not found")
            return

        training_session.add_log("INFO", "Starting Stage 1: YOLO Detection Training (Generic Part Detection)")
        safe_commit(db)

        device = select_device()
        training_session.device_used = str(device)

        # Get dataset and models paths
        dataset_base_path = os.getenv('DATASET_BASE_PATH', '/app/shared/dataset')
        models_base_path = os.getenv('MODELS_BASE_PATH', '/app/shared/models')

        # Prepare detection dataset (all parts as single 'part' class)
        dataset_detection_path = os.path.join(dataset_base_path, 'detection_dataset')
        prepare_detection_dataset(piece_labels, db, dataset_detection_path, training_session)

        # Create detection data.yaml
        data_yaml_path = os.path.join(dataset_detection_path, "data.yaml")
        detection_yaml_content = {
            'train': os.path.join(dataset_detection_path, 'images', 'train'),
            'val': os.path.join(dataset_detection_path, 'images', 'valid'),
            'nc': 1,  # Single class: 'part'
            'names': ['part']
        }

        with open(data_yaml_path, 'w') as f:
            yaml.dump(detection_yaml_content, f)

        # Train YOLO detector
        detection_model_path = os.path.join(models_base_path, "part_detector.pt")

        # Use smaller YOLO model for T600
        if os.path.exists(os.path.join(models_base_path, "yolov8s.pt")):
            model = YOLO(os.path.join(models_base_path, "yolov8s.pt"))
        else:
            model = YOLO("yolov8s.pt")  # Small model for 4GB VRAM

        model.to(device)

        batch_size = adjust_batch_size_for_stage(device, 'detection', 16)
        training_session.batch_size = batch_size

        training_session.add_log("INFO", f"Training YOLO detector - Batch size: {batch_size}, Image size: {DETECTION_IMAGE_SIZE}")
        safe_commit(db)

        # Train with optimized parameters for T600
        results = model.train(
            data=data_yaml_path,
            epochs=30,  # Fewer epochs for detection stage
            imgsz=DETECTION_IMAGE_SIZE,
            batch=batch_size,
            device=device,
            project=os.path.dirname(detection_model_path),
            name="part_detection",
            exist_ok=True,
            amp=True,  # Mixed precision for T600
            plots=False,
            patience=10,
            augment=True,
            cos_lr=False,
            lr0=0.001,
            lrf=0.01,
            momentum=0.937,
            weight_decay=0.0005
        )

        # Save detection model
        model.save(detection_model_path)
        training_session.add_log("INFO", f"Stage 1 completed: YOLO detector saved at {detection_model_path}")
        safe_commit(db)

        return detection_model_path

    except Exception as e:
        logger.error(f"Detection stage training failed: {str(e)}", exc_info=True)
        training_session.add_log("ERROR", f"Detection stage failed: {str(e)}")
        safe_commit(db)
        raise

def prepare_detection_dataset(piece_labels: list, db: Session, dataset_path: str, training_session):
    """Prepare dataset for detection stage (all parts as single 'part' class)."""

    # Create directories
    train_img_dir = os.path.join(dataset_path, "images", "train")
    val_img_dir = os.path.join(dataset_path, "images", "valid")
    train_label_dir = os.path.join(dataset_path, "labels", "train")
    val_label_dir = os.path.join(dataset_path, "labels", "valid")

    for dir_path in [train_img_dir, val_img_dir, train_label_dir, val_label_dir]:
        os.makedirs(dir_path, exist_ok=True)

    total_images = 0

    for piece_label in piece_labels:
        if stop_event.is_set():
            break

        piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
        if not piece or not piece.is_annotated:
            continue

        images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).all()

        # Split 80/20 for train/val
        split_idx = int(len(images) * 0.8)
        train_images = images[:split_idx]
        val_images = images[split_idx:]

        # Process training images
        for img in train_images:
            copy_image_with_detection_labels(img, piece, train_img_dir, train_label_dir, db)
            total_images += 1

        # Process validation images
        for img in val_images:
            copy_image_with_detection_labels(img, piece, val_img_dir, val_label_dir, db)
            total_images += 1

    training_session.total_images = total_images
    training_session.add_log("INFO", f"Prepared detection dataset with {total_images} images")
    safe_commit(db)

def copy_image_with_detection_labels(image: PieceImage, piece: Piece, img_dir: str, label_dir: str, db: Session):
    """Copy image and create detection labels (all as class 0 'part')."""

    # Copy image
    img_name = os.path.basename(image.image_path)
    shutil.copy(image.image_path, os.path.join(img_dir, img_name))

    # Create label file (all annotations as class 0)
    annotations = db.query(Annotation).filter(Annotation.piece_image_id == image.id).all()

    label_name = os.path.splitext(img_name)[0] + ".txt"
    label_path = os.path.join(label_dir, label_name)

    with open(label_path, "w") as f:
        for annotation in annotations:
            # All parts are class 0 for detection
            f.write(f"0 {annotation.x} {annotation.y} {annotation.width} {annotation.height}\n")

def train_embedding_stage(piece_labels: list, db: Session, session_id: int):
    """Train embedding network for part identification (Stage 2)."""
    try:
        training_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if not training_session:
            logger.error(f"Training session {session_id} not found")
            return

        training_session.add_log("INFO", "Starting Stage 2: Embedding Network Training (Part Identification)")
        safe_commit(db)

        device = select_device()
        models_base_path = os.getenv('MODELS_BASE_PATH', '/app/shared/models')

        # Prepare embedding dataset
        train_dataset, val_dataset, num_classes, class_to_label = prepare_embedding_dataset(piece_labels, db, training_session)

        # Create data loaders with smaller batch size for T600
        batch_size = adjust_batch_size_for_stage(device, 'embedding', 8)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        # Initialize embedding network
        model = EmbeddingNetwork(
            num_classes=num_classes, 
            embedding_dim=EMBEDDING_DIM, 
            backbone='efficientnet_b0',  # Smaller backbone for T600
            pretrained=True
        )
        model.to(device)

        # Use ArcFace loss for better embeddings
        criterion = ArcFaceLoss(EMBEDDING_DIM, num_classes)
        criterion.to(device)

        # Optimizer with smaller learning rate
        optimizer = optim.AdamW(
            list(model.parameters()) + list(criterion.parameters()), 
            lr=0.0001, 
            weight_decay=0.01
        )
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

        training_session.add_log("INFO", f"Training embedding network - Classes: {num_classes}, Batch size: {batch_size}")
        safe_commit(db)

        # Training loop
        best_val_loss = float('inf')
        epochs = 50  # Fewer epochs for T600

        for epoch in range(epochs):
            if stop_event.is_set():
                break

            # Training phase
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (images, labels) in enumerate(train_loader):
                if stop_event.is_set():
                    break

                images, labels = images.to(device), labels.to(device)

                optimizer.zero_grad()

                # Forward pass
                logits, embeddings = model(images)
                arcface_logits = criterion(embeddings, labels)

                loss = nn.functional.cross_entropy(arcface_logits, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                trainloss += loss.item()
                _, predicted = arcface_logits.max(1)                    
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                if batch_idx % 10 == 0:
                    training_session.add_log("INFO", 
                        f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
                    safe_commit(db)

            scheduler.step()

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)

                    logits, embeddings = model(images)
                    arcface_logits = criterion(embeddings, labels)

                    loss = nn.functional.cross_entropy(arcface_logits, labels)
                    val_loss += loss.item()

                    _, predicted = arcface_logits.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

            train_acc = 100. * correct / total
            val_acc = 100. * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            training_session.add_log("INFO", 
                f"Epoch {epoch+1}/{epochs}: Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
                f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            safe_commit(db)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                embedding_model_path = os.path.join(models_base_path, "part_embedding_model.pt")
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'criterion_state_dict': criterion.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'class_to_label': class_to_label,
                    'num_classes': num_classes,
                    'embedding_dim': EMBEDDING_DIM
                }, embedding_model_path)

        training_session.add_log("INFO", f"Stage 2 completed: Embedding model saved at {embedding_model_path}")
        safe_commit(db)

        # Build vector database
        build_vector_database(model, piece_labels, db, models_base_path, training_session)

        return embedding_model_path

    except Exception as e:
        logger.error(f"Embedding stage training failed: {str(e)}", exc_info=True)
        training_session.add_log("ERROR", f"Embedding stage failed: {str(e)}")
        safe_commit(db)
        raise

def prepare_embedding_dataset(piece_labels: list, db: Session, training_session):
    """Prepare dataset for embedding training."""

    # Data transforms for embedding training
    train_transform = transforms.Compose([
        transforms.Resize((EMBEDDING_IMAGE_SIZE, EMBEDDING_IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((EMBEDDING_IMAGE_SIZE, EMBEDDING_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    all_image_paths = []
    all_labels = []
    class_to_label = {}

    for class_id, piece_label in enumerate(piece_labels):
        piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
        if not piece or not piece.is_annotated:
            continue

        class_to_label[class_id] = piece_label
        images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).all()

        for image in images:
            all_image_paths.append(image.image_path)
            all_labels.append(class_id)

    # Split dataset
    split_idx = int(len(all_image_paths) * 0.8)
    train_paths = all_image_paths[:split_idx]
    train_labels = all_labels[:split_idx]
    val_paths = all_image_paths[split_idx:]
    val_labels = all_labels[split_idx:]

    train_dataset = AirplanePartDataset(train_paths, train_labels, train_transform)
    val_dataset = AirplanePartDataset(val_paths, val_labels, val_transform)

    training_session.add_log("INFO", 
        f"Prepared embedding dataset: {len(train_dataset)} train, {len(val_dataset)} val samples")
    safe_commit(db)

    return train_dataset, val_dataset, len(piece_labels), class_to_label

def build_vector_database(model, piece_labels: list, db: Session, models_base_path: str, training_session):
    """Build FAISS vector database from trained embeddings."""

    training_session.add_log("INFO", "Building vector database...")
    safe_commit(db)

    device = next(model.parameters()).device
    model.eval()

    # Extract embeddings for each part
    all_embeddings = []
    all_labels = []
    all_metadata = []

    transform = transforms.Compose([
        transforms.Resize((EMBEDDING_IMAGE_SIZE, EMBEDDING_IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    with torch.no_grad():
        for piece_label in piece_labels:
            piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
            if not piece:
                continue

            images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).all()

            # Use a representative image for each part (or average multiple embeddings)
            for image in images[:3]:  # Use up to 3 images per part
                try:
                    img = Image.open(image.image_path).convert('RGB')
                    img_tensor = transform(img).unsqueeze(0).to(device)

                    embedding = model(img_tensor, return_embedding=True)
                    embedding_np = embedding.cpu().numpy().flatten()

                    all_embeddings.append(embedding_np)
                    all_labels.append(piece_label)
                    all_metadata.append({
                        'piece_id': piece.id,
                        'image_path': image.image_path,
                        'class_data_id': piece.class_data_id
                    })

                except Exception as e:
                    logger.warning(f"Failed to process image {image.image_path}: {e}")
                    continue

    # Build FAISS index
    embeddings_array = np.vstack(all_embeddings)

   
    # Normalize embeddings for cosine similarity
    embeddings_array = embeddings_array / np.linalg.norm(embeddings_array, axis=1, keepdims=True)
        
    # Create vector database
    vector_db = VectorDatabase(embedding_dim=EMBEDDING_DIM, index_type='IndexFlatIP')
    vector_db.build_index(embeddings_array, all_labels, 
                         {i: meta for i, meta in enumerate(all_metadata)})
    
    # Save vector database
    vector_db_path = os.path.join(models_base_path, "vector_database")
    vector_db.save_index(vector_db_path)
    
    training_session.add_log("INFO", 
        f"Vector database built with {len(all_embeddings)} embeddings and saved to {vector_db_path}")
    safe_commit(db)
    
    return vector_db_path

def train_hybrid_model(piece_labels: list, db: Session, session_id: int):
    """
    Main training function implementing the two-stage hybrid approach:
    Stage 1: YOLO detection (generic 'part' detection)  
    Stage 2: Embedding network + Vector database (part identification)
    """
    
    try:
        training_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if not training_session:
            logger.error(f"Training session {session_id} not found")
            return

        # Ensure piece_labels is a list
        if isinstance(piece_labels, str):
            piece_labels = [piece_labels]
        
        training_session.add_log("INFO", 
            f"Starting hybrid training for {len(piece_labels)} piece types: {piece_labels}")
        safe_commit(db)

        # Set device and configuration
        device = select_device()
        training_session.device_used = str(device)
        training_session.add_log("INFO", f"Using device: {device}")
        safe_commit(db)

        # Count total images
        if not training_session.total_images:
            total_images = 0
            for piece_label in piece_labels:
                piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
                if piece:
                    images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).all()
                    total_images += len(images)
            training_session.total_images = total_images
            safe_commit(db)

        # Stage 1: Train YOLO detector for generic part detection
        training_session.progress_percentage = 10
        training_session.add_log("INFO", "=== STAGE 1: YOLO DETECTION TRAINING ===")
        safe_commit(db)
        
        detection_model_path = train_detection_stage(piece_labels, db, session_id)
        
        if stop_event.is_set():
            training_session.add_log("INFO", "Training stopped by user during detection stage")
            safe_commit(db)
            return

        # Stage 2: Train embedding network and build vector database
        training_session.progress_percentage = 60
        training_session.add_log("INFO", "=== STAGE 2: EMBEDDING NETWORK TRAINING ===")
        safe_commit(db)
        
        embedding_model_path = train_embedding_stage(piece_labels, db, session_id)
        
        if stop_event.is_set():
            training_session.add_log("INFO", "Training stopped by user during embedding stage")
            safe_commit(db)
            return

        # Mark all pieces as trained
        training_session.progress_percentage = 100
        for piece_label in piece_labels:
            piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
            if piece:
                piece.is_yolo_trained = True
        
        training_session.add_log("INFO", 
            "=== HYBRID TRAINING COMPLETED SUCCESSFULLY ===")
        training_session.add_log("INFO", 
            f"Detection model: {detection_model_path}")
        training_session.add_log("INFO", 
            f"Embedding model: {embedding_model_path}")
        safe_commit(db)
        
        logger.info("Hybrid training completed successfully")
        
    except Exception as e:
        logger.error(f"Hybrid training failed: {str(e)}", exc_info=True)
        training_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if training_session:
            training_session.add_log("ERROR", f"Hybrid training failed: {str(e)}")
            safe_commit(db)
        raise
    finally:
        stop_event.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Training process finished and GPU memory cleared.")

class HybridInferenceEngine:
    """Inference engine that combines YOLO detection + embedding retrieval."""
    
    def __init__(self, detection_model_path: str, embedding_model_path: str, vector_db_path: str):
        self.device = select_device()
        
        # Load YOLO detection model
        self.detection_model = YOLO(detection_model_path)
        self.detection_model.to(self.device)
        
        # Load embedding model
        checkpoint = torch.load(embedding_model_path, map_location=self.device)
        self.num_classes = checkpoint['num_classes']
        self.class_to_label = checkpoint['class_to_label']
        
        self.embedding_model = EmbeddingNetwork(
            num_classes=self.num_classes,
            embedding_dim=checkpoint['embedding_dim'],
            backbone='efficientnet_b0',
            pretrained=False
        )
        self.embedding_model.load_state_dict(checkpoint['model_state_dict'])
        self.embedding_model.to(self.device)
        self.embedding_model.eval()
        
        # Load vector database
        self.vector_db = VectorDatabase(embedding_dim=checkpoint['embedding_dim'])
        self.vector_db.load_index(vector_db_path)
        
        # Embedding transform
        self.embedding_transform = transforms.Compose([
            transforms.Resize((EMBEDDING_IMAGE_SIZE, EMBEDDING_IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Hybrid inference engine initialized successfully")
    
    def predict(self, image_path: str, detection_threshold=0.5, similarity_threshold=0.7, top_k=5):
        """
        Full inference pipeline:
        1. Detect parts using YOLO
        2. Extract embeddings for each detected part  
        3. Search vector database for similar parts
        4. Return combined results
        """
        try:
            # Stage 1: Detect parts
            detection_results = self.detection_model(image_path, conf=detection_threshold)
            
            if len(detection_results[0].boxes) == 0:
                return {"detections": [], "message": "No parts detected"}
            
            # Load original image
            original_image = Image.open(image_path).convert('RGB')
            image_width, image_height = original_image.size
            
            final_results = []
            
            # Stage 2: Process each detection
            for i, box in enumerate(detection_results[0].boxes):
                # Get bounding box coordinates (normalized -> absolute)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                
                # Crop detected part
                cropped_part = original_image.crop((int(x1), int(y1), int(x2), int(y2)))
                
                # Extract embedding
                part_tensor = self.embedding_transform(cropped_part).unsqueeze(0).to(self.device)
                
                with torch.no_grad():
                    embedding = self.embedding_model(part_tensor, return_embedding=True)
                    embedding_np = embedding.cpu().numpy().flatten()
                
                # Search vector database
                search_results = self.vector_db.search(
                    embedding_np, 
                    k=top_k, 
                    threshold=similarity_threshold
                )
                
                # Compile results
                detection_result = {
                    "detection_id": i,
                    "bounding_box": {
                        "x1": float(x1), "y1": float(y1), 
                        "x2": float(x2), "y2": float(y2)
                    },
                    "detection_confidence": float(confidence),
                    "identified_parts": search_results,
                    "best_match": search_results[0] if search_results else None,
                    "is_mismatch": len(search_results) == 0 or (search_results[0]['similarity_score'] < similarity_threshold if search_results else True)
                }
                
                final_results.append(detection_result)
            
            return {
                "total_detections": len(final_results),
                "detections": final_results,
                "image_dimensions": {"width": image_width, "height": image_height}
            }
            
        except Exception as e:
            logger.error(f"Inference failed: {str(e)}", exc_info=True)
            return {"error": str(e), "detections": []}

def validate_and_prepare_pieces(piece_labels: list, db: Session):
    """Validate that all pieces exist and are properly annotated."""
    
    valid_pieces = []
    invalid_pieces = []
    
    for piece_label in piece_labels:
        piece = db.query(Piece).filter(Piece.piece_label == piece_label).first()
        
        if not piece:
            invalid_pieces.append(f"{piece_label}: Piece not found")
            continue
            
        if not piece.is_annotated:
            invalid_pieces.append(f"{piece_label}: Piece not annotated")
            continue
            
        # Check if piece has images
        images = db.query(PieceImage).filter(PieceImage.piece_id == piece.id).all()
        if len(images) < 5:  # Minimum threshold
            invalid_pieces.append(f"{piece_label}: Insufficient images ({len(images)} found, minimum 5 required)")
            continue
            
        # Check if images have annotations
        annotated_images = 0
        for image in images:
            annotations = db.query(Annotation).filter(Annotation.piece_image_id == image.id).all()
            if annotations:
                annotated_images += 1
        
        if annotated_images < 3:  # Minimum annotated images
            invalid_pieces.append(f"{piece_label}: Insufficient annotated images ({annotated_images} found, minimum 3 required)")
            continue
        
        valid_pieces.append(piece_label)
    
    return valid_pieces, invalid_pieces

# Utility functions for the hybrid system
def get_model_info(models_base_path: str):
    """Get information about trained models."""
    
    detection_model_path = os.path.join(models_base_path, "part_detector.pt")
    embedding_model_path = os.path.join(models_base_path, "part_embedding_model.pt")
    vector_db_path = os.path.join(models_base_path, "vector_database")
    
    info = {
        "detection_model": {
            "exists": os.path.exists(detection_model_path),
            "path": detection_model_path,
            "size_mb": os.path.getsize(detection_model_path) / 1024**2 if os.path.exists(detection_model_path) else 0
        },
        "embedding_model": {
            "exists": os.path.exists(embedding_model_path),
            "path": embedding_model_path,
            "size_mb": os.path.getsize(embedding_model_path) / 1024**2 if os.path.exists(embedding_model_path) else 0
        },
        "vector_database": {
            "exists": os.path.exists(vector_db_path),
            "path": vector_db_path,
            "metadata_exists": os.path.exists(os.path.join(vector_db_path, "metadata.json"))
        }
    }
    
    # Load embedding model info if available
    if info["embedding_model"]["exists"]:
        try:
            checkpoint = torch.load(embedding_model_path, map_location='cpu')
            info["embedding_model"]["num_classes"] = checkpoint.get('num_classes', 0)
            info["embedding_model"]["embedding_dim"] = checkpoint.get('embedding_dim', 0)
        except Exception as e:
            info["embedding_model"]["error"] = str(e)
    
    # Load vector database info if available
    if info["vector_database"]["metadata_exists"]:
        try:
            with open(os.path.join(vector_db_path, "metadata.json"), 'r') as f:
                metadata = json.load(f)
            info["vector_database"]["num_embeddings"] = len(metadata.get('part_labels', []))
            info["vector_database"]["part_labels"] = metadata.get('part_labels', [])
        except Exception as e:
            info["vector_database"]["error"] = str(e)
    
    return info

def cleanup_old_models(models_base_path: str, keep_last_n=3):
    """Clean up old model checkpoints to save disk space."""
    
    try:
        # Clean up YOLO training runs
        yolo_runs_path = os.path.join(models_base_path, "part_detection")
        if os.path.exists(yolo_runs_path):
            runs = [d for d in os.listdir(yolo_runs_path) if d.startswith("train")]
            runs.sort(key=lambda x: os.path.getctime(os.path.join(yolo_runs_path, x)), reverse=True)
            
            for run in runs[keep_last_n:]:
                run_path = os.path.join(yolo_runs_path, run)
                shutil.rmtree(run_path)
                logger.info(f"Cleaned up old YOLO run: {run_path}")
        
        logger.info("Model cleanup completed")
        
    except Exception as e:
        logger.error(f"Model cleanup failed: {str(e)}")

# Replace the original train_model function with the hybrid version
def train_model(piece_labels: list, db: Session, session_id: int):
    """
    Entry point for training - now uses hybrid approach.
    This replaces the original YOLO-only training function.
    """
    return train_hybrid_model(piece_labels, db, session_id)