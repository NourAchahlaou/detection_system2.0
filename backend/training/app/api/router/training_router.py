import logging
import threading
import os
from typing import Annotated, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from training.app.request.training_request import TrainRequest
from training.app.services.model_training_service import train_model, stop_training
from training.app.db.session import get_session
from datetime import datetime
import json

# Professional logging setup with volume mounting
log_dir = os.getenv("LOG_PATH", "/usr/srv/logs")
log_file = os.path.join(log_dir, "training.log")

# Ensure log directory exists with proper permissions
os.makedirs(log_dir, exist_ok=True)

# Configure logging with rotation for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler(log_file, mode='a', encoding='utf-8')  # File output
    ]
)

logger = logging.getLogger(__name__)

training_router = APIRouter(
    prefix="/training",
    tags=["Training"],
    responses={404: {"description": "Not found"}},
)

db_dependency = Annotated[Session, Depends(get_session)]
stop_event = threading.Event()

# Global training status tracking
training_status = {
    "is_training": False,
    "current_pieces": [],
    "start_time": None,
    "current_epoch": 0,
    "total_epochs": 25,
    "progress": 0,
    "batch_size": 4,
    "image_size": 640,
    "device": "cpu",
    "total_images": 0,
    "augmented_images": 0,
    "validation_images": 0,
    "losses": {
        "box_loss": 0.0,
        "cls_loss": 0.0,
        "dfl_loss": 0.0,
    },
    "metrics": {
        "instances": 0,
        "lr": 0.002,
        "momentum": 0.9,
    },
    "logs": [],
    "last_updated": None
}

def update_training_status(status_update: Dict[str, Any]):
    """Update the global training status with new information."""
    global training_status
    training_status.update(status_update)
    training_status["last_updated"] = datetime.now().isoformat()
    logger.info(f"Training status updated: {status_update}")

def add_training_log(level: str, message: str):
    """Add a new log entry to the training status."""
    global training_status
    log_entry = {
        "level": level,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    training_status["logs"].append(log_entry)
    # Keep only the last 100 log entries to prevent memory issues
    if len(training_status["logs"]) > 100:
        training_status["logs"] = training_status["logs"][-100:]
    training_status["last_updated"] = datetime.now().isoformat()

def reset_training_status():
    """Reset training status to initial state."""
    global training_status
    training_status.update({
        "is_training": False,
        "current_pieces": [],
        "start_time": None,
        "current_epoch": 0,
        "total_epochs": 25,
        "progress": 0,
        "batch_size": 4,
        "image_size": 640,
        "device": "cpu",
        "total_images": 0,
        "augmented_images": 0,
        "validation_images": 0,
        "losses": {
            "box_loss": 0.0,
            "cls_loss": 0.0,
            "dfl_loss": 0.0,
        },
        "metrics": {
            "instances": 0,
            "lr": 0.002,
            "momentum": 0.9,
        },
        "logs": [],
        "last_updated": datetime.now().isoformat()
    })

@training_router.post("/train")
def train_piece_model(request: TrainRequest, db: db_dependency):
    """
    Start training process for specified piece labels.
    
    Args:
        request: Training request containing piece labels
        db: Database session
        
    Returns:
        Success message with training initiation confirmation
    """
    try:
        logger.info(f"Starting training for piece labels: {request.piece_labels}")
        
        # Validate that piece_labels is not empty
        if not request.piece_labels:
            raise HTTPException(status_code=400, detail="piece_labels cannot be empty")
        
        # Check if training is already in progress
        if training_status["is_training"]:
            raise HTTPException(
                status_code=409, 
                detail="Training is already in progress. Please stop the current training before starting a new one."
            )
        
        # Initialize training status
        update_training_status({
            "is_training": True,
            "current_pieces": request.piece_labels,
            "start_time": datetime.now().isoformat(),
            "current_epoch": 1,
            "progress": 4,
            "logs": []
        })
        
        add_training_log("INFO", f"Starting training process for pieces: {', '.join(request.piece_labels)}")
        add_training_log("INFO", f"Training initiated at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Start training in a separate thread to avoid blocking
        training_thread = threading.Thread(
            target=train_model_with_status_updates,
            args=(request.piece_labels, db),
            daemon=True
        )
        training_thread.start()
        
        logger.info("Training thread started successfully")
        return {
            "message": "Training process started successfully", 
            "piece_labels": request.piece_labels,
            "status": "initiated",
            "training_status": training_status
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training initiation failed: {str(e)}", exc_info=True)
        reset_training_status()
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

def train_model_with_status_updates(piece_labels: list, db: Session):
    """
    Wrapper function that calls the actual training function while updating status.
    """
    try:
        # Add initial logs
        add_training_log("INFO", f"Training started for {len(piece_labels)} piece(s)")
        
        # Call the actual training function
        train_model(piece_labels, db)
        
        # Update final status
        update_training_status({
            "is_training": False,
            "progress": 100,
            "current_epoch": training_status["total_epochs"]
        })
        add_training_log("SUCCESS", "Training completed successfully")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        update_training_status({
            "is_training": False,
            "progress": 0
        })
        add_training_log("ERROR", f"Training failed: {str(e)}")

@training_router.post("/stop_training")
async def stop_training_yolo():
    """
    Stop the currently running training process.
    
    Returns:
        Success message confirming stop signal was sent
    """
    try:
        logger.info("Received stop training request")
        
        if not training_status["is_training"]:
            raise HTTPException(status_code=400, detail="No training is currently in progress")
        
        await stop_training()
        
        # Update training status
        update_training_status({
            "is_training": False,
            "progress": 0
        })
        add_training_log("INFO", "Training stopped by user request")
        
        logger.info("Stop training signal sent successfully")
        return {
            "message": "Stop training signal sent.",
            "training_status": training_status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop training: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@training_router.get("/status")
def get_training_status():
    """
    Get the current training status.
    
    Returns:
        Current training status including progress, logs, and metrics
    """
    try:
        return {
            "status": "success",
            "data": training_status
        }
    except Exception as e:
        logger.error(f"Failed to get training status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@training_router.get("/health")
def health_check():
    """Health check endpoint for the training service."""
    return {
        "status": "healthy", 
        "service": "training",
        "is_training": training_status["is_training"],
        "last_updated": training_status["last_updated"]
    }

# Additional endpoint to get training history/logs
@training_router.get("/logs")
def get_training_logs(limit: Optional[int] = 50):
    """
    Get training logs with optional limit.
    
    Args:
        limit: Maximum number of log entries to return
        
    Returns:
        Training logs array
    """
    try:
        logs = training_status["logs"]
        if limit and len(logs) > limit:
            logs = logs[-limit:]
        
        return {
            "status": "success",
            "data": {
                "logs": logs,
                "total_count": len(training_status["logs"]),
                "is_training": training_status["is_training"]
            }
        }
    except Exception as e:
        logger.error(f"Failed to get training logs: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

# Endpoint to update training progress (called by the training service)
@training_router.put("/progress")
def update_training_progress(progress_data: dict):
    """
    Update training progress. This endpoint can be called by the training service
    to update real-time progress information.
    
    Args:
        progress_data: Dictionary containing progress information
        
    Returns:
        Success confirmation
    """
    try:
        if not training_status["is_training"]:
            raise HTTPException(status_code=400, detail="No training is currently in progress")
        
        # Update the training status with new progress data
        update_training_status(progress_data)
        
        # Add log entry if message is provided
        if "message" in progress_data:
            add_training_log("INFO", progress_data["message"])
        
        return {
            "status": "success",
            "message": "Training progress updated successfully",
            "current_status": training_status
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update training progress: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")