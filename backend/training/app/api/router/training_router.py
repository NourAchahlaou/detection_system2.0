import logging
import threading
import os
from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from training.app.request.training_request import TrainRequest
from training.app.services.model_training_service import train_model, stop_training
from training.app.db.session import get_session

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
        
        # Start training in a separate thread to avoid blocking
        training_thread = threading.Thread(
            target=train_model,
            args=(request.piece_labels, db),
            daemon=True
        )
        training_thread.start()
        
        logger.info("Training thread started successfully")
        return {
            "message": "Training process started successfully", 
            "piece_labels": request.piece_labels,
            "status": "initiated"
        }
        
    except Exception as e:
        logger.error(f"Training initiation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@training_router.post("/stop_training")
async def stop_training_yolo():
    """
    Stop the currently running training process.
    
    Returns:
        Success message confirming stop signal was sent
    """
    try:
        logger.info("Received stop training request")
        await stop_training()
        logger.info("Stop training signal sent successfully")
        return {"message": "Stop training signal sent."}
    except Exception as e:
        logger.error(f"Failed to stop training: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@training_router.get("/health")
def health_check():
    """Health check endpoint for the training service."""
    return {"status": "healthy", "service": "training"}