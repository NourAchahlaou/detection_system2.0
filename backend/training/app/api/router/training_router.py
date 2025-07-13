import logging
import threading
import os
from typing import Annotated, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from training.app.request.training_request import TrainRequest
from training.app.services.model_training_service import train_model, stop_training
from training.app.db.session import get_session, create_new_session, safe_commit, safe_close
from training.app.db.models.training import TrainingSession
from datetime import datetime

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


def get_active_training_session(db: Session) -> Optional[TrainingSession]:
    """Get the current active training session from database."""
    try:
        return db.query(TrainingSession).filter(
            TrainingSession.is_training == True
        ).first()
    except Exception as e:
        logger.error(f"Error getting active training session: {str(e)}")
        return None


def find_existing_training_session(db: Session, piece_labels: list) -> Optional[TrainingSession]:
    """Find existing training session for the same piece labels."""
    try:
        # Sort piece_labels to ensure consistent comparison
        sorted_labels = sorted(piece_labels)
        
        # Find sessions with matching piece labels (incomplete sessions)
        sessions = db.query(TrainingSession).filter(
            TrainingSession.is_training == False,
            
           
        ).all()
        
        for session in sessions:
            if session.piece_labels and sorted(session.piece_labels) == sorted_labels:
                return session
        
        return None
    except Exception as e:
        logger.error(f"Error finding existing training session: {str(e)}")
        return None


def create_training_session(db: Session, piece_labels: list) -> TrainingSession:
    """Create a new training session with proper error handling."""
    try:
        session_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        training_session = TrainingSession(
            session_name=session_name,
            model_type="YOLOV8X",
            epochs=25,
            batch_size=8,
            learning_rate=0.0001,
            image_size=640,
            piece_labels=piece_labels,
            is_training=False,
            current_epoch=1  
        )
        
        db.add(training_session)
        if not safe_commit(db):
            raise Exception("Failed to commit training session")
        
        db.refresh(training_session)
        return training_session
        
    except Exception as e:
        logger.error(f"Error creating training session: {str(e)}")
        raise

def resume_training_session(db: Session, session: TrainingSession, piece_labels: list) -> TrainingSession:
    """Resume an existing training session."""
    try:
        # Update session name to indicate resume
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        session.session_name = f"{session.session_name}_resumed_{current_time}"
        
        # Reset training state but keep progress
        session.is_training = False
        session.completed_at = None

        
        # Add log entry about resuming
        session.add_log("INFO", f"Resuming training from epoch {session.current_epoch}")
        
        if not safe_commit(db):
            raise Exception("Failed to update training session for resume")
        
        db.refresh(session)
        return session
        
    except Exception as e:
        logger.error(f"Error resuming training session: {str(e)}")
        raise

@training_router.post("/train")
def train_piece_model(request: TrainRequest, db: db_dependency):
    """
    Start training process for specified piece labels.
    If a previous incomplete session exists for the same pieces, resume from where it left off.
    
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
        active_session = get_active_training_session(db)
        if active_session:
            raise HTTPException(
                status_code=409, 
                detail=f"Training session '{active_session.session_name}' is already in progress. Please stop the current training before starting a new one."
            )
        
        # Look for existing incomplete session with same piece labels
        existing_session = find_existing_training_session(db, request.piece_labels)
        
        if existing_session:
            # Resume existing session
            training_session = resume_training_session(db, existing_session, request.piece_labels)
            logger.info(f"Resuming training session: {training_session.session_name} with ID: {training_session.id} from epoch {training_session.current_epoch}")
            
            action = "resumed"
            is_resume = True
            resume_from_epoch = training_session.current_epoch
        else:
            # Create new training session
            training_session = create_training_session(db, request.piece_labels)
            logger.info(f"Created new training session: {training_session.session_name} with ID: {training_session.id}")
            
            action = "created"
            is_resume = False
            resume_from_epoch = 1
        
        # Start training in a separate thread to avoid blocking
        training_thread = threading.Thread(
            target=train_model_with_status_updates,
            args=(request.piece_labels, training_session.id, is_resume),
            daemon=True
        )
        training_thread.start()
        
        logger.info("Training thread started successfully")
        
        return {
            "message": f"Training process {action} successfully", 
            "piece_labels": request.piece_labels,
            "session_id": training_session.id,
            "session_name": training_session.session_name,
            "status": "initiated",
            "action": action,
            "resume_from_epoch": resume_from_epoch
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training initiation failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")



def train_model_with_status_updates(piece_labels: list, session_id: int, is_resume: bool = False):
    """
    Wrapper function that calls the actual training function while updating status.
    Creates its own database session to avoid concurrency issues.
    """
    # Create a new session for this thread
    db = create_new_session()
    
    try:
        # Get the training session
        training_session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if not training_session:
            logger.error(f"Training session {session_id} not found")
            return
        
        # Start the training session (this will set is_training=True)
        training_session.start_training(piece_labels, is_resume=is_resume)
        if not safe_commit(db):
            logger.error("Failed to commit training session start")
            return
        
        # Call the actual training function with the new session
        train_model(piece_labels, db, session_id)
        
        # Mark training as completed
        # Refresh the session to get latest state
        db.refresh(training_session)
        training_session.complete_training()
        if not safe_commit(db):
            logger.error("Failed to commit training completion")
        
        logger.info(f"Training completed successfully for session {session_id}")
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}", exc_info=True)
        
        # Mark training as failed
        try:
            db.rollback()  # Rollback any pending changes
        except Exception as update_error:
            logger.error(f"Failed to update training session with error: {str(update_error)}")
    
    finally:
        # Always close the session
        safe_close(db)


@training_router.post("/stop_training")
async def stop_training_yolo(db: db_dependency):
    """
    Stop the currently running training process.
    
    Returns:
        Success message confirming stop signal was sent
    """
    try:
        logger.info("Received stop training request")
        
        # Check database for active session
        active_session = get_active_training_session(db)
        if not active_session:
            raise HTTPException(status_code=400, detail="No training is currently in progress")
        
        # Send stop signal to training service
        await stop_training()
        
        # Update training session as stopped
        active_session.stop_training()
        if not safe_commit(db):
            logger.error("Failed to commit training stop")
        
        logger.info("Stop training signal sent successfully")
        
        return {
            "message": "Stop training signal sent.",
            "session_id": active_session.id,
            "session_name": active_session.session_name,
            "status": active_session.get_status()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop training: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@training_router.get("/status")
def get_training_status(db: db_dependency):
    """
    Get the current training status.
    
    Returns:
        Current training status including progress, logs, and metrics
    """
    try:
        # Get active session from database
        active_session = get_active_training_session(db)
        
        if not active_session:
            return {
                "status": "success",
                "data": {
                    "is_training": False,
                    "session_info": None,
                    "message": "No active training session"
                }
            }
        
        return {
            "status": "success",
            "data": {
                "is_training": active_session.is_training,
                "session_info": active_session.to_dict(),
                "logs": active_session.get_recent_logs(50)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get training status: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@training_router.get("/health")
def health_check(db: db_dependency):
    """Health check endpoint for the training service."""
    try:
        active_session = get_active_training_session(db)
        return {
            "status": "healthy", 
            "service": "training",
            "is_training": active_session is not None,
            "active_session_id": active_session.id if active_session else None,
            "last_updated": active_session.last_updated.isoformat() if active_session else None
        }
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return {
            "status": "unhealthy",
            "service": "training",
            "error": str(e)
        }


@training_router.get("/sessions")
def get_training_sessions(db: db_dependency, limit: Optional[int] = 10):
    """
    Get training sessions history.
    
    Args:
        limit: Maximum number of sessions to return
        
    Returns:
        List of training sessions
    """
    try:
        sessions = db.query(TrainingSession).order_by(TrainingSession.started_at.desc()).limit(limit).all()
        
        sessions_data = [session.to_dict() for session in sessions]
        
        return {
            "status": "success",
            "data": {
                "sessions": sessions_data,
                "total_count": len(sessions_data)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get training sessions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@training_router.get("/logs")
def get_training_logs(db: db_dependency, limit: Optional[int] = 50):
    """
    Get training logs with optional limit.
    
    Args:
        limit: Maximum number of log entries to return
        
    Returns:
        Training logs array
    """
    try:
        active_session = get_active_training_session(db)
        
        if not active_session:
            return {
                "status": "success",
                "data": {
                    "logs": [],
                    "total_count": 0,
                    "is_training": False,
                    "session_id": None
                }
            }
        
        logs = active_session.get_recent_logs(limit)
        
        return {
            "status": "success",
            "data": {
                "logs": logs,
                "total_count": len(active_session.training_logs or []),
                "is_training": active_session.is_training,
                "session_id": active_session.id
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get training logs: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@training_router.put("/progress")
def update_training_progress(progress_data: dict, db: db_dependency):
    """
    Update training progress. This endpoint can be called by the training service
    to update real-time progress information.
    
    Args:
        progress_data: Dictionary containing progress information
        db: Database session
        
    Returns:
        Success confirmation
    """
    try:
        # Check if training is in progress
        active_session = get_active_training_session(db)
        if not active_session:
            raise HTTPException(status_code=400, detail="No training is currently in progress")
        
        # Update progress using the entity methods
        if "current_epoch" in progress_data:
            active_session.current_epoch = progress_data["current_epoch"]
        
        if "progress_percentage" in progress_data:
            active_session.progress_percentage = progress_data["progress_percentage"]
        
        if "device" in progress_data and not active_session.device_used:
            active_session.device_used = progress_data["device"]
        
        if "total_images" in progress_data:
            active_session.total_images = progress_data["total_images"]
        
        if "augmented_images" in progress_data:
            active_session.augmented_images = progress_data["augmented_images"]
        
        if "validation_images" in progress_data:
            active_session.validation_images = progress_data["validation_images"]
        
        # Update losses if provided
        if "losses" in progress_data:
            losses = progress_data["losses"]
            active_session.update_losses(
                box_loss=losses.get("box_loss"),
                cls_loss=losses.get("cls_loss"),
                dfl_loss=losses.get("dfl_loss")
            )
        
        # Update metrics if provided
        if "metrics" in progress_data:
            metrics = progress_data["metrics"]
            active_session.update_metrics(
                instances=metrics.get("instances"),
                lr=metrics.get("lr"),
                momentum=metrics.get("momentum")
            )
        
        # Add log entry if message is provided
        if "message" in progress_data:
            log_level = progress_data.get("log_level", "INFO")
            active_session.add_log(log_level, progress_data["message"])
        
        # Update timestamp
        active_session.last_updated = datetime.utcnow()
        
        if not safe_commit(db):
            raise HTTPException(status_code=500, detail="Failed to update training progress")
        
        return {
            "status": "success",
            "message": "Training progress updated successfully",
            "session_id": active_session.id,
            "current_status": active_session.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update training progress: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@training_router.get("/session/{session_id}")
def get_training_session(session_id: int, db: db_dependency):
    """
    Get specific training session details.
    
    Args:
        session_id: ID of the training session
        db: Database session
        
    Returns:
        Training session details
    """
    try:
        session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Training session not found")
        
        return {
            "status": "success",
            "data": session.to_dict()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training session: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@training_router.delete("/session/{session_id}")
def delete_training_session(session_id: int, db: db_dependency):
    """
    Delete a specific training session.
    
    Args:
        session_id: ID of the training session to delete
        db: Database session
        
    Returns:
        Success confirmation
    """
    try:
        session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        
        if not session:
            raise HTTPException(status_code=404, detail="Training session not found")
        
        if session.is_training:
            raise HTTPException(status_code=400, detail="Cannot delete an active training session")
        
        db.delete(session)
        if not safe_commit(db):
            raise HTTPException(status_code=500, detail="Failed to delete training session")
        
        return {
            "status": "success",
            "message": f"Training session {session_id} deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete training session: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")


@training_router.get("/resumable")
def get_resumable_sessions(db: db_dependency):
    """
    Get list of training sessions that can be resumed.
    
    Returns:
        List of incomplete training sessions
    """
    try:
        # Find sessions that are not completed, failed, or currently running
        resumable_sessions = db.query(TrainingSession).filter(
            TrainingSession.is_training == False,
            TrainingSession.completed_at.is_(None),
            TrainingSession.failed_at.is_(None)
        ).order_by(TrainingSession.started_at.desc()).all()
        
        sessions_data = [session.to_dict() for session in resumable_sessions]
        
        return {
            "status": "success",
            "data": {
                "resumable_sessions": sessions_data,
                "total_count": len(sessions_data)
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get resumable sessions: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")