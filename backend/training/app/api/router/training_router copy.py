import logging
import threading
import os
from typing import Annotated, Optional, Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, Query, BackgroundTasks
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError
from training.app.request.training_request import TrainRequest
from training.app.services.model_training_service import train_model, stop_training
from training.app.db.session import get_session, create_new_session, safe_commit, safe_close
from training.app.db.models.training import TrainingSession
from datetime import datetime
from enum import Enum
import asyncio
from datetime import timedelta
# Professional logging setup with volume mounting
log_dir = os.getenv("LOG_PATH", "/usr/srv/logs")
log_file = os.path.join(log_dir, "training.log")

# Ensure log directory exists with proper permissions
os.makedirs(log_dir, exist_ok=True)

# Configure logging with rotation for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    handlers=[
        logging.StreamHandler(),  # Console output
        logging.FileHandler(log_file, mode='a', encoding='utf-8')  # File output
    ]
)

logger = logging.getLogger(__name__)

# Response status enums for consistency
class ResponseStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"

class TrainingStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"
    PAUSED = "paused"

training_router = APIRouter(
    prefix="/training",
    tags=["Training"],
    responses={
        404: {"description": "Resource not found"},
        409: {"description": "Conflict - Training already in progress"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"}
    },
)

db_dependency = Annotated[Session, Depends(get_session)]


# ==================== UTILITY FUNCTIONS ====================

def create_error_response(message: str, details: Any = None, status_code: int = 500) -> Dict[str, Any]:
    """Create standardized error response."""
    response = {
        "status": ResponseStatus.ERROR,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    if details:
        response["details"] = details
    return response


def create_success_response(data: Any = None, message: str = "Success") -> Dict[str, Any]:
    """Create standardized success response."""
    response = {
        "status": ResponseStatus.SUCCESS,
        "message": message,
        "timestamp": datetime.utcnow().isoformat()
    }
    if data is not None:
        response["data"] = data
    return response


def get_active_training_session(db: Session) -> Optional[TrainingSession]:
    """Get the current active training session from database with error handling."""
    try:
        return db.query(TrainingSession).filter(
            TrainingSession.is_training == True
        ).first()
    except SQLAlchemyError as e:
        logger.error(f"Database error getting active training session: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error occurred")
    except Exception as e:
        logger.error(f"Unexpected error getting active training session: {str(e)}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


def find_existing_training_session(db: Session, piece_labels: List[str]) -> Optional[TrainingSession]:
    """Find existing training session for the same piece labels with improved logic."""
    try:
        if not piece_labels:
            return None
            
        sorted_labels = sorted(piece_labels)
        
        # Find incomplete sessions (not training, not completed, not failed)
        sessions = db.query(TrainingSession).filter(
            TrainingSession.is_training == False,
            TrainingSession.completed_at.is_(None),
            TrainingSession.failed_at.is_(None)
        ).order_by(TrainingSession.started_at.desc()).all()
        
        for session in sessions:
            if session.piece_labels and sorted(session.piece_labels) == sorted_labels:
                return session
        
        return None
    except SQLAlchemyError as e:
        logger.error(f"Database error finding existing training session: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error occurred")
    except Exception as e:
        logger.error(f"Error finding existing training session: {str(e)}")
        return None


def create_training_session(db: Session, piece_labels: List[str]) -> TrainingSession:
    """Create a new training session with comprehensive error handling."""
    try:
        session_name = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        training_session = TrainingSession(
            session_name=session_name,
            model_type="yolov8n",
            epochs=30,
            batch_size=8,
            learning_rate=0.0001,
            image_size=640,
            piece_labels=piece_labels,
            is_training=False,
            current_epoch=1
        )
        
        db.add(training_session)
        if not safe_commit(db):
            raise HTTPException(status_code=500, detail="Failed to create training session")
        
        db.refresh(training_session)
        logger.info(f"Created training session: {session_name} (ID: {training_session.id})")
        return training_session
        
    except SQLAlchemyError as e:
        logger.error(f"Database error creating training session: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error occurred")
    except Exception as e:
        logger.error(f"Error creating training session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to create training session: {str(e)}")


def resume_training_session(db: Session, session: TrainingSession, piece_labels: List[str]) -> TrainingSession:
    """Resume an existing training session with improved error handling."""
    try:
        current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_name = session.session_name
        session.session_name = f"{original_name}_resumed_{current_time}"
        
        # Reset training state but keep progress
        session.is_training = False
        session.completed_at = None
        session.failed_at = None
        
        # Add log entry about resuming
        resume_message = f"Resuming training from epoch {session.current_epoch}"
        session.add_log("INFO", resume_message)
        
        if not safe_commit(db):
            raise HTTPException(status_code=500, detail="Failed to resume training session")
        
        db.refresh(session)
        logger.info(f"Resumed training session: {original_name} -> {session.session_name}")
        return session
        
    except SQLAlchemyError as e:
        logger.error(f"Database error resuming training session: {str(e)}")
        raise HTTPException(status_code=500, detail="Database error occurred")
    except Exception as e:
        logger.error(f"Error resuming training session: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to resume training session: {str(e)}")


# ==================== TRAINING ENDPOINTS ====================

@training_router.post("/train")
async def train_piece_model(request: TrainRequest, db: db_dependency, background_tasks: BackgroundTasks):
    """
    Start training process for specified piece labels.
    Enhanced with better error handling and background task management.
    """
    try:
        logger.info(f"Training request received for pieces: {request.piece_labels}")
        
        # Validate input
        if not request.piece_labels or len(request.piece_labels) == 0:
            raise HTTPException(
                status_code=422, 
                detail="piece_labels cannot be empty"
            )
        
        if len(request.piece_labels) > 50:  # Reasonable limit
            raise HTTPException(
                status_code=422,
                detail="Too many pieces specified. Maximum 50 pieces allowed per training session."
            )
        
        # Check for active training
        active_session = get_active_training_session(db)
        if active_session:
            raise HTTPException(
                status_code=409,
                detail={
                    "message": "Training session already in progress",
                    "active_session": {
                        "id": active_session.id,
                        "name": active_session.session_name,
                        "started_at": active_session.started_at.isoformat(),
                        "current_epoch": active_session.current_epoch,
                        "progress_percentage": active_session.progress_percentage
                    }
                }
            )
        
        # Look for resumable session
        existing_session = find_existing_training_session(db, request.piece_labels)
        
        if existing_session:
            training_session = resume_training_session(db, existing_session, request.piece_labels)
            action = "resumed"
            is_resume = True
            start_epoch = training_session.current_epoch
        else:
            training_session = create_training_session(db, request.piece_labels)
            action = "created"
            is_resume = False
            start_epoch = 1
        
        # Start training as background task
        background_tasks.add_task(
            train_model_with_status_updates,
            request.piece_labels,
            training_session.id,
            is_resume
        )
        
        response_data = {
            "session_id": training_session.id,
            "session_name": training_session.session_name,
            "piece_labels": request.piece_labels,
            "action": action,
            "is_resume": is_resume,
            "start_epoch": start_epoch,
            "total_epochs": training_session.epochs,
            "model_type": training_session.model_type,
            "status": "initiated"
        }
        
        return create_success_response(
            data=response_data,
            message=f"Training process {action} successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Training initiation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Training initiation failed",
                details=str(e)
            )
        )


async def train_model_with_status_updates(piece_labels: List[str], session_id: int, is_resume: bool = False):
    """
    Enhanced wrapper function for training with better error handling and async support.
    """
    db = None
    try:
        db = create_new_session()
        
        # Get training session
        training_session = db.query(TrainingSession).filter(
            TrainingSession.id == session_id
        ).first()
        
        if not training_session:
            logger.error(f"Training session {session_id} not found")
            return
        
        # Start training session
        training_session.start_training(piece_labels, is_resume=is_resume)
        if not safe_commit(db):
            logger.error("Failed to start training session")
            return
        
        logger.info(f"Starting training for session {session_id} ({'resume' if is_resume else 'new'})")
        
        # Execute training
        await asyncio.to_thread(train_model, piece_labels, db, session_id)
        
        # Mark as completed
        db.refresh(training_session)
        training_session.complete_training()
        if not safe_commit(db):
            logger.error("Failed to mark training as completed")
        
        logger.info(f"Training completed successfully for session {session_id}")
        
    except Exception as e:
        logger.error(f"Training failed for session {session_id}: {str(e)}", exc_info=True)
        
        # Mark as failed
        if db:
            try:
                training_session = db.query(TrainingSession).filter(
                    TrainingSession.id == session_id
                ).first()
                if training_session:
                    training_session.fail_training(str(e))
                    safe_commit(db)
            except Exception as update_error:
                logger.error(f"Failed to update training session with error: {str(update_error)}")
    
    finally:
        if db:
            safe_close(db)


@training_router.post("/stop")
async def stop_training_process(db: db_dependency):
    """
    Stop the currently running training process with enhanced error handling.
    """
    try:
        logger.info("Stop training request received")
        
        active_session = get_active_training_session(db)
        if not active_session:
            return JSONResponse(
                status_code=400,
                content=create_error_response(
                    "No training is currently in progress",
                    status_code=400
                )
            )
        
        # Send stop signal
        await stop_training()
        
        # Update session status
        active_session.stop_training()
        if not safe_commit(db):
            logger.warning("Failed to update training session stop status")
        
        logger.info(f"Training stopped for session {active_session.id}")
        
        return create_success_response(
            data={
                "session_id": active_session.id,
                "session_name": active_session.session_name,
                "stopped_at": datetime.utcnow().isoformat(),
                "final_epoch": active_session.current_epoch,
                "progress_percentage": active_session.progress_percentage
            },
            message="Training stopped successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to stop training: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Failed to stop training",
                details=str(e)
            )
        )


@training_router.get("/status")
def get_training_status(db: db_dependency, include_logs: bool = Query(False, description="Include recent logs in response")):
    """
    Get comprehensive training status with optional logs.
    """
    try:
        active_session = get_active_training_session(db)
        
        if not active_session:
            return create_success_response(
                data={
                    "is_training": False,
                    "status": TrainingStatus.IDLE,
                    "session_info": None,
                    "message": "No active training session"
                }
            )
        
        session_data = active_session.to_dict()
        if include_logs:
            session_data["recent_logs"] = active_session.get_recent_logs(50)
        
        return create_success_response(
            data={
                "is_training": active_session.is_training,
                "status": TrainingStatus.RUNNING if active_session.is_training else TrainingStatus.PAUSED,
                "session_info": session_data
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get training status: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Failed to get training status",
                details=str(e)
            )
        )


# ==================== SESSION MANAGEMENT ====================

@training_router.get("/sessions")
def get_training_sessions(
    db: db_dependency,
    limit: Optional[int] = Query(10, ge=1, le=100, description="Number of sessions to return"),
    offset: Optional[int] = Query(0, ge=0, description="Number of sessions to skip"),
    status_filter: Optional[str] = Query(None, description="Filter by session status")
):
    """
    Get training sessions with pagination and filtering.
    """
    try:
        query = db.query(TrainingSession).order_by(TrainingSession.started_at.desc())
        
        # Apply status filter if provided
        if status_filter:
            if status_filter.lower() == "active":
                query = query.filter(TrainingSession.is_training == True)
            elif status_filter.lower() == "completed":
                query = query.filter(TrainingSession.completed_at.is_not(None))
            elif status_filter.lower() == "failed":
                query = query.filter(TrainingSession.failed_at.is_not(None))
            elif status_filter.lower() == "resumable":
                query = query.filter(
                    TrainingSession.is_training == False,
                    TrainingSession.completed_at.is_(None),
                    TrainingSession.failed_at.is_(None)
                )
        
        # Get total count before pagination
        total_count = query.count()
        
        # Apply pagination
        sessions = query.offset(offset).limit(limit).all()
        sessions_data = [session.to_dict() for session in sessions]
        
        return create_success_response(
            data={
                "sessions": sessions_data,
                "pagination": {
                    "total_count": total_count,
                    "limit": limit,
                    "offset": offset,
                    "has_more": (offset + limit) < total_count
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get training sessions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Failed to get training sessions",
                details=str(e)
            )
        )


@training_router.get("/session/{session_id}")
def get_training_session_details(session_id: int, db: db_dependency, include_logs: bool = Query(True)):
    """
    Get detailed information about a specific training session.
    """
    try:
        session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=create_error_response(
                    f"Training session {session_id} not found",
                    status_code=404
                )
            )
        
        session_data = session.to_dict()
        if include_logs:
            session_data["logs"] = session.get_recent_logs(100)
        
        return create_success_response(data=session_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training session {session_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Failed to get training session",
                details=str(e)
            )
        )


@training_router.delete("/session/{session_id}")
def delete_training_session(session_id: int, db: db_dependency, force: bool = Query(False)):
    """
    Delete a training session with safety checks.
    """
    try:
        session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=create_error_response(
                    f"Training session {session_id} not found",
                    status_code=404
                )
            )
        
        if session.is_training and not force:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "Cannot delete active training session. Use force=true to override or stop training first.",
                    status_code=400
                )
            )
        
        session_name = session.session_name
        db.delete(session)
        
        if not safe_commit(db):
            raise HTTPException(status_code=500, detail="Failed to delete training session")
        
        logger.info(f"Deleted training session: {session_name} (ID: {session_id})")
        
        return create_success_response(
            message=f"Training session '{session_name}' deleted successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete training session {session_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Failed to delete training session",
                details=str(e)
            )
        )


# ==================== MONITORING & UTILITIES ====================

@training_router.get("/health")
def health_check(db: db_dependency):
    """Comprehensive health check for training service."""
    try:
        active_session = get_active_training_session(db)
        
        # Test database connectivity
        db.execute("SELECT 1")
        
        health_data = {
            "service": "training",
            "database": "healthy",
            "is_training": active_session is not None,
            "uptime": datetime.utcnow().isoformat(),
            "log_file": log_file,
            "log_dir_writable": os.access(log_dir, os.W_OK)
        }
        
        if active_session:
            health_data.update({
                "active_session_id": active_session.id,
                "session_name": active_session.session_name,
                "current_epoch": active_session.current_epoch,
                "progress_percentage": active_session.progress_percentage,
                "last_updated": active_session.last_updated.isoformat() if active_session.last_updated else None
            })
        
        return create_success_response(
            data=health_data,
            message="Service is healthy"
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "training",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
        )


@training_router.get("/logs")
def get_training_logs(
    db: db_dependency,
    session_id: Optional[int] = Query(None, description="Get logs for specific session"),
    limit: Optional[int] = Query(50, ge=1, le=1000, description="Number of log entries"),
    level: Optional[str] = Query(None, description="Filter by log level (INFO, WARNING, ERROR)")
):
    """
    Get training logs with filtering options.
    """
    try:
        if session_id:
            # Get logs for specific session
            session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
            if not session:
                raise HTTPException(
                    status_code=404,
                    detail=create_error_response(
                        f"Training session {session_id} not found",
                        status_code=404
                    )
                )
            
            all_logs = session.training_logs or []
            session_info = {
                "id": session.id,
                "name": session.session_name,
                "is_training": session.is_training
            }
        else:
            # Get logs from active session
            active_session = get_active_training_session(db)
            if not active_session:
                return create_success_response(
                    data={
                        "logs": [],
                        "total_count": 0,
                        "session_info": None
                    },
                    message="No active training session"
                )
            
            all_logs = active_session.training_logs or []
            session_info = {
                "id": active_session.id,
                "name": active_session.session_name,
                "is_training": active_session.is_training
            }
        
        # Filter by level if specified
        if level:
            filtered_logs = [log for log in all_logs if log.get("level", "").upper() == level.upper()]
        else:
            filtered_logs = all_logs
        
        # Apply limit
        recent_logs = filtered_logs[-limit:] if len(filtered_logs) > limit else filtered_logs
        
        return create_success_response(
            data={
                "logs": recent_logs,
                "total_count": len(all_logs),
                "filtered_count": len(filtered_logs),
                "session_info": session_info,
                "filters": {
                    "level": level,
                    "limit": limit
                }
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get training logs: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Failed to get training logs",
                details=str(e)
            )
        )


@training_router.put("/progress")
def update_training_progress(progress_data: Dict[str, Any], db: db_dependency):
    """
    Update training progress with comprehensive validation and error handling.
    """
    try:
        active_session = get_active_training_session(db)
        if not active_session:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "No training is currently in progress",
                    status_code=400
                )
            )
        
        # Validate and update progress fields
        updates_made = []
        
        if "current_epoch" in progress_data:
            epoch = progress_data["current_epoch"]
            if isinstance(epoch, int) and epoch > 0:
                active_session.current_epoch = epoch
                updates_made.append(f"epoch: {epoch}")
        
        if "progress_percentage" in progress_data:
            progress = progress_data["progress_percentage"]
            if isinstance(progress, (int, float)) and 0 <= progress <= 100:
                active_session.progress_percentage = progress
                updates_made.append(f"progress: {progress}%")
        
        if "device" in progress_data and not active_session.device_used:
            active_session.device_used = progress_data["device"]
            updates_made.append(f"device: {progress_data['device']}")
        
        # Update image counts
        for field in ["total_images", "augmented_images", "validation_images"]:
            if field in progress_data:
                value = progress_data[field]
                if isinstance(value, int) and value >= 0:
                    setattr(active_session, field, value)
                    updates_made.append(f"{field}: {value}")
        
        # Update losses
        if "losses" in progress_data:
            losses = progress_data["losses"]
            if isinstance(losses, dict):
                active_session.update_losses(
                    box_loss=losses.get("box_loss"),
                    cls_loss=losses.get("cls_loss"),
                    dfl_loss=losses.get("dfl_loss")
                )
                updates_made.append("losses updated")
        
        # Update metrics
        if "metrics" in progress_data:
            metrics = progress_data["metrics"]
            if isinstance(metrics, dict):
                active_session.update_metrics(
                    instances=metrics.get("instances"),
                    lr=metrics.get("lr"),
                    momentum=metrics.get("momentum")
                )
                updates_made.append("metrics updated")
        
        # Add log entry
        if "message" in progress_data:
            log_level = progress_data.get("log_level", "INFO")
            active_session.add_log(log_level, progress_data["message"])
            updates_made.append("log entry added")
        
        # Update timestamp
        active_session.last_updated = datetime.utcnow()
        
        if not safe_commit(db):
            raise HTTPException(status_code=500, detail="Failed to update training progress")
        
        return create_success_response(
            data={
                "session_id": active_session.id,
                "updates_made": updates_made,
                "current_status": {
                    "epoch": active_session.current_epoch,
                    "progress": active_session.progress_percentage,
                    "last_updated": active_session.last_updated.isoformat()
                }
            },
            message="Training progress updated successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update training progress: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Failed to update training progress",
                details=str(e)
            )
        )


@training_router.get("/statistics")
def get_training_statistics(db: db_dependency):
    """
    Get comprehensive training statistics and analytics.
    """
    try:
        # Basic counts
        total_sessions = db.query(TrainingSession).count()
        completed_sessions = db.query(TrainingSession).filter(
            TrainingSession.completed_at.is_not(None)
        ).count()
        failed_sessions = db.query(TrainingSession).filter(
            TrainingSession.failed_at.is_not(None)
        ).count()
        active_sessions = db.query(TrainingSession).filter(
            TrainingSession.is_training == True
        ).count()
        
        # Recent activity (last 7 days)
        week_ago = datetime.utcnow() - timedelta(days=7)
        recent_sessions = db.query(TrainingSession).filter(
            TrainingSession.started_at >= week_ago
        ).count()
        
        # Most trained pieces
        all_sessions = db.query(TrainingSession).all()
        piece_counts = {}
        for session in all_sessions:
            if session.piece_labels:
                for piece in session.piece_labels:
                    piece_counts[piece] = piece_counts.get(piece, 0) + 1
        
        most_trained = sorted(piece_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return create_success_response(
            data={
                "overview": {
                    "total_sessions": total_sessions,
                    "completed_sessions": completed_sessions,
                    "failed_sessions": failed_sessions,
                    "active_sessions": active_sessions,
                    "success_rate": (completed_sessions / total_sessions * 100) if total_sessions > 0 else 0
                },
                "recent_activity": {
                    "sessions_last_7_days": recent_sessions
                },
                "most_trained_pieces": [
                    {"piece": piece, "training_count": count}
                    for piece, count in most_trained
                ],
                "system_info": {
                    "log_directory": log_dir,
                    "log_file_exists": os.path.exists(log_file),
                    "log_file_size_mb": round(os.path.getsize(log_file) / (1024 * 1024), 2) if os.path.exists(log_file) else 0
                }
            }
        )
        
    except Exception as e:
        logger.error(f"Failed to get training statistics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Failed to get training statistics",
                details=str(e)
            )
        )


@training_router.get("/resumable")
def get_resumable_sessions(db: db_dependency):
    """
    Get list of training sessions that can be resumed with enhanced filtering.
    """
    try:
        resumable_sessions = db.query(TrainingSession).filter(
            TrainingSession.is_training == False,
            TrainingSession.completed_at.is_(None),
            TrainingSession.failed_at.is_(None),
            TrainingSession.current_epoch > 1  # Only sessions that have made some progress
        ).order_by(TrainingSession.started_at.desc()).all()
        
        sessions_data = []
        for session in resumable_sessions:
            session_dict = session.to_dict()
            # Add resumability score based on progress
            progress_score = (session.current_epoch / session.epochs) * 100 if session.epochs else 0
            session_dict["resumability_score"] = round(progress_score, 1)
            session_dict["estimated_remaining_epochs"] = max(0, session.epochs - session.current_epoch)
            sessions_data.append(session_dict)
        
        return create_success_response(
            data={
                "resumable_sessions": sessions_data,
                "total_count": len(sessions_data),
                "recommendation": sessions_data[0] if sessions_data else None
            },
            message=f"Found {len(sessions_data)} resumable training sessions"
        )
        
    except Exception as e:
        logger.error(f"Failed to get resumable sessions: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Failed to get resumable sessions",
                details=str(e)
            )
        )


# ==================== BATCH OPERATIONS ====================

@training_router.post("/batch/train")
async def batch_train_multiple(
    piece_groups: List[List[str]],
    db: db_dependency,
    background_tasks: BackgroundTasks,
    sequential: bool = Query(False, description="Train groups sequentially instead of parallel")
):
    """
    Start multiple training sessions for different piece groups.
    """
    try:
        if not piece_groups:
            raise HTTPException(
                status_code=422,
                detail=create_error_response("piece_groups cannot be empty", status_code=422)
            )
        
        if len(piece_groups) > 5:  # Reasonable limit
            raise HTTPException(
                status_code=422,
                detail=create_error_response("Maximum 5 training groups allowed per batch", status_code=422)
            )
        
        # Check if any training is active
        active_session = get_active_training_session(db)
        if active_session:
            raise HTTPException(
                status_code=409,
                detail=create_error_response(
                    "Cannot start batch training while another session is active",
                    details={"active_session_id": active_session.id},
                    status_code=409
                )
            )
        
        created_sessions = []
        
        for i, piece_labels in enumerate(piece_groups):
            if not piece_labels:
                continue
                
            # Create training session for this group
            session = create_training_session(db, piece_labels)
            created_sessions.append({
                "session_id": session.id,
                "session_name": session.session_name,
                "piece_labels": piece_labels,
                "group_index": i
            })
        
        if not created_sessions:
            raise HTTPException(
                status_code=422,
                detail=create_error_response("No valid piece groups found", status_code=422)
            )
        
        # Start training sessions
        if sequential:
            # Sequential training - start first session only
            first_session = created_sessions[0]
            background_tasks.add_task(
                batch_sequential_training,
                [session["session_id"] for session in created_sessions]
            )
            message = f"Sequential batch training started with {len(created_sessions)} sessions"
        else:
            # Parallel training (not recommended for resource-intensive training)
            for session_info in created_sessions:
                background_tasks.add_task(
                    train_model_with_status_updates,
                    session_info["piece_labels"],
                    session_info["session_id"],
                    False
                )
            message = f"Parallel batch training started with {len(created_sessions)} sessions"
        
        return create_success_response(
            data={
                "created_sessions": created_sessions,
                "total_sessions": len(created_sessions),
                "mode": "sequential" if sequential else "parallel"
            },
            message=message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch training initiation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Batch training initiation failed",
                details=str(e)
            )
        )


async def batch_sequential_training(session_ids: List[int]):
    """
    Execute training sessions sequentially.
    """
    db = None
    try:
        db = create_new_session()
        
        for session_id in session_ids:
            session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
            if not session:
                logger.error(f"Session {session_id} not found, skipping")
                continue
            
            logger.info(f"Starting sequential training for session {session_id}")
            
            # Train this session
            await train_model_with_status_updates(
                session.piece_labels,
                session_id,
                False
            )
            
            # Wait a bit between sessions
            await asyncio.sleep(5)
        
        logger.info("Sequential batch training completed")
        
    except Exception as e:
        logger.error(f"Sequential batch training failed: {str(e)}", exc_info=True)
    finally:
        if db:
            safe_close(db)


# ==================== ADVANCED MONITORING ====================

@training_router.get("/metrics/{session_id}")
def get_session_metrics(session_id: int, db: db_dependency):
    """
    Get detailed metrics for a specific training session.
    """
    try:
        session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        
        if not session:
            raise HTTPException(
                status_code=404,
                detail=create_error_response(
                    f"Training session {session_id} not found",
                    status_code=404
                )
            )
        
        # Calculate derived metrics
        total_duration = None
        if session.started_at:
            end_time = session.completed_at or session.failed_at or datetime.utcnow()
            total_duration = (end_time - session.started_at).total_seconds()
        
        avg_epoch_time = None
        if total_duration and session.current_epoch > 1:
            avg_epoch_time = total_duration / (session.current_epoch - 1)
        
        estimated_completion = None
        if avg_epoch_time and session.epochs and session.current_epoch < session.epochs:
            remaining_epochs = session.epochs - session.current_epoch
            estimated_seconds = remaining_epochs * avg_epoch_time
            estimated_completion = datetime.utcnow() + timedelta(seconds=estimated_seconds)
        
        metrics_data = {
            "session_info": session.to_dict(),
            "performance_metrics": {
                "total_duration_seconds": total_duration,
                "average_epoch_time_seconds": avg_epoch_time,
                "estimated_completion": estimated_completion.isoformat() if estimated_completion else None,
                "epochs_per_hour": (3600 / avg_epoch_time) if avg_epoch_time else None
            },
            "training_progress": {
                "completion_percentage": (session.current_epoch / session.epochs * 100) if session.epochs else 0,
                "epochs_completed": session.current_epoch - 1,
                "epochs_remaining": max(0, session.epochs - session.current_epoch) if session.epochs else 0
            },
            "resource_usage": {
                "device_used": session.device_used,
                "total_images": session.total_images,
                "augmented_images": session.augmented_images,
                "validation_images": session.validation_images,
                "batch_size": session.batch_size,
                "image_size": session.image_size
            }
        }
        
        return create_success_response(data=metrics_data)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get session metrics: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Failed to get session metrics",
                details=str(e)
            )
        )


@training_router.post("/pause")
async def pause_training(db: db_dependency):
    """
    Pause the currently running training process.
    """
    try:
        active_session = get_active_training_session(db)
        if not active_session:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "No training is currently in progress",
                    status_code=400
                )
            )
        
        # Add pause logic here (this would depend on your training implementation)
        # For now, we'll just update the session status
        active_session.add_log("INFO", "Training paused by user request")
        active_session.is_training = False
        
        if not safe_commit(db):
            raise HTTPException(status_code=500, detail="Failed to pause training")
        
        return create_success_response(
            data={
                "session_id": active_session.id,
                "session_name": active_session.session_name,
                "paused_at_epoch": active_session.current_epoch,
                "progress_percentage": active_session.progress_percentage
            },
            message="Training paused successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to pause training: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Failed to pause training",
                details=str(e)
            )
        )


@training_router.post("/resume/{session_id}")
async def resume_specific_session(session_id: int, db: db_dependency, background_tasks: BackgroundTasks):
    """
    Resume a specific paused or incomplete training session.
    """
    try:
        # Check if any training is active
        active_session = get_active_training_session(db)
        if active_session:
            raise HTTPException(
                status_code=409,
                detail=create_error_response(
                    "Cannot resume session while another training is active",
                    details={"active_session_id": active_session.id},
                    status_code=409
                )
            )
        
        # Get the session to resume
        session = db.query(TrainingSession).filter(TrainingSession.id == session_id).first()
        if not session:
            raise HTTPException(
                status_code=404,
                detail=create_error_response(
                    f"Training session {session_id} not found",
                    status_code=404
                )
            )
        
        # Check if session can be resumed
        if session.is_training:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "Session is already running",
                    status_code=400
                )
            )
        
        if session.completed_at or session.failed_at:
            raise HTTPException(
                status_code=400,
                detail=create_error_response(
                    "Cannot resume completed or failed session",
                    status_code=400
                )
            )
        
        # Resume the session
        session.add_log("INFO", f"Resuming training from epoch {session.current_epoch}")
        
        # Start training as background task
        background_tasks.add_task(
            train_model_with_status_updates,
            session.piece_labels,
            session_id,
            True  # is_resume = True
        )
        
        return create_success_response(
            data={
                "session_id": session.id,
                "session_name": session.session_name,
                "resume_from_epoch": session.current_epoch,
                "piece_labels": session.piece_labels,
                "total_epochs": session.epochs
            },
            message="Training session resumed successfully"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to resume training session {session_id}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Failed to resume training session",
                details=str(e)
            )
        )


# ==================== CLEANUP OPERATIONS ====================

@training_router.delete("/cleanup")
async def cleanup_old_sessions(
    db: db_dependency,
    older_than_days: int = Query(30, ge=1, description="Delete sessions older than specified days"),
    keep_completed: bool = Query(True, description="Keep completed sessions"),
    dry_run: bool = Query(True, description="Show what would be deleted without actually deleting")
):
    """
    Clean up old training sessions with safety options.
    """
    try:
        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
        
        query = db.query(TrainingSession).filter(
            TrainingSession.started_at < cutoff_date,
            TrainingSession.is_training == False  # Never delete active sessions
        )
        
        if keep_completed:
            query = query.filter(TrainingSession.completed_at.is_(None))
        
        sessions_to_delete = query.all()
        
        if dry_run:
            return create_success_response(
                data={
                    "sessions_to_delete": [
                        {
                            "id": s.id,
                            "name": s.session_name,
                            "started_at": s.started_at.isoformat(),
                            "status": "completed" if s.completed_at else "failed" if s.failed_at else "incomplete"
                        }
                        for s in sessions_to_delete
                    ],
                    "total_count": len(sessions_to_delete),
                    "dry_run": True
                },
                message=f"Would delete {len(sessions_to_delete)} sessions (dry run)"
            )
        
        # Actually delete sessions
        deleted_count = 0
        deleted_sessions = []
        
        for session in sessions_to_delete:
            try:
                deleted_sessions.append({
                    "id": session.id,
                    "name": session.session_name,
                    "started_at": session.started_at.isoformat()
                })
                db.delete(session)
                deleted_count += 1
            except Exception as e:
                logger.error(f"Failed to delete session {session.id}: {str(e)}")
        
        if not safe_commit(db):
            raise HTTPException(status_code=500, detail="Failed to commit cleanup changes")
        
        logger.info(f"Cleaned up {deleted_count} old training sessions")
        
        return create_success_response(
            data={
                "deleted_sessions": deleted_sessions,
                "deleted_count": deleted_count
            },
            message=f"Successfully deleted {deleted_count} old training sessions"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Cleanup operation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Cleanup operation failed",
                details=str(e)
            )
        )


# ==================== EXPORT OPERATIONS ====================

@training_router.get("/export/report")
async def export_training_report(
    db: db_dependency,
    format_type: str = Query("json", regex="^(json|csv)$", description="Export format"),
    include_logs: bool = Query(False, description="Include training logs in export"),
    session_ids: Optional[List[int]] = Query(None, description="Specific session IDs to export")
):
    """
    Export training data in various formats.
    """
    try:
        query = db.query(TrainingSession)
        
        if session_ids:
            query = query.filter(TrainingSession.id.in_(session_ids))
        
        sessions = query.order_by(TrainingSession.started_at.desc()).all()
        
        if not sessions:
            return create_success_response(
                data={"sessions": [], "total_count": 0},
                message="No sessions found for export"
            )
        
        export_data = []
        for session in sessions:
            session_data = session.to_dict()
            if not include_logs:
                session_data.pop("training_logs", None)
                session_data.pop("logs", None)
            export_data.append(session_data)
        
        if format_type == "json":
            return create_success_response(
                data={
                    "sessions": export_data,
                    "total_count": len(export_data),
                    "exported_at": datetime.utcnow().isoformat(),
                    "format": "json"
                }
            )
        
        elif format_type == "csv":
            # For CSV, we'll flatten the data structure
            import csv
            import io
            
            output = io.StringIO()
            if export_data:
                # Get all possible field names
                all_fields = set()
                for session in export_data:
                    all_fields.update(session.keys())
                
                writer = csv.DictWriter(output, fieldnames=sorted(all_fields))
                writer.writeheader()
                
                for session in export_data:
                    # Flatten complex fields
                    row = {}
                    for key, value in session.items():
                        if isinstance(value, (dict, list)):
                            row[key] = str(value)
                        else:
                            row[key] = value
                    writer.writerow(row)
            
            return JSONResponse(
                content={
                    "status": ResponseStatus.SUCCESS,
                    "data": {
                        "csv_content": output.getvalue(),
                        "total_count": len(export_data),
                        "exported_at": datetime.utcnow().isoformat(),
                        "format": "csv"
                    },
                    "message": f"Exported {len(export_data)} sessions in CSV format"
                }
            )
        
    except Exception as e:
        logger.error(f"Export operation failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=create_error_response(
                "Export operation failed",
                details=str(e)
            )
        )


