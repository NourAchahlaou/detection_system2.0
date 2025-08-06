# basic_detection_router.py - Database-integrated detection router with synchronous DB operations

import logging
import time
from typing import List, Optional, Dict, Any, Annotated
from dataclasses import asdict
from fastapi import APIRouter, HTTPException, Request, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from detection.app.service.alternative.basic_detection_service import (
    DetectionRequest, 
    DetectionResponse,
    LotCreationRequest,
    LotResponse,
    basic_detection_processor
)
from detection.app.db.session import get_session

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database dependency
db_dependency = Annotated[Session, Depends(get_session)]

# Pydantic models for request validation
class CreateLotRequest(BaseModel):
    lot_name: str = Field(..., min_length=1, max_length=100, description="Name of the detection lot")
    expected_piece_id: int = Field(..., gt=0, description="ID of the expected piece")
    expected_piece_number: int = Field(..., gt=0, description="Expected piece number")

class DetectionWithLotRequest(BaseModel):
    target_label: str = Field(..., min_length=1, description="Target label to detect")
    lot_id: Optional[int] = Field(None, gt=0, description="Optional lot ID for tracking")
    expected_piece_id: Optional[int] = Field(None, gt=0, description="Expected piece ID")
    expected_piece_number: Optional[int] = Field(None, gt=0, description="Expected piece number")
    quality: int = Field(85, ge=50, le=100, description="JPEG quality for response")

class UpdateLotStatusRequest(BaseModel):
    is_target_match: bool = Field(..., description="Whether the lot matches the target requirements")

# Create the router
basic_detection_router = APIRouter(
    prefix="/basic",
    tags=["Basic Detection with Database"],
    responses={404: {"description": "Not found"}},
)

@basic_detection_router.post("/lots", response_model=Dict[str, Any])
async def create_detection_lot(lot_request: CreateLotRequest, db: db_dependency):
    """
    Create a new detection lot for tracking detection sessions
    
    A lot represents a batch or group of items that need to be detected and validated.
    Each lot tracks multiple detection sessions until the target requirements are met.
    """
    try:
        logger.info(f"📦 Creating new detection lot: '{lot_request.lot_name}' expecting piece {lot_request.expected_piece_number}")
        
        # Create lot creation request
        creation_request = LotCreationRequest(
            lot_name=lot_request.lot_name,
            expected_piece_id=lot_request.expected_piece_id,
            expected_piece_number=lot_request.expected_piece_number
        )
        
        # Create the lot
        lot_response = basic_detection_processor.create_detection_lot(creation_request, db)
        
        return JSONResponse(
            status_code=201,
            content={
                "success": True,
                "data": asdict(lot_response),
                "message": f"Detection lot '{lot_request.lot_name}' created successfully",
                "lot_id": lot_response.lot_id
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error creating detection lot: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create detection lot: {str(e)}")

@basic_detection_router.get("/all_lots", response_model=List[LotResponse])
async def get_all_detection_lots(db: db_dependency):
    """
    Get all detection lots with their current status and statistics
    
    This endpoint returns a list of all detection lots, including their IDs, names,
    expected pieces, and whether they have been completed or not.
    """
    try:
        lots = basic_detection_processor.get_all_detection_lots(db)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": [asdict(lot) for lot in lots],
                "message": "All detection lots retrieved successfully"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting all detection lots: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get detection lots: {str(e)}")

@basic_detection_router.get("/lots/{lot_id}", response_model=Dict[str, Any])
async def get_detection_lot(lot_id: int, db: db_dependency):
    """
    Get detection lot information including statistics
    """
    try:
        if lot_id <= 0:
            raise HTTPException(status_code=400, detail="Invalid lot_id")
        
        lot_info = basic_detection_processor.get_detection_lot(lot_id, db)
        
        if not lot_info:
            raise HTTPException(status_code=404, detail=f"Detection lot {lot_id} not found")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": asdict(lot_info),
                "message": f"Detection lot {lot_id} retrieved successfully"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting detection lot {lot_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get detection lot: {str(e)}")

@basic_detection_router.put("/lots/{lot_id}/status", response_model=Dict[str, Any])
async def update_lot_target_match_status(lot_id: int, status_request: UpdateLotStatusRequest, db: db_dependency):
    """
    Update lot target match status
    
    Use this endpoint to mark a lot as matching target requirements (is_target_match=True)
    or to indicate it needs correction (is_target_match=False).
    
    When marked as True, the lot will be considered completed.
    """
    try:
        if lot_id <= 0:
            raise HTTPException(status_code=400, detail="Invalid lot_id")
        
        # Check if lot exists
        lot_info = basic_detection_processor.get_detection_lot(lot_id, db)
        if not lot_info:
            raise HTTPException(status_code=404, detail=f"Detection lot {lot_id} not found")
        
        # Update lot status
        success = basic_detection_processor.update_lot_target_match(
            lot_id, 
            status_request.is_target_match,
            db
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update lot status")
        
        # Get updated lot info
        updated_lot_info = basic_detection_processor.get_detection_lot(lot_id, db)
        
        status_message = "completed" if status_request.is_target_match else "marked for correction"
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": asdict(updated_lot_info) if updated_lot_info else None,
                "message": f"Detection lot {lot_id} {status_message} successfully",
                "is_target_match": status_request.is_target_match
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating lot status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update lot status: {str(e)}")

@basic_detection_router.post("/detect/{camera_id}")
async def detect_with_database_tracking(
    camera_id: int,
    request: DetectionWithLotRequest,
    db: db_dependency
):
    """
    Perform detection with database tracking
    
    This endpoint:
    1. Freezes the video stream
    2. Captures and analyzes the current frame
    3. Creates a detection session record in the database
    4. Updates lot status if target is matched
    5. Returns detection results with database references
    
    If lot_id is provided, the detection will be tracked as part of that lot.
    If lot_id is not provided, detection will run without database tracking.
    """
    try:
        if camera_id < 0:
            raise HTTPException(status_code=400, detail="Invalid camera_id")
        
        logger.info(f"🎯 Enhanced detection request for camera {camera_id}, target: '{request.target_label}'")
        
        # Create detection request
        detection_request = DetectionRequest(
            camera_id=camera_id,
            target_label=request.target_label,
            lot_id=request.lot_id,
            expected_piece_id=request.expected_piece_id,
            expected_piece_number=request.expected_piece_number,
            timestamp=time.time(),
            quality=request.quality
        )
        
        # Perform detection with lot tracking
        response = await basic_detection_processor.detect_with_lot_tracking(detection_request, db)
        
        # Convert dataclass to dict for JSON response
        response_dict = asdict(response)
        
        # Prepare response message
        message_parts = ["Detection completed successfully"]
        if response.lot_id:
            message_parts.append(f"tracked in lot {response.lot_id}")
        if response.session_id:
            message_parts.append(f"session {response.session_id} created")
        if response.detected_target:
            message_parts.append("🎯 TARGET DETECTED!")
        
        message_parts.append("Stream is frozen - use /unfreeze to resume live feed")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": response_dict,
                "message": " - ".join(message_parts),
                "stream_frozen": response.stream_frozen,
                "target_detected": response.detected_target,
                "lot_id": response.lot_id,
                "session_id": response.session_id
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing enhanced detection request: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@basic_detection_router.get("/lots/{lot_id}/sessions")
async def get_lot_detection_sessions(lot_id: int, db: db_dependency):
    """
    Get all detection sessions for a specific lot
    """
    try:
        if lot_id <= 0:
            raise HTTPException(status_code=400, detail="Invalid lot_id")
        
        # Check if lot exists
        lot_info = basic_detection_processor.get_detection_lot(lot_id, db)
        if not lot_info:
            raise HTTPException(status_code=404, detail=f"Detection lot {lot_id} not found")
        
        # Get sessions
        sessions = basic_detection_processor.get_lot_sessions(lot_id, db)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "lot_id": lot_id,
                "lot_name": lot_info.lot_name,
                "total_sessions": len(sessions),
                "successful_detections": sum(1 for s in sessions if s.get('is_target_match', False)),
                "sessions": sessions,
                "lot_status": {
                    "is_target_match": lot_info.is_target_match,
                    "completed_at": lot_info.completed_at if lot_info.completed_at else None  # FIXED HERE
                }
            }
        )

        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting lot sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get lot sessions: {str(e)}")

@basic_detection_router.post("/detect/{camera_id}/with-lot-creation")
async def detect_with_automatic_lot_creation(
    camera_id: int,
    request: Request,
    db: db_dependency
):
    """
    Create a lot and perform detection in one operation
    
    Body should contain:
    {
        "lot_name": "Batch_001",
        "expected_piece_id": 123,
        "expected_piece_number": 456,
        "target_label": "person",
        "quality": 85
    }
    """
    try:
        if camera_id < 0:
            raise HTTPException(status_code=400, detail="Invalid camera_id")
        
        body = await request.json()
        
        # Validate required fields
        required_fields = ['lot_name', 'expected_piece_id', 'expected_piece_number', 'target_label']
        for field in required_fields:
            if field not in body:
                raise HTTPException(status_code=400, detail=f"Missing required field: {field}")
        
        logger.info(f"🚀 Creating lot and detecting for camera {camera_id}")
        
        # Create lot first
        lot_creation = LotCreationRequest(
            lot_name=body['lot_name'],
            expected_piece_id=body['expected_piece_id'],
            expected_piece_number=body['expected_piece_number']
        )
        
        lot_response = basic_detection_processor.create_detection_lot(lot_creation, db)
        
        # Perform detection
        detection_request = DetectionRequest(
            camera_id=camera_id,
            target_label=body['target_label'],
            lot_id=lot_response.lot_id,
            expected_piece_id=body['expected_piece_id'],
            expected_piece_number=body['expected_piece_number'],
            timestamp=time.time(),
            quality=body.get('quality', 85)
        )
        
        detection_response = await basic_detection_processor.detect_with_lot_tracking(detection_request, db)
        
        return JSONResponse(
            status_code=201,
            content={
                "success": True,
                "lot_created": asdict(lot_response),
                "detection_result": asdict(detection_response),
                "message": f"Lot '{body['lot_name']}' created and detection completed",
                "target_detected": detection_response.detected_target
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in lot creation and detection: {e}")
        raise HTTPException(status_code=500, detail=f"Operation failed: {str(e)}")

@basic_detection_router.post("/stream/{camera_id}/unfreeze")
async def unfreeze_stream_after_detection(camera_id: int):
    """
    Unfreeze the video stream to resume live feed after detection
    """
    try:
        if camera_id < 0:
            raise HTTPException(status_code=400, detail="Invalid camera_id")
        
        logger.info(f"🔥 Unfreezing stream for camera {camera_id}")
        
        success = await basic_detection_processor.unfreeze_stream(camera_id)
        
        if success:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "camera_id": camera_id,
                    "message": f"Stream unfrozen for camera {camera_id}. Live feed resumed.",
                    "timestamp": time.time()
                }
            )
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"No active stream found for camera {camera_id} or failed to unfreeze"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error unfreezing stream: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unfreeze stream: {str(e)}")

@basic_detection_router.post("/detect/{camera_id}/with-auto-correction")
async def detect_with_automatic_lot_correction(
    camera_id: int,
    request: Request,
    db: db_dependency
):
    """
    Perform detection with automatic lot correction workflow
    
    This endpoint:
    1. Performs detection
    2. If target is detected and matches expected piece, marks lot as complete
    3. If target is not detected or doesn't match, keeps lot open for correction
    4. Automatically manages lot status based on detection results
    
    Body should contain:
    {
        "lot_id": 123,
        "target_label": "person",
        "quality": 85,
        "auto_complete_on_match": true
    }
    """
    try:
        if camera_id < 0:
            raise HTTPException(status_code=400, detail="Invalid camera_id")
        
        body = await request.json()
        lot_id = body.get('lot_id')
        target_label = body.get('target_label')
        
        if not lot_id or not target_label:
            raise HTTPException(status_code=400, detail="lot_id and target_label are required")
        
        auto_complete = body.get('auto_complete_on_match', True)
        
        # Check if lot exists and get current status
        lot_info = basic_detection_processor.get_detection_lot(lot_id, db)
        if not lot_info:
            raise HTTPException(status_code=404, detail=f"Detection lot {lot_id} not found")
        
        if lot_info.is_target_match:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "message": f"Lot {lot_id} is already completed and matches target requirements",
                    "lot_status": "completed",
                    "detection_skipped": True
                }
            )
        
        logger.info(f"🔄 Auto-correction detection for lot {lot_id}")
        
        # Perform detection
        detection_request = DetectionRequest(
            camera_id=camera_id,
            target_label=target_label,
            lot_id=lot_id,
            expected_piece_id=lot_info.expected_piece_id,
            expected_piece_number=lot_info.expected_piece_number,
            timestamp=time.time(),
            quality=body.get('quality', 85)
        )
        
        detection_response = await basic_detection_processor.detect_with_lot_tracking(detection_request, db)
        
        # Auto-correction logic
        correction_action = "none"
        if detection_response.detected_target and auto_complete:
            # Mark lot as completed since target was detected
            basic_detection_processor.update_lot_target_match(lot_id, True, db)
            correction_action = "lot_completed"
            logger.info(f"✅ Lot {lot_id} automatically marked as complete - target detected!")
        elif not detection_response.detected_target:
            # Keep lot open for further correction attempts
            correction_action = "needs_correction"
            logger.info(f"🔄 Lot {lot_id} still needs correction - target not detected")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "detection_result": asdict(detection_response),
                "correction_action": correction_action,
                "lot_completed": correction_action == "lot_completed",
                "message": f"Detection completed with auto-correction: {correction_action}",
                "target_detected": detection_response.detected_target
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in auto-correction detection: {e}")
        raise HTTPException(status_code=500, detail=f"Auto-correction detection failed: {str(e)}")

@basic_detection_router.get("/lots", response_model=Dict[str, Any])
async def list_detection_lots(
    limit: int = Query(50, ge=1, le=100, description="Maximum number of lots to return"),
    offset: int = Query(0, ge=0, description="Number of lots to skip"),
    completed_only: bool = Query(False, description="Return only completed lots"),
    pending_only: bool = Query(False, description="Return only pending lots"),
    db: db_dependency = None
):
    """
    List detection lots with optional filtering
    """
    try:
        # This would require implementing a list method in the processor
        # For now, return a placeholder response
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "List lots endpoint - implementation needed in processor",
                "filters": {
                    "limit": limit,
                    "offset": offset,
                    "completed_only": completed_only,
                    "pending_only": pending_only
                }
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error listing lots: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list lots: {str(e)}")

@basic_detection_router.get("/health")
async def health_check():
    """
    Check if the enhanced detection service is healthy and ready
    """
    try:
        # Check if processor is initialized
        stats = basic_detection_processor.get_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy" if stats['is_initialized'] else "initializing",
                "is_initialized": stats['is_initialized'],
                "device": stats.get('device', 'unknown'),
                "statistics": {
                    "detections_performed": stats.get('detections_performed', 0),
                    "targets_detected": stats.get('targets_detected', 0),
                    "lots_created": stats.get('lots_created', 0),
                    "lots_completed": stats.get('lots_completed', 0),
                    "avg_processing_time": stats.get('avg_processing_time', 0)
                },
                "message": "Enhanced detection service with database integration is operational"
            }
        )
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "message": "Enhanced detection service is not operational"
            }
        )

@basic_detection_router.post("/initialize")
async def initialize_processor():
    """
    Initialize the enhanced detection processor
    """
    try:
        logger.info("🚀 Initializing enhanced detection processor...")
        
        await basic_detection_processor.initialize()
        
        stats = basic_detection_processor.get_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Enhanced detection processor initialized successfully",
                "device": stats.get('device', 'unknown'),
                "is_initialized": stats['is_initialized']
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error initializing processor: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to initialize processor: {str(e)}"
        )

@basic_detection_router.get("/stats")
async def get_detection_stats():
    """
    Get enhanced detection service statistics including database metrics
    """
    try:
        stats = basic_detection_processor.get_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "stats": stats,
                "timestamp": time.time(),
                "service_type": "enhanced_detection_with_database"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@basic_detection_router.delete("/lots/{lot_id}")
async def delete_detection_lot(lot_id: int, db: db_dependency):
    """
    Delete a detection lot and all its associated sessions
    
    Warning: This action cannot be undone!
    """
    try:
        if lot_id <= 0:
            raise HTTPException(status_code=400, detail="Invalid lot_id")
        
        # Check if lot exists
        lot_info = basic_detection_processor.get_detection_lot(lot_id, db)
        if not lot_info:
            raise HTTPException(status_code=404, detail=f"Detection lot {lot_id} not found")
        
        # TODO: Implement delete method in processor
        # For now, return a placeholder response
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Delete lot {lot_id} - implementation needed in processor",
                "lot_name": lot_info.lot_name,
                "warning": "This action would permanently delete the lot and all associated sessions"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error deleting lot: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete lot: {str(e)}")