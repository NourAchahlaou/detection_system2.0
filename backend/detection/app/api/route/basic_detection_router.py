# enhanced_basic_detection_router.py - Enhanced with detected pieces support

import logging
import time
from typing import List, Optional, Dict, Any, Annotated
from dataclasses import asdict
from fastapi import APIRouter, HTTPException, Request, Depends, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from detection.app.service.detection.alternative.basic_detection_service import basic_detection_processor

from detection.app.schema.lotRequest import DetectionRequest,LotCreationRequest
from detection.app.schema.lotResponse import LotResponse
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

class InitializeWithLotRequest(BaseModel):
    lot_id: Optional[int] = Field(None, gt=0, description="Lot ID for context")
    piece_label: Optional[str] = Field(None, min_length=1, description="Piece label for model initialization")
    target_label: Optional[str] = Field(None, min_length=1, description="Target label (same as piece_label)")
    initialize_model_for_piece: bool = Field(True, description="Whether to initialize model for specific piece")

# Create the router
basic_detection_router = APIRouter(
    prefix="/basic",
    tags=["Basic Detection with Database"],
    responses={404: {"description": "Not found"}},
)

@basic_detection_router.post("/initialize")
async def initialize_processor_with_context(
    request: Optional[InitializeWithLotRequest] = None,
    db: Session = Depends(get_session)
):
    """Initialize the detection processor with optional lot context"""
    try:
        logger.info("🚀 Starting basic detection processor initialization...")
        
        lot_id = None
        piece_label = None
        
        if request:
            lot_id = request.lot_id
            piece_label = request.piece_label or request.target_label
            if lot_id and piece_label:
                logger.info(f"📋 Initializing with lot context - Lot ID: {lot_id}, Piece: {piece_label}")
        
        initialization_message = "Basic detection processor initialized successfully"
        context_info = {}
        
        lot_info = None
        if lot_id and piece_label:
            try:
                logger.info(f"🔧 Step 1: Validating lot context for lot {lot_id} with piece {piece_label}")
                
                lot_info = basic_detection_processor.get_detection_lot(lot_id, db)
                if not lot_info:
                    raise HTTPException(status_code=404, detail=f"Detection lot {lot_id} not found")
                
                logger.info(f"✅ Found lot {lot_id}: {lot_info.lot_name}")
                
                logger.info(f"🔧 Step 2: Initializing detection system for lot {lot_id} with piece {piece_label}")
                lot_init_result = await basic_detection_processor.initialize_with_lot_context(
                    lot_id, piece_label, db
                )
                
                if lot_init_result and lot_init_result.get('success', False):
                    initialization_message = f"Detection processor initialized for lot {lot_id} with piece: {piece_label}"
                    context_info = {
                        "lot_id": lot_id,
                        "piece_label": piece_label,
                        "lot_name": lot_info.lot_name,
                        "expected_pieces": lot_info.expected_piece_number,
                        "lot_context": lot_init_result.get('lot_context', {})
                    }
                    logger.info(f"✅ Lot context initialization successful for {piece_label}")
                else:
                    logger.error(f"❌ Lot context initialization failed for {piece_label}")
                    raise HTTPException(status_code=500, detail="Failed to initialize with lot context")
                    
            except HTTPException:
                raise
            except Exception as context_error:
                logger.error(f"❌ Error setting lot context: {context_error}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Failed to set lot context: {str(context_error)}")
        else:
            logger.info("🔧 Step 2: Initializing basic detection processor without lot context...")
            await basic_detection_processor.initialize()
        
        stats = basic_detection_processor.get_stats()
        
        response_data = {
            "status": "initialized",
            "success": True,
            "message": initialization_message,
            "device": stats.get('device', 'unknown'),
            "is_initialized": stats['is_initialized'],
            "context": context_info,
            "timestamp": time.time()
        }
        
        logger.info(f"✅ Basic detection processor initialization completed: {initialization_message}")
        
        return JSONResponse(
            status_code=200,
            content=response_data
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error initializing basic detection processor: {e}", exc_info=True)
        
        error_details = {
            "error_type": type(e).__name__,
            "error_message": str(e),
            "lot_id": lot_id if 'lot_id' in locals() else None,
            "piece_label": piece_label if 'piece_label' in locals() else None
        }
        
        return JSONResponse(
            status_code=500,
            content={
                "status": "failed",
                "success": False,
                "error": str(e),
                "error_details": error_details,
                "message": f"Failed to initialize basic detection processor: {str(e)}",
                "timestamp": time.time()
            }
        )

@basic_detection_router.get("/health")
async def health_check_with_context():
    """Enhanced health check that includes lot context information"""
    try:
        stats = basic_detection_processor.get_stats()
        
        current_context = {}
        if hasattr(basic_detection_processor, 'current_lot_context'):
            current_context = basic_detection_processor.current_lot_context
        
        is_healthy = stats['is_initialized']
        status = "healthy" if is_healthy else "initializing"
        
        response_data = {
            "status": status,
            "is_initialized": stats['is_initialized'],
            "device": stats.get('device', 'unknown'),
            "current_context": current_context,
            "statistics": {
                "detections_performed": stats.get('detections_performed', 0),
                "targets_detected": stats.get('targets_detected', 0),
                "lots_created": stats.get('lots_created', 0),
                "lots_completed": stats.get('lots_completed', 0),
                "detected_pieces_stored": stats.get('detected_pieces_stored', 0),
                "avg_processing_time": stats.get('avg_processing_time', 0)
            },
            "message": "Basic detection service with database integration is operational",
            "timestamp": time.time()
        }
        
        return JSONResponse(
            status_code=200 if is_healthy else 503,
            content=response_data
        )
        
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "message": "Basic detection service is not operational",
                "timestamp": time.time()
            }
        )

@basic_detection_router.post("/lots", response_model=Dict[str, Any])
async def create_detection_lot(lot_request: CreateLotRequest, db: Session = Depends(get_session)):
    """Create a new detection lot for tracking detection sessions"""
    try:
        logger.info(f"📦 Creating new detection lot: '{lot_request.lot_name}' expecting piece {lot_request.expected_piece_number}")
        
        creation_request = LotCreationRequest(
            lot_name=lot_request.lot_name,
            expected_piece_id=lot_request.expected_piece_id,
            expected_piece_number=lot_request.expected_piece_number
        )
        
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
async def get_all_detection_lots(db: Session = Depends(get_session)):
    """Get all detection lots with their current status and statistics"""
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
async def get_detection_lot(lot_id: int, db: Session = Depends(get_session)):
    """Get detection lot information including statistics"""
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
async def update_lot_target_match_status(lot_id: int, status_request: UpdateLotStatusRequest, db: Session = Depends(get_session)):
    """Update lot target match status"""
    try:
        if lot_id <= 0:
            raise HTTPException(status_code=400, detail="Invalid lot_id")
        
        lot_info = basic_detection_processor.get_detection_lot(lot_id, db)
        if not lot_info:
            raise HTTPException(status_code=404, detail=f"Detection lot {lot_id} not found")
        
        success = basic_detection_processor.update_lot_target_match(
            lot_id, 
            status_request.is_target_match,
            db
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update lot status")
        
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
    db: Session = Depends(get_session)
):
    """Perform detection with database tracking and detected pieces storage"""
    try:
        if camera_id < 0:
            raise HTTPException(status_code=400, detail="Invalid camera_id")
        
        logger.info(f"🎯 Enhanced detection request for camera {camera_id}, target: '{request.target_label}'")
        
        detection_request = DetectionRequest(
            camera_id=camera_id,
            target_label=request.target_label,
            lot_id=request.lot_id,
            expected_piece_id=request.expected_piece_id,
            expected_piece_number=request.expected_piece_number,
            timestamp=time.time(),
            quality=request.quality
        )
        
        response = await basic_detection_processor.detect_with_lot_tracking(detection_request, db)
        
        response_dict = asdict(response)
        
        message_parts = ["Detection completed successfully"]
        if response.lot_id:
            message_parts.append(f"tracked in lot {response.lot_id}")
        if response.session_id:
            message_parts.append(f"session {response.session_id} created with detected pieces data")
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
async def get_lot_detection_sessions_with_pieces(lot_id: int, db: Session = Depends(get_session)):
    """Get all detection sessions for a specific lot with detected pieces data"""
    try:
        if lot_id <= 0:
            raise HTTPException(status_code=400, detail="Invalid lot_id")
        
        lot_info = basic_detection_processor.get_detection_lot(lot_id, db)
        if not lot_info:
            raise HTTPException(status_code=404, detail=f"Detection lot {lot_id} not found")
        
        # Get sessions with detected pieces data
        sessions = basic_detection_processor.get_lot_sessions(lot_id, db)
        
        # Calculate totals for summary
        total_sessions = len(sessions)
        successful_detections = sum(1 for s in sessions if s.get('is_target_match', False))
        total_pieces_detected = sum(s.get('detected_pieces_count', 0) for s in sessions)
        total_correct_pieces = sum(
            len([p for p in s.get('detected_pieces', []) if p.get('is_correct_piece', False)]) 
            for s in sessions
        )
        total_incorrect_pieces = sum(
            len([p for p in s.get('detected_pieces', []) if not p.get('is_correct_piece', False)]) 
            for s in sessions
        )
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "lot_id": lot_id,
                "lot_name": lot_info.lot_name,
                "lot_status": {
                    "is_target_match": lot_info.is_target_match,
                    "completed_at": lot_info.completed_at if lot_info.completed_at else None,
                    "expected_piece_number": lot_info.expected_piece_number
                },
                "summary": {
                    "total_sessions": total_sessions,
                    "successful_detections": successful_detections,
                    "total_pieces_detected": total_pieces_detected,
                    "total_correct_pieces": total_correct_pieces,
                    "total_incorrect_pieces": total_incorrect_pieces
                },
                "sessions": sessions,
                "message": f"Retrieved {total_sessions} sessions with {total_pieces_detected} total detected pieces"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting lot sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get lot sessions: {str(e)}")

@basic_detection_router.post("/stream/{camera_id}/unfreeze")
async def unfreeze_stream_after_detection(camera_id: int):
    """Unfreeze the video stream to resume live feed after detection"""
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

@basic_detection_router.get("/stats")
async def get_detection_stats():
    """Get enhanced detection service statistics including database metrics"""
    try:
        stats = basic_detection_processor.get_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "stats": stats,
                "timestamp": time.time(),
                "service_type": "enhanced_detection_with_database_and_pieces"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
    
# Add this to your lot session API router (wherever you have the /lotSession endpoints)

# Fixed version of the get_lot_session_dashboard_data function
# Replace the existing function in your basic_detection_router.py

@basic_detection_router.get("/lotSession/data")
async def get_lot_session_dashboard_data(
    group_filter: Optional[str] = None,
    search: Optional[str] = None,
    status_filter: Optional[str] = None,
    db: Session = Depends(get_session)
):
    """Get dashboard data for lot sessions with detected pieces"""
    try:
        from detection.app.db.models.detectionLot import DetectionLot
        from detection.app.db.models.detectionSession import DetectionSession
        from detection.app.db.models.detectedPiece import DetectedPiece
        from detection.app.db.models.piece import Piece  # Import the Piece model
        from sqlalchemy.orm import selectinload
        from sqlalchemy import func
        import json
        
        # Base query for lots with sessions and detected pieces
        query = db.query(DetectionLot).options(
            selectinload(DetectionLot.detection_sessions).selectinload(DetectionSession.detected_pieces)
        )
        
        # Apply filters if provided
        if group_filter:
            # Assuming you have a way to determine group from lot name or have a group field
            query = query.filter(DetectionLot.lot_name.like(f"%{group_filter}%"))
        
        if search:
            query = query.filter(
                DetectionLot.lot_name.ilike(f"%{search}%")
            )
        
        lots = query.order_by(DetectionLot.created_at.desc()).all()
        
        # Group lots by production group (extract from lot name or use a group field)
        grouped_lots = {}
        statistics = {
            'totalGroups': 0,
            'totalLots': len(lots),
            'totalSessions': 0,
            'sessionSuccessRate': 0,
            'avgLotConfidence': 0,
            'activeGroups': 0
        }
        
        for lot in lots:
            # Extract group name from lot name (adapt this logic to your naming convention)
            group_name = lot.lot_name.split('_')[0] if '_' in lot.lot_name else 'DEFAULT'
            
            if group_name not in grouped_lots:
                grouped_lots[group_name] = []
            
            # Get the expected piece label from the database using the lot's expected_piece_id
            expected_piece_label = "target"  # Default fallback
            if lot.expected_piece_id:
                expected_piece = db.query(Piece).filter(Piece.id == lot.expected_piece_id).first()
                if expected_piece and expected_piece.piece_label:
                    expected_piece_label = expected_piece.piece_label
            
            # Process sessions for this lot
            processed_sessions = []
            total_confidence = 0
            successful_sessions = 0
            
            for session_num, session in enumerate(lot.detection_sessions, 1):
                # Process detected pieces
                detected_pieces_data = []
                for piece in session.detected_pieces:
                    piece_data = {
                        'id': piece.id,
                        'piece_id': piece.piece_id,
                        'detected_label': piece.detected_label,
                        'confidence_score': piece.confidence_score,
                        'bounding_box_x1': piece.bounding_box_x1,
                        'bounding_box_y1': piece.bounding_box_y1,
                        'bounding_box_x2': piece.bounding_box_x2,
                        'bounding_box_y2': piece.bounding_box_y2,
                        'is_correct_piece': piece.is_correct_piece,
                        'created_at': piece.created_at.isoformat() if piece.created_at else None
                    }
                    detected_pieces_data.append(piece_data)
                
                # Use the expected piece label from the lot instead of "target"
                target_piece = expected_piece_label
                detected_piece = "unknown"
                
                if detected_pieces_data:
                    # Use the most confident correct piece, or most confident overall
                    correct_pieces = [p for p in detected_pieces_data if p['is_correct_piece']]
                    if correct_pieces:
                        best_piece = max(correct_pieces, key=lambda p: p['confidence_score'])
                        detected_piece = best_piece['detected_label']
                    else:
                        best_piece = max(detected_pieces_data, key=lambda p: p['confidence_score'])
                        detected_piece = best_piece['detected_label']
                
                session_data = {
                    'id': session.id,
                    'sessionNumber': session_num,
                    'targetPiece': target_piece,  # Now uses the actual piece label
                    'detectedPiece': detected_piece,
                    'confidence': session.confidence_score or 0,
                    'isTargetMatch': session.is_target_match,
                    'status': 'completed' if session.is_target_match else 'failed',
                    'timestamp': session.created_at.isoformat() if session.created_at else None,
                    'processingTime': 1000,  # Default processing time
                    'detectionRate': session.detection_rate or 0,
                    'detectedPieces': detected_pieces_data,
                    'detected_pieces': detected_pieces_data  # Alias
                }
                
                processed_sessions.append(session_data)
                total_confidence += session.confidence_score or 0
                if session.is_target_match:
                    successful_sessions += 1
                statistics['totalSessions'] += 1
            
            # Calculate lot statistics
            avg_confidence = (total_confidence / len(processed_sessions)*100) if processed_sessions else 0
            success_rate = (successful_sessions / len(processed_sessions) * 100) if processed_sessions else 0
            
            lot_data = {
                'id': lot.id,
                'group': group_name,
                'lotName': lot.lot_name.replace(f"{group_name}_", "") if group_name in lot.lot_name else lot.lot_name,
                'expectedPiece': expected_piece_label,  # Now uses the actual piece label
                'expectedPieceNumber': lot.expected_piece_number,
                'sessions': processed_sessions,
                'totalSessions': len(processed_sessions),
                'completedSessions': len(processed_sessions),  # All sessions are considered completed
                'successfulSessions': successful_sessions,
                'successRate': success_rate,
                'sessionSuccessRate': success_rate,
                'avgConfidence': avg_confidence,
                'lotMatchConfidence': avg_confidence,
                'createdAt': lot.created_at.isoformat() if lot.created_at else None,
                'lastActivity': max([s['timestamp'] for s in processed_sessions]) if processed_sessions else lot.created_at.isoformat()
            }
            
            grouped_lots[group_name].append(lot_data)
        
        # Calculate group statistics
        group_stats = {}
        for group_name, group_lots in grouped_lots.items():
            if group_lots:
                avg_success_rate = sum(lot['successRate'] for lot in group_lots) / len(group_lots)
                avg_confidence = sum(lot['avgConfidence'] for lot in group_lots) / len(group_lots)
                last_activity = max(lot['lastActivity'] for lot in group_lots)
                
                group_stats[group_name] = {
                    'avgSessionSuccessRate': avg_success_rate,
                    'avgLotConfidence': avg_confidence,
                    'lastActivity': last_activity
                }
        
        # Update overall statistics
        if lots:
            all_lots = [lot for group_lots in grouped_lots.values() for lot in group_lots]
            statistics['sessionSuccessRate'] = sum(lot['successRate'] for lot in all_lots) / len(all_lots)
            statistics['avgLotConfidence'] = sum(lot['avgConfidence'] for lot in all_lots) / len(all_lots)
            statistics['totalGroups'] = len(grouped_lots)
            statistics['activeGroups'] = len(grouped_lots)
        
        return {
            'success': True,
            'groupedLots': grouped_lots,
            'groupStats': group_stats,
            'statistics': statistics,
            'message': f'Retrieved {len(lots)} lots across {len(grouped_lots)} groups'
        }
        
    except Exception as e:
        logger.error(f"Error getting dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")
    
@basic_detection_router.get("/lotSession/lots/{lot_id}/sessions")
async def get_lot_sessions_with_pieces(lot_id: int, db: Session = Depends(get_session)):
    """Get all sessions for a lot with detected pieces data"""
    try:
        from detection.app.service.detection.alternative.basic_detection_service import basic_detection_processor
        
        # Use your existing service method
        sessions = basic_detection_processor.get_lot_sessions(lot_id, db)
        
        return {
            'success': True,
            'lot_id': lot_id,
            'sessions': sessions,
            'message': f'Retrieved {len(sessions)} sessions for lot {lot_id}'
        }
        
    except Exception as e:
        logger.error(f"Error getting lot sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get lot sessions: {str(e)}")


@basic_detection_router.get("/lotSession/sessions/{session_id}/pieces")
async def get_session_detected_pieces(session_id: int, db: Session = Depends(get_session)):
    """Get detected pieces for a specific session"""
    try:
        from detection.app.db.models.detectionSession import DetectionSession
        from sqlalchemy.orm import selectinload
        
        session = db.query(DetectionSession).options(
            selectinload(DetectionSession.detected_pieces)
        ).filter(DetectionSession.id == session_id).first()
        
        if not session:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        
        detected_pieces = []
        for piece in session.detected_pieces:
            piece_data = {
                'id': piece.id,
                'piece_id': piece.piece_id,
                'detected_label': piece.detected_label,
                'confidence_score': piece.confidence_score,
                'bounding_box_x1': piece.bounding_box_x1,
                'bounding_box_y1': piece.bounding_box_y1,
                'bounding_box_x2': piece.bounding_box_x2,
                'bounding_box_y2': piece.bounding_box_y2,
                'is_correct_piece': piece.is_correct_piece,
                'created_at': piece.created_at.isoformat() if piece.created_at else None
            }
            detected_pieces.append(piece_data)
        
        return {
            'success': True,
            'session_id': session_id,
            'detected_pieces': detected_pieces,
            'pieces_count': len(detected_pieces),
            'message': f'Retrieved {len(detected_pieces)} detected pieces for session {session_id}'
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session detected pieces: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get detected pieces: {str(e)}")    