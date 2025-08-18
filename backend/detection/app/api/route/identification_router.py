# identification_router.py - Router for piece identification functionality

import logging
import time
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from detection.app.service.identification.identification_detection_service import piece_identification_processor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request validation
class IdentificationRequest(BaseModel):
    freeze_stream: bool = Field(True, description="Whether to freeze the stream during identification")
    quality: int = Field(85, ge=50, le=100, description="JPEG quality for response")

class ConfidenceThresholdRequest(BaseModel):
    threshold: float = Field(..., ge=0.1, le=1.0, description="Confidence threshold for identification (0.1 to 1.0)")

class PieceAnalysisRequest(BaseModel):
    analyze_frame_only: bool = Field(False, description="Only analyze the frame without capturing new one")
    quality: int = Field(85, ge=50, le=100, description="JPEG quality for response")

# Create the router
identification_router = APIRouter(
    prefix="/identification",
    tags=["Piece Identification"],
    responses={404: {"description": "Not found"}},
)

@identification_router.post("/identify/{camera_id}")
async def identify_pieces(camera_id: int, request: IdentificationRequest):
    """
    Identify all pieces visible in the camera feed
    
    This endpoint:
    1. Captures the current frame from the specified camera
    2. Optionally freezes the stream for analysis
    3. Identifies all pieces in the frame
    4. Returns detailed information about each piece found
    5. Updates the frozen frame with identification overlay
    
    No lot tracking is involved - this is pure piece identification.
    """
    try:
        if camera_id < 0:
            raise HTTPException(status_code=400, detail="Invalid camera_id")
        
        logger.info(f"🔍 Starting piece identification for camera {camera_id}")
        
        # Perform identification with frame capture
        result = await piece_identification_processor.identify_with_frame_capture(
            camera_id=camera_id,
            freeze_stream=request.freeze_stream,
            quality=request.quality
        )
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Identification failed'))
        
        # Generate summary
        pieces = result['identification_result']['pieces']
        summary = piece_identification_processor.get_identification_summary(pieces)
        
        # Prepare response message
        message_parts = [f"Identification completed - Found {summary['total_pieces']} pieces"]
        if summary['unique_labels']:
            labels_str = ', '.join([f"{label}({count})" for label, count in summary['label_counts'].items()])
            message_parts.append(f"Labels: {labels_str}")
        if request.freeze_stream:
            message_parts.append("Stream is frozen - use /unfreeze to resume")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "camera_id": camera_id,
                "identification_result": result['identification_result'],
                "summary": summary,
                "processing_time_ms": result['processing_time_ms'],
                "frame_with_overlay": result['frame_with_overlay'],
                "timestamp": result['timestamp'],
                "stream_frozen": result['stream_frozen'],
                "message": " | ".join(message_parts)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in piece identification: {e}")
        raise HTTPException(status_code=500, detail=f"Identification failed: {str(e)}")

@identification_router.post("/analyze/{camera_id}")
async def analyze_current_view(camera_id: int, request: PieceAnalysisRequest):
    """
    Quick analysis of current camera view without freezing
    
    This is a lighter version of identification that:
    - Doesn't freeze the stream by default
    - Focuses on quick piece analysis
    - Useful for real-time identification feedback
    """
    try:
        if camera_id < 0:
            raise HTTPException(status_code=400, detail="Invalid camera_id")
        
        logger.info(f"🔍 Quick analysis for camera {camera_id}")
        
        # Perform identification without freezing
        result = await piece_identification_processor.identify_with_frame_capture(
            camera_id=camera_id,
            freeze_stream=False,  # Don't freeze for quick analysis
            quality=request.quality
        )
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Analysis failed'))
        
        # Generate summary
        pieces = result['identification_result']['pieces']
        summary = piece_identification_processor.get_identification_summary(pieces)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "camera_id": camera_id,
                "analysis_type": "quick_view",
                "pieces_found": summary['total_pieces'],
                "pieces": pieces,
                "summary": summary,
                "processing_time_ms": result['processing_time_ms'],
                "timestamp": result['timestamp'],
                "message": f"Quick analysis complete - {summary['total_pieces']} pieces identified"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in quick analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@identification_router.get("/piece-types")
async def get_available_piece_types():
    """
    Get available piece types that can be identified
    
    This endpoint returns information about the model's capabilities
    and what types of pieces it can identify.
    """
    try:
        if not piece_identification_processor.is_initialized:
            await piece_identification_processor.initialize()
        
        # Get model class names
        class_names = piece_identification_processor.detection_system.model.names
        
        # Get stats to show what pieces have been identified recently
        stats = piece_identification_processor.get_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "available_piece_types": list(class_names.values()),
                "total_classes": len(class_names),
                "recently_identified_pieces": stats['unique_pieces_list'],
                "identification_stats": {
                    "total_identifications": stats['identifications_performed'],
                    "unique_pieces_identified": stats['unique_pieces_count']
                },
                "confidence_threshold": stats['confidence_threshold'],
                "message": f"Model can identify {len(class_names)} different piece types"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting piece types: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get piece types: {str(e)}")

@identification_router.put("/settings/confidence-threshold")
async def update_confidence_threshold(request: ConfidenceThresholdRequest):
    """
    Update the confidence threshold for piece identification
    
    Lower values will identify more pieces but with potentially lower accuracy.
    Higher values will be more selective but might miss some pieces.
    """
    try:
        success = piece_identification_processor.set_confidence_threshold(request.threshold)
        
        if not success:
            raise HTTPException(status_code=400, detail="Invalid confidence threshold")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "new_threshold": request.threshold,
                "message": f"Confidence threshold updated to {request.threshold}",
                "effect": "Lower threshold = more detections, Higher threshold = more selective"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error updating confidence threshold: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update threshold: {str(e)}")

@identification_router.get("/settings")
async def get_identification_settings():
    """
    Get current identification settings and capabilities
    """
    try:
        stats = piece_identification_processor.get_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "settings": {
                    "confidence_threshold": stats['confidence_threshold'],
                    "is_initialized": stats['is_initialized'],
                    "device": stats['device']
                },
                "capabilities": {
                    "can_identify_multiple_pieces": True,
                    "provides_confidence_scores": True,
                    "provides_bounding_boxes": True,
                    "supports_real_time_analysis": True
                },
                "performance": {
                    "avg_processing_time_ms": stats['avg_processing_time']
                }
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting settings: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get settings: {str(e)}")

@identification_router.post("/stream/{camera_id}/unfreeze")
async def unfreeze_identification_stream(camera_id: int):
    """
    Unfreeze the video stream to resume live feed after identification
    
    Use this after identification operations that freeze the stream.
    """
    try:
        if camera_id < 0:
            raise HTTPException(status_code=400, detail="Invalid camera_id")
        
        logger.info(f"🔥 Unfreezing stream for camera {camera_id}")
        
        success = await piece_identification_processor.unfreeze_stream(camera_id)
        
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

@identification_router.get("/history")
async def get_identification_history():
    """
    Get recent identification history
    
    Returns the last 10 identification operations with their results.
    """
    try:
        stats = piece_identification_processor.get_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "recent_identifications": stats['recent_identifications'],
                "total_identifications": stats['identifications_performed'],
                "unique_pieces_identified": stats['unique_pieces_list'],
                "message": f"Retrieved {len(stats['recent_identifications'])} recent identification records"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting identification history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get history: {str(e)}")

@identification_router.get("/stats")
async def get_identification_stats():
    """
    Get detailed identification service statistics
    """
    try:
        stats = piece_identification_processor.get_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "stats": stats,
                "timestamp": time.time(),
                "service_type": "piece_identification",
                "message": "Identification service statistics retrieved successfully"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")
