# identification_router.py - Updated router with new initialization flow

import logging
import time
from typing import Dict, Any, Optional, List
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from detection.app.service.identification.identification_detection_service import piece_identification_processor

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models for request validation
class IdentificationRequest(BaseModel):
    group_name: Optional[str] = Field(None, min_length=1, max_length=50, description="Group name for piece identification (optional if group already selected)")
    freeze_stream: bool = Field(True, description="Whether to freeze the stream during identification")
    quality: int = Field(85, ge=50, le=100, description="JPEG quality for response")

class GroupSwitchRequest(BaseModel):
    group_name: str = Field(..., min_length=1, max_length=50, description="Group name to switch to")

class GroupSelectionRequest(BaseModel):
    group_name: str = Field(..., min_length=1, max_length=50, description="Group name to select")

class BatchIdentificationRequest(BaseModel):
    group_name: Optional[str] = Field(None, min_length=1, max_length=50, description="Group name for piece identification (optional if group already selected)")
    num_frames: int = Field(5, ge=1, le=20, description="Number of frames to capture for batch analysis")
    interval_seconds: float = Field(1.0, ge=0.1, le=10.0, description="Interval between frame captures")

class ConfidenceThresholdRequest(BaseModel):
    threshold: float = Field(..., ge=0.1, le=1.0, description="Confidence threshold for identification (0.1 to 1.0)")

class QuickAnalysisRequest(BaseModel):
    group_name: Optional[str] = Field(None, min_length=1, max_length=50, description="Group name for piece identification (optional if group already selected)")
    quality: int = Field(85, ge=50, le=100, description="JPEG quality for response")

# Create the router
identification_router = APIRouter(
    prefix="/identification",
    tags=["Piece Identification"],
    responses={404: {"description": "Not found"}},
)

@identification_router.post("/initialize")
async def initialize_processor():
    """
    Initialize the identification processor (basic setup without group)
    
    This initializes the basic processor infrastructure.
    After initialization, you need to select a group using /groups/select before identification.
    """
    try:
        logger.info("🚀 Initializing identification processor (basic setup)")
        
        await piece_identification_processor.initialize()
        
        stats = piece_identification_processor.get_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Identification processor initialized successfully",
                "is_initialized": stats['is_initialized'],
                "is_group_loaded": stats.get('is_group_loaded', False),
                "current_group": stats.get('current_group_name', None),
                "device": stats.get('device', 'unknown'),
                "next_step": "Select a group using /groups/select/{group_name} before performing identification"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error initializing processor: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to initialize processor: {str(e)}"
        )

@identification_router.post("/groups/select/{group_name}")
async def select_group(group_name: str):
    """
    Select/switch to a specific group for identification
    
    This loads the model for the specified group and makes it ready for identification.
    You can call this multiple times to switch between different groups.
    """
    try:
        logger.info(f"🔄 Selecting group: {group_name}")
        
        # Ensure basic initialization
        if not piece_identification_processor.is_initialized:
            await piece_identification_processor.initialize()
        
        # Switch to the group
        success = await piece_identification_processor.switch_group(group_name)
        
        if not success:
            raise HTTPException(status_code=400, detail=f"Failed to select group: {group_name}")
        
        stats = piece_identification_processor.get_stats()
        model_info = piece_identification_processor.get_model_info()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Successfully selected group: {group_name}",
                "current_group": stats['current_group_name'],
                "is_group_loaded": stats['is_group_loaded'],
                "device": stats.get('device', 'unknown'),
                "model_info": model_info,
                "status": "Ready for identification",
                "available_operations": [
                    "POST /identification/identify/{camera_id}",
                    "POST /identification/analyze/{camera_id}", 
                    "POST /identification/batch/{camera_id}"
                ]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error selecting group {group_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to select group {group_name}: {str(e)}")
    
@identification_router.post("/identify/{camera_id}")
async def identify_pieces_for_group(camera_id: int, request: IdentificationRequest):
    """
    Identify all pieces visible in the camera feed for a specific group
    
    This endpoint:
    1. Initializes the identification system for the specified group
    2. Captures the current frame from the specified camera
    3. Optionally freezes the stream for analysis
    4. Identifies all pieces in the frame that belong to the specified group
    5. Returns detailed information about each piece found
    6. Updates the frozen frame with identification overlay
    
    No lot tracking is involved - this is pure piece identification for group classification.
    """
    try:
        if camera_id < 0:
            raise HTTPException(status_code=400, detail="Invalid camera_id")
        
        logger.info(f"🔍 Starting piece identification for camera {camera_id}, group: '{request.group_name}'")
        
        # Perform identification with frame capture for the specified group
        result = await piece_identification_processor.identify_with_frame_capture(
            camera_id=camera_id,
            group_name=request.group_name,
            freeze_stream=request.freeze_stream,
            quality=request.quality
        )
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Identification failed'))
        
        # Generate summary
        pieces = result['identification_result']['pieces']
        summary = piece_identification_processor.get_identification_summary(pieces)
        
        # Prepare response message
        message_parts = [f"Identification completed for group '{request.group_name}' - Found {summary['total_pieces']} pieces"]
        if summary['unique_labels']:
            labels_str = ', '.join([f"{label}({count})" for label, count in summary['label_counts'].items()])
            message_parts.append(f"Labels: {labels_str}")
        if result['stream_frozen']:
            message_parts.append("Stream is frozen - use /unfreeze to resume")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "camera_id": camera_id,
                "group_name": request.group_name,
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

@identification_router.post("/identify/{camera_id}/switch-group")
async def switch_group_and_identify(camera_id: int, request: GroupSwitchRequest):
    """
    Switch to a different group model and perform identification
    
    This endpoint allows you to change the group model and immediately perform
    identification with the new group. Useful for identifying pieces from
    different product lines or categories.
    """
    try:
        if camera_id < 0:
            raise HTTPException(status_code=400, detail="Invalid camera_id")
        
        logger.info(f"🔄 Switching to group '{request.new_group_name}' and identifying for camera {camera_id}")
        
        # Switch group and perform identification
        result = await piece_identification_processor.switch_group_and_identify(
            camera_id=camera_id,
            new_group_name=request.new_group_name,
            freeze_stream=request.freeze_stream,
            quality=request.quality
        )
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Group switch and identification failed'))
        
        # Generate summary
        pieces = result['identification_result']['pieces']
        summary = piece_identification_processor.get_identification_summary(pieces)
        
        # Prepare response message
        message_parts = [f"Switched to group '{request.new_group_name}' and identified {summary['total_pieces']} pieces"]
        if summary['unique_labels']:
            labels_str = ', '.join([f"{label}({count})" for label, count in summary['label_counts'].items()])
            message_parts.append(f"Labels: {labels_str}")
        if result['stream_frozen']:
            message_parts.append("Stream is frozen - use /unfreeze to resume")
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "camera_id": camera_id,
                "previous_group": piece_identification_processor.stats.get('previous_group', 'unknown'),
                "current_group": request.new_group_name,
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
        logger.error(f"❌ Error in group switch and identification: {e}")
        raise HTTPException(status_code=500, detail=f"Group switch and identification failed: {str(e)}")

@identification_router.post("/analyze/{camera_id}")
async def analyze_current_view_for_group(camera_id: int, request: QuickAnalysisRequest):
    """
    Quick analysis of current camera view for a specific group without freezing
    
    This is a lighter version of identification that:
    - Doesn't freeze the stream by default
    - Focuses on quick piece analysis for the specified group
    - Useful for real-time identification feedback
    - Returns only essential information for fast processing
    """
    try:
        if camera_id < 0:
            raise HTTPException(status_code=400, detail="Invalid camera_id")
        
        logger.info(f"🔍 Quick analysis for camera {camera_id}, group: '{request.group_name}'")
        
        # Perform identification without freezing for the specified group
        result = await piece_identification_processor.identify_with_frame_capture(
            camera_id=camera_id,
            group_name=request.group_name,
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
                "group_name": request.group_name,
                "analysis_type": "quick_view",
                "pieces_found": summary['total_pieces'],
                "pieces": pieces,
                "summary": summary,
                "processing_time_ms": result['processing_time_ms'],
                "timestamp": result['timestamp'],
                "stream_frozen": False,
                "message": f"Quick analysis complete for group '{request.group_name}' - {summary['total_pieces']} pieces identified"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in quick analysis: {e}")
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@identification_router.post("/batch/{camera_id}")
async def batch_identify_pieces(camera_id: int, request: BatchIdentificationRequest):
    """
    Perform batch identification over multiple frames for better accuracy
    
    This endpoint captures multiple frames over time and analyzes them all
    for the specified group to provide more comprehensive identification results.
    Useful when you need higher confidence in identification results.
    """
    try:
        if camera_id < 0:
            raise HTTPException(status_code=400, detail="Invalid camera_id")
        
        logger.info(f"📦 Starting batch identification for camera {camera_id}, group: '{request.group_name}', {request.num_frames} frames")
        
        # Perform batch identification
        result = await piece_identification_processor.batch_identify_frames(
            camera_id=camera_id,
            group_name=request.group_name,
            num_frames=request.num_frames,
            interval_seconds=request.interval_seconds
        )
        
        if not result['success']:
            raise HTTPException(status_code=500, detail=result.get('error', 'Batch identification failed'))
        
        batch_result = result['batch_identification_result']
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "camera_id": camera_id,
                "group_name": request.group_name,
                "batch_identification_result": batch_result,
                "frames_processed": batch_result['frames_processed'],
                "total_pieces_found": batch_result['total_pieces_found'],
                "unique_labels": batch_result['unique_labels_found'],
                "average_pieces_per_frame": batch_result['average_pieces_per_frame'],
                "average_confidence": batch_result['average_confidence'],
                "timestamp": result['timestamp'],
                "message": f"Batch identification completed for group '{request.group_name}' - {batch_result['total_pieces_found']} total pieces across {batch_result['frames_processed']} frames"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in batch identification: {e}")
        raise HTTPException(status_code=500, detail=f"Batch identification failed: {str(e)}")

@identification_router.get("/groups/available")
async def get_available_groups():
    """
    Get list of available groups for piece identification
    
    Returns all group models that are available for identification.
    Each group represents a different category or product line of pieces.
    """
    try:
        available_groups = piece_identification_processor.get_available_groups()
        current_group = piece_identification_processor.current_group_name
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "available_groups": available_groups,
                "current_group": current_group,
                "total_groups": len(available_groups),
                "message": f"Found {len(available_groups)} available groups for identification"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting available groups: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get available groups: {str(e)}")

@identification_router.get("/groups/{group_name}/piece-types")
async def get_piece_types_for_group(group_name: str):
    """
    Get available piece types that can be identified for a specific group
    
    This endpoint returns information about what types of pieces
    the specified group model can identify.
    """
    try:
        # Initialize with the specified group if not already initialized
        if not piece_identification_processor.is_initialized or piece_identification_processor.current_group_name != group_name:
            await piece_identification_processor.initialize(group_name)
        
        # Get model class names
        class_names = piece_identification_processor.detection_system.model.names
        
        # Get stats to show what pieces have been identified recently for this group
        stats = piece_identification_processor.get_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "group_name": group_name,
                "available_piece_types": list(class_names.values()),
                "total_classes": len(class_names),
                "recently_identified_pieces": stats['unique_pieces_list'],
                "identification_stats": {
                    "total_identifications": stats['identifications_performed'],
                    "unique_pieces_identified": stats['unique_pieces_count']
                },
                "confidence_threshold": stats['confidence_threshold'],
                "message": f"Group '{group_name}' can identify {len(class_names)} different piece types"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting piece types for group {group_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get piece types for group {group_name}: {str(e)}")

@identification_router.put("/settings/confidence-threshold")
async def update_confidence_threshold(request: ConfidenceThresholdRequest):
    """
    Update the confidence threshold for piece identification
    
    Lower values will identify more pieces but with potentially lower accuracy.
    Higher values will be more selective but might miss some pieces.
    This setting applies to all groups.
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
                "applies_to": "all_groups",
                "message": f"Confidence threshold updated to {request.threshold} for all groups",
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
        model_info = piece_identification_processor.get_model_info()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "current_settings": {
                    "confidence_threshold": stats['confidence_threshold'],
                    "current_group": stats['current_group'],
                    "is_initialized": stats['is_initialized'],
                    "device": stats['device']
                },
                "model_info": model_info,
                "capabilities": {
                    "supports_multiple_groups": True,
                    "can_switch_groups_dynamically": True,
                    "can_identify_multiple_pieces": True,
                    "provides_confidence_scores": True,
                    "provides_bounding_boxes": True,
                    "supports_real_time_analysis": True,
                    "supports_batch_processing": True
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
    This endpoint is identical to the detection service unfreeze functionality.
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
    Get recent identification history across all groups
    
    Returns the last 10 identification operations with their results,
    including which group was used for each identification.
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
                "current_group": stats['current_group'],
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
        model_info = piece_identification_processor.get_model_info()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "stats": stats,
                "model_info": model_info,
                "timestamp": time.time(),
                "service_type": "piece_identification_with_groups",
                "message": "Identification service statistics retrieved successfully"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@identification_router.get("/health")
async def health_check():
    """
    Check if the identification service is healthy and ready
    """
    try:
        # Check if processor is initialized
        stats = piece_identification_processor.get_stats()
        model_info = piece_identification_processor.get_model_info()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy" if stats['is_initialized'] else "initializing",
                "is_initialized": stats['is_initialized'],
                "current_group": stats.get('current_group', 'none'),
                "device": stats.get('device', 'unknown'),
                "model_loaded": model_info.get('is_loaded', False),
                "statistics": {
                    "identifications_performed": stats.get('identifications_performed', 0),
                    "total_pieces_identified": stats.get('total_pieces_identified', 0),
                    "unique_pieces_count": stats.get('unique_pieces_count', 0),
                    "avg_processing_time": stats.get('avg_processing_time', 0)
                },
                "message": "Piece identification service with group support is operational"
            }
        )
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "message": "Piece identification service is not operational"
            }
        )

@identification_router.post("/initialize_with_group")
async def initialize_processor_with_group():
    """
    Initialize the identification processor with a specific group
    
    Body should contain:
    {
        "group_name": "E539"
    }
    """
    try:
        # Parse request body
        request = await Request.json() if hasattr(Request, 'json') else {}
        group_name = request.get('group_name') if request else None
        
        if not group_name:
            raise HTTPException(status_code=400, detail="group_name is required for initialization")
        
        logger.info(f"🚀 Initializing identification processor with group: {group_name}")
        
        await piece_identification_processor.initialize(group_name)
        
        stats = piece_identification_processor.get_stats()
        model_info = piece_identification_processor.get_model_info()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": f"Identification processor initialized successfully with group '{group_name}'",
                "group_name": group_name,
                "device": stats.get('device', 'unknown'),
                "is_initialized": stats['is_initialized'],
                "model_info": model_info
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error initializing processor: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to initialize processor: {str(e)}"
        )

# Additional endpoint for backward compatibility
@identification_router.post("/identify/{camera_id}/legacy")
async def identify_pieces_legacy(camera_id: int, request: IdentificationRequest):
    """
    Legacy endpoint for piece identification
    
    This endpoint provides backward compatibility with the old identification format.
    It's identical to the main identification endpoint but maintains the old response structure.
    """
    try:
        # Use the main identification method
        result = await identify_pieces_for_group(camera_id, request)
        return result
        
    except Exception as e:
        logger.error(f"❌ Error in legacy identification: {e}")
        raise HTTPException(status_code=500, detail=f"Legacy identification failed: {str(e)}")

@identification_router.get("/groups/{group_name}/stats")
async def get_group_specific_stats(group_name: str):
    """
    Get statistics specific to a particular group
    
    This endpoint provides detailed statistics about identifications
    performed with the specified group model.
    """
    try:
        # This would require group-specific tracking in the processor
        # For now, return current stats with group context
        stats = piece_identification_processor.get_stats()
        
        # Filter recent identifications for the specified group
        group_identifications = [
            identification for identification in stats['recent_identifications']
            if identification.get('group_name') == group_name
        ]
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "group_name": group_name,
                "is_current_group": stats.get('current_group') == group_name,
                "group_specific_identifications": group_identifications,
                "total_group_identifications": len(group_identifications),
                "overall_stats": stats,
                "message": f"Statistics for group '{group_name}' retrieved successfully"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting group stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats for group {group_name}: {str(e)}")