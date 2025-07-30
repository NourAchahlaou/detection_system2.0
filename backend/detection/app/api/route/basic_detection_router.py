# basic_detection_router.py - On-demand detection router with stream coordination

import logging
import time
from dataclasses import asdict
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

from detection.app.service.alternative.basic_detection_service import (
    DetectionRequest, 
    DetectionResponse,
    basic_detection_processor
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the router
basic_detection_router = APIRouter(
    prefix="/basic",
    tags=["Basic Detection"],
    responses={404: {"description": "Not found"}},
)

@basic_detection_router.post("/detect/{camera_id}")
async def detect_on_stream(
    camera_id: int,
    request: Request
):
    """
    Perform on-demand detection on current video stream frame
    
    This endpoint:
    1. Freezes the current video stream
    2. Captures the current frame
    3. Runs detection on that frame
    4. Updates the frozen frame with detection results
    5. Returns detection results (stream remains frozen)
    
    Body should contain:
    {
        "target_label": "person",
        "quality": 85  // optional, JPEG quality for response
    }
    
    Note: Stream remains frozen after detection. Use /unfreeze endpoint to resume live feed.
    """
    try:
        # Parse request body
        body = await request.json()
        target_label = body.get('target_label')
        quality = body.get('quality', 85)
        
        if not target_label:
            raise HTTPException(status_code=400, detail="target_label is required")
        
        # Validate camera_id
        if camera_id < 0:
            raise HTTPException(status_code=400, detail="Invalid camera_id")
        
        # Validate quality
        if not isinstance(quality, int) or quality < 50 or quality > 100:
            quality = 85
        
        logger.info(f"🎯 On-demand detection request for camera {camera_id}, target: '{target_label}'")
        
        # Create detection request
        detection_request = DetectionRequest(
            camera_id=camera_id,
            target_label=target_label,
            timestamp=time.time(),
            quality=quality
        )
        
        # Perform detection (this will freeze stream, detect, and update frozen frame)
        response = await basic_detection_processor.detect_on_current_frame(detection_request)
        
        # Convert dataclass to dict for JSON response
        response_dict = asdict(response)
        
        # Return JSON response
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": response_dict,
                "message": "Detection completed successfully. Stream is frozen - use /unfreeze to resume live feed.",
                "stream_frozen": response.stream_frozen
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing detection request: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@basic_detection_router.post("/stream/{camera_id}/unfreeze")
async def unfreeze_stream_after_detection(camera_id: int):
    """
    Unfreeze the video stream to resume live feed after detection
    
    This endpoint unfreezes the stream to return to normal live video streaming
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

@basic_detection_router.get("/stream/{camera_id}/status")
async def get_detection_stream_status(camera_id: int):
    """
    Get current status of the video stream from detection service perspective
    """
    try:
        if camera_id < 0:
            raise HTTPException(status_code=400, detail="Invalid camera_id")
        
        status = basic_detection_processor.get_stream_status(camera_id)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "status": status,
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting stream status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stream status: {str(e)}")

@basic_detection_router.post("/detect/{camera_id}/with-unfreeze")
async def detect_and_unfreeze(
    camera_id: int,
    request: Request
):
    """
    Perform detection and automatically unfreeze stream afterwards
    
    This is a convenience endpoint that:
    1. Performs detection (freezes stream)
    2. Automatically unfreezes stream after detection
    3. Returns detection results
    
    Body should contain:
    {
        "target_label": "person",
        "quality": 85,  // optional
        "unfreeze_delay": 2.0  // optional, seconds to wait before unfreezing
    }
    """
    try:
        # Parse request body
        body = await request.json()
        target_label = body.get('target_label')
        quality = body.get('quality', 85)
        unfreeze_delay = body.get('unfreeze_delay', 2.0)
        
        if not target_label:
            raise HTTPException(status_code=400, detail="target_label is required")
        
        # Validate inputs
        if camera_id < 0:
            raise HTTPException(status_code=400, detail="Invalid camera_id")
        
        if not isinstance(quality, int) or quality < 50 or quality > 100:
            quality = 85
            
        if not isinstance(unfreeze_delay, (int, float)) or unfreeze_delay < 0:
            unfreeze_delay = 2.0
        
        logger.info(f"🎯 Detection with auto-unfreeze for camera {camera_id}, target: '{target_label}'")
        
        # Create detection request
        detection_request = DetectionRequest(
            camera_id=camera_id,
            target_label=target_label,
            timestamp=time.time(),
            quality=quality
        )
        
        # Perform detection
        response = await basic_detection_processor.detect_on_current_frame(detection_request)
        
        # Wait for specified delay then unfreeze
        import asyncio
        if unfreeze_delay > 0:
            await asyncio.sleep(unfreeze_delay)
        
        # Unfreeze stream
        await basic_detection_processor.unfreeze_stream(camera_id)
        
        # Convert dataclass to dict for JSON response
        response_dict = asdict(response)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": response_dict,
                "message": f"Detection completed and stream unfrozen after {unfreeze_delay}s delay",
                "auto_unfrozen": True,
                "unfreeze_delay": unfreeze_delay
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in detect and unfreeze: {e}")
        raise HTTPException(status_code=500, detail=f"Detection with unfreeze failed: {str(e)}")

@basic_detection_router.get("/health")
async def health_check():
    """
    Check if the basic detection service is healthy and ready
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
                "detections_performed": stats.get('detections_performed', 0),
                "targets_detected": stats.get('targets_detected', 0),
                "avg_processing_time": stats.get('avg_processing_time', 0),
                "message": "Basic detection service is operational"
            }
        )
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "message": "Basic detection service is not operational"
            }
        )

@basic_detection_router.post("/initialize")
async def initialize_processor():
    """
    Initialize the basic detection processor
    """
    try:
        logger.info("🚀 Initializing basic detection processor...")
        
        await basic_detection_processor.initialize()
        
        stats = basic_detection_processor.get_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Basic detection processor initialized successfully",
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
    Get basic detection service statistics
    """
    try:
        stats = basic_detection_processor.get_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "stats": stats,
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@basic_detection_router.post("/detect/batch")
async def detect_multiple_streams(request: Request):
    """
    Perform detection on multiple camera streams
    
    Body should contain:
    {
        "detections": [
            {
                "camera_id": 0,
                "target_label": "person",
                "quality": 85,
                "auto_unfreeze": true,
                "unfreeze_delay": 2.0
            }
        ]
    }
    """
    try:
        body = await request.json()
        detections = body.get('detections', [])
        
        if not detections or not isinstance(detections, list):
            raise HTTPException(
                status_code=400, 
                detail="detections array is required"
            )
        
        if len(detections) > 3:  # Limit batch size for basic mode
            raise HTTPException(
                status_code=400, 
                detail="Maximum 3 detections per batch in basic mode"
            )
        
        logger.info(f"🎯 Batch detection request for {len(detections)} streams")
        
        results = []
        
        for i, detection_data in enumerate(detections):
            try:
                camera_id = detection_data.get('camera_id')
                target_label = detection_data.get('target_label')
                quality = detection_data.get('quality', 85)
                auto_unfreeze = detection_data.get('auto_unfreeze', False)
                unfreeze_delay = detection_data.get('unfreeze_delay', 2.0)
                
                if camera_id is None or not target_label:
                    results.append({
                        "index": i,
                        "camera_id": camera_id,
                        "success": False,
                        "error": "camera_id and target_label are required"
                    })
                    continue
                
                # Create detection request
                detection_request = DetectionRequest(
                    camera_id=camera_id,
                    target_label=target_label,
                    timestamp=time.time(),
                    quality=quality
                )
                
                # Perform detection
                response = await basic_detection_processor.detect_on_current_frame(detection_request)
                
                # Auto-unfreeze if requested
                if auto_unfreeze:
                    import asyncio
                    if unfreeze_delay > 0:
                        await asyncio.sleep(unfreeze_delay)
                    await basic_detection_processor.unfreeze_stream(camera_id)
                
                results.append({
                    "index": i,
                    "camera_id": camera_id,
                    "success": True,
                    "data": asdict(response),
                    "auto_unfrozen": auto_unfreeze
                })
                
            except Exception as e:
                logger.error(f"❌ Error in batch detection item {i}: {e}")
                results.append({
                    "index": i,
                    "camera_id": detection_data.get('camera_id'),
                    "success": False,
                    "error": str(e)
                })
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "results": results,
                "total_processed": len(results),
                "successful": sum(1 for r in results if r['success']),
                "failed": sum(1 for r in results if not r['success'])
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing batch detection: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Batch detection failed: {str(e)}"
        )