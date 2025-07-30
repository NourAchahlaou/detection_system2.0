# redis_basic_detection_router.py - Redis-coordinated on-demand detection router

import logging
import time
from dataclasses import asdict
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse

# Import Redis-based detection service (update this import to match your actual structure)
from detection.app.service.alternative.basic_detection_service import (
    DetectionRequest, 
    DetectionResponse,
    redis_basic_detection_processor
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the router
redis_basic_detection_router = APIRouter(
    prefix="/redis-basic",
    tags=["Redis Basic Detection"],
    responses={404: {"description": "Not found"}},
)

@redis_basic_detection_router.post("/detect/{camera_id}")
async def redis_detect_on_stream(
    camera_id: int,
    request: Request
):
    """
    Perform Redis-coordinated on-demand detection on current video stream frame
    
    This endpoint uses Redis for cross-container communication and:
    1. Sends freeze command to video streaming service via Redis
    2. Requests current frame from frozen stream via Redis
    3. Runs detection on that frame
    4. Updates the frozen frame with detection results via Redis
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
        
        logger.info(f"🎯 Redis-coordinated detection request for camera {camera_id}, target: '{target_label}'")
        
        # Create detection request
        detection_request = DetectionRequest(
            camera_id=camera_id,
            target_label=target_label,
            timestamp=time.time(),
            quality=quality
        )
        
        # Perform Redis-coordinated detection
        response = await redis_basic_detection_processor.detect_on_current_frame(detection_request)
        
        # Convert dataclass to dict for JSON response
        response_dict = asdict(response)
        
        # Return JSON response
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": response_dict,
                "message": "Redis-coordinated detection completed successfully. Stream is frozen - use /unfreeze to resume live feed.",
                "stream_frozen": response.stream_frozen,
                "coordination": "redis",
                "redis_operations": "freeze, get_frame, update_frozen_frame"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing Redis detection request: {e}")
        raise HTTPException(status_code=500, detail=f"Redis detection failed: {str(e)}")

@redis_basic_detection_router.post("/stream/{camera_id}/unfreeze")
async def redis_unfreeze_stream_after_detection(camera_id: int):
    """
    Unfreeze the video stream to resume live feed after detection via Redis
    
    This endpoint sends an unfreeze command via Redis to return to normal live video streaming
    """
    try:
        if camera_id < 0:
            raise HTTPException(status_code=400, detail="Invalid camera_id")
        
        logger.info(f"🔥 Redis unfreezing stream for camera {camera_id}")
        
        success = await redis_basic_detection_processor.unfreeze_stream(camera_id)
        
        if success:
            return JSONResponse(
                status_code=200,
                content={
                    "success": True,
                    "camera_id": camera_id,
                    "message": f"Stream unfrozen for camera {camera_id} via Redis. Live feed resumed.",
                    "coordination": "redis",
                    "timestamp": time.time()
                }
            )
        else:
            raise HTTPException(
                status_code=404, 
                detail=f"No active stream found for camera {camera_id} or failed to unfreeze via Redis"
            )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error unfreezing stream via Redis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unfreeze stream via Redis: {str(e)}")

@redis_basic_detection_router.get("/stream/{camera_id}/status")
async def get_redis_detection_stream_status(camera_id: int):
    """
    Get current status of the video stream via Redis from detection service perspective
    """
    try:
        if camera_id < 0:
            raise HTTPException(status_code=400, detail="Invalid camera_id")
        
        status = redis_basic_detection_processor.get_stream_status(camera_id)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "status": status,
                "coordination": "redis",
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting stream status via Redis: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stream status via Redis: {str(e)}")

@redis_basic_detection_router.post("/detect/{camera_id}/with-unfreeze")
async def redis_detect_and_unfreeze(
    camera_id: int,
    request: Request
):
    """
    Perform Redis-coordinated detection and automatically unfreeze stream afterwards
    
    This is a convenience endpoint that:
    1. Performs detection via Redis coordination (freezes stream)
    2. Automatically unfreezes stream after detection via Redis
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
        
        logger.info(f"🎯 Redis detection with auto-unfreeze for camera {camera_id}, target: '{target_label}'")
        
        # Create detection request
        detection_request = DetectionRequest(
            camera_id=camera_id,
            target_label=target_label,
            timestamp=time.time(),
            quality=quality
        )
        
        # Perform Redis-coordinated detection
        response = await redis_basic_detection_processor.detect_on_current_frame(detection_request)
        
        # Wait for specified delay then unfreeze via Redis
        import asyncio
        if unfreeze_delay > 0:
            await asyncio.sleep(unfreeze_delay)
        
        # Unfreeze stream via Redis
        await redis_basic_detection_processor.unfreeze_stream(camera_id)
        
        # Convert dataclass to dict for JSON response
        response_dict = asdict(response)
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": response_dict,
                "message": f"Redis-coordinated detection completed and stream unfrozen after {unfreeze_delay}s delay",
                "auto_unfrozen": True,
                "unfreeze_delay": unfreeze_delay,
                "coordination": "redis"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error in Redis detect and unfreeze: {e}")
        raise HTTPException(status_code=500, detail=f"Redis detection with unfreeze failed: {str(e)}")

@redis_basic_detection_router.get("/health")
async def redis_health_check():
    """
    Check if the Redis-coordinated basic detection service is healthy and ready
    """
    try:
        # Check if processor is initialized
        stats = redis_basic_detection_processor.get_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy" if stats['is_initialized'] else "initializing",
                "is_initialized": stats['is_initialized'],
                "device": stats.get('device', 'unknown'),
                "redis_connected": stats.get('redis_connected', False),
                "detections_performed": stats.get('detections_performed', 0),
                "targets_detected": stats.get('targets_detected', 0),
                "avg_processing_time": stats.get('avg_processing_time', 0),
                "redis_commands_sent": stats.get('redis_commands_sent', 0),
                "stream_operations": stats.get('stream_operations', 0),
                "coordination": "redis",
                "message": "Redis basic detection service is operational"
            }
        )
    except Exception as e:
        logger.error(f"❌ Redis health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "coordination": "redis",
                "message": "Redis basic detection service is not operational"
            }
        )

@redis_basic_detection_router.post("/initialize")
async def initialize_redis_processor():
    """
    Initialize the Redis-coordinated basic detection processor
    """
    try:
        logger.info("🚀 Initializing Redis basic detection processor...")
        
        await redis_basic_detection_processor.initialize()
        
        stats = redis_basic_detection_processor.get_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Redis basic detection processor initialized successfully",
                "device": stats.get('device', 'unknown'),
                "is_initialized": stats['is_initialized'],
                "redis_connected": stats.get('redis_connected', False),
                "coordination": "redis"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error initializing Redis processor: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to initialize Redis processor: {str(e)}"
        )

@redis_basic_detection_router.get("/stats")
async def get_redis_detection_stats():
    """
    Get Redis-coordinated basic detection service statistics
    """
    try:
        stats = redis_basic_detection_processor.get_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "stats": stats,
                "coordination": "redis",
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting Redis stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Redis stats: {str(e)}")

@redis_basic_detection_router.post("/detect/batch")
async def redis_detect_multiple_streams(request: Request):
    """
    Perform Redis-coordinated detection on multiple camera streams
    
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
                detail="Maximum 3 detections per batch in Redis basic mode"
            )
        
        logger.info(f"🎯 Redis batch detection request for {len(detections)} streams")
        
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
                        "error": "camera_id and target_label are required",
                        "coordination": "redis"
                    })
                    continue
                
                # Create detection request
                detection_request = DetectionRequest(
                    camera_id=camera_id,
                    target_label=target_label,
                    timestamp=time.time(),
                    quality=quality
                )
                
                # Perform Redis-coordinated detection
                response = await redis_basic_detection_processor.detect_on_current_frame(detection_request)
                
                # Auto-unfreeze if requested
                if auto_unfreeze:
                    import asyncio
                    if unfreeze_delay > 0:
                        await asyncio.sleep(unfreeze_delay)
                    await redis_basic_detection_processor.unfreeze_stream(camera_id)
                
                results.append({
                    "index": i,
                    "camera_id": camera_id,
                    "success": True,
                    "data": asdict(response),
                    "auto_unfrozen": auto_unfreeze,
                    "coordination": "redis"
                })
                
            except Exception as e:
                logger.error(f"❌ Error in Redis batch detection item {i}: {e}")
                results.append({
                    "index": i,
                    "camera_id": detection_data.get('camera_id'),
                    "success": False,
                    "error": str(e),
                    "coordination": "redis"
                })
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "results": results,
                "total_processed": len(results),
                "successful": sum(1 for r in results if r['success']),
                "failed": sum(1 for r in results if not r['success']),
                "coordination": "redis"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing Redis batch detection: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Redis batch detection failed: {str(e)}"
        )

@redis_basic_detection_router.post("/cleanup")
async def cleanup_redis_detection():
    """
    Cleanup Redis-coordinated detection service resources
    """
    try:
        logger.info("🧹 Cleaning up Redis detection service...")
        
        await redis_basic_detection_processor.cleanup()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Redis detection service cleaned up successfully",
                "coordination": "redis",
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error cleaning up Redis detection service: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to cleanup Redis detection: {str(e)}"
        )

@redis_basic_detection_router.get("/redis/connection-test")
async def test_redis_connection():
    """
    Test Redis connection for detection service
    """
    try:
        # Test Redis connections
        redis_status = {
            "async_redis": False,
            "sync_redis": False,
            "error": None
        }
        
        try:
            if redis_basic_detection_processor.async_redis_client:
                await redis_basic_detection_processor.async_redis_client.ping()
                redis_status["async_redis"] = True
        except Exception as e:
            redis_status["error"] = f"Async Redis error: {str(e)}"
        
        try:
            if redis_basic_detection_processor.sync_redis_client:
                redis_basic_detection_processor.sync_redis_client.ping()
                redis_status["sync_redis"] = True
        except Exception as e:
            if not redis_status["error"]:
                redis_status["error"] = f"Sync Redis error: {str(e)}"
            else:
                redis_status["error"] += f" | Sync Redis error: {str(e)}"
        
        overall_status = redis_status["async_redis"] and redis_status["sync_redis"]
        
        return JSONResponse(
            status_code=200 if overall_status else 503,
            content={
                "success": overall_status,
                "redis_connections": redis_status,
                "message": "Redis connections healthy" if overall_status else "Redis connection issues detected",
                "coordination": "redis",
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error testing Redis connection: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e),
                "message": "Failed to test Redis connections",
                "coordination": "redis",
                "timestamp": time.time()
            }
        )

@redis_basic_detection_router.get("/redis/stats")
async def get_redis_coordination_stats():
    """
    Get Redis coordination statistics for detection service
    """
    try:
        stats = redis_basic_detection_processor.get_stats()
        
        redis_stats = {
            "redis_commands_sent": stats.get('redis_commands_sent', 0),
            "stream_operations": stats.get('stream_operations', 0),
            "redis_connected": stats.get('redis_connected', False),
            "active_sessions": len(redis_basic_detection_processor.active_sessions) if hasattr(redis_basic_detection_processor, 'active_sessions') else 0
        }
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "redis_stats": redis_stats,
                "general_stats": stats,
                "coordination": "redis",
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting Redis coordination stats: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to get Redis coordination stats: {str(e)}"
        )