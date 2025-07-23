from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import time
import logging
import asyncio
import uuid

from video_streaming.app.db.session import get_session
from video_streaming.app.service.video_streaming_service import (
    generate_video_frames_optimized, 
    stop_camera_via_hardware_service,
    stream_manager,
    active_stream_states
)

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

video_router = APIRouter(
    prefix="/video",
    tags=["Video Streaming"],
    responses={404: {"description": "Not found"}},
)

@video_router.get("/stream/{camera_id}")
async def video_stream_optimized(camera_id: int, db: Session = Depends(get_session)):
    """
    OPTIMIZED video streaming endpoint that eliminates multiple GET requests.
    Uses a single persistent connection with frame buffering for multiple consumers.
    """
    logger.info(f"Starting optimized video stream for camera {camera_id}")
    
    # Generate unique consumer ID
    consumer_id = f"web_consumer_{uuid.uuid4().hex[:8]}_{camera_id}"
    
    return StreamingResponse(
        generate_video_frames_optimized(camera_id, db, consumer_id), 
        media_type='multipart/x-mixed-replace; boundary=frame',
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close",
            "X-Consumer-ID": consumer_id  # For debugging
        }
    )

@video_router.post("/stop/{camera_id}")
async def stop_video_stream_optimized(camera_id: int):
    """
    OPTIMIZED stop endpoint that properly handles shared streams.
    Only stops the camera when no consumers are left.
    """
    try:
        logger.info(f"Received stop request for camera {camera_id}")
        
        if camera_id not in active_stream_states:
            logger.info(f"Camera {camera_id} is not currently streaming")
            return {"message": f"Camera {camera_id} is not currently streaming"}
        
        stream_state = active_stream_states[camera_id]
        consumer_count_before = len(stream_state.consumers)
        
        # Force cleanup of the stream state
        await stream_state.cleanup()
        
        # Remove from global tracking
        if camera_id in active_stream_states:
            del active_stream_states[camera_id]
        
        # Use stream manager to clean up
        await stream_manager.stop_camera_stream(camera_id)
        
        # Stop hardware service in background
        asyncio.create_task(stop_camera_via_hardware_service())
        
        logger.info(f"Optimized video stream stopped for camera {camera_id}")
        return {
            "message": f"Optimized video stream stopped for camera {camera_id}",
            "consumers_before_stop": consumer_count_before,
            "stopped_immediately": True,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error stopping optimized video stream for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@video_router.post("/force-stop/{camera_id}")
async def force_stop_video_stream_optimized(camera_id: int):
    """
    OPTIMIZED force stop that properly cleans up shared resources.
    """
    try:
        logger.info(f"Force stopping optimized camera {camera_id}")
        
        # Force cleanup of stream state
        if camera_id in active_stream_states:
            stream_state = active_stream_states[camera_id]
            consumers_before = len(stream_state.consumers)
            
            await stream_state.cleanup()
            del active_stream_states[camera_id]
            
            logger.info(f"Force cleaned up {consumers_before} consumers for camera {camera_id}")
        
        # Use stream manager's async method
        await stream_manager.stop_camera_stream(camera_id)
        
        # Stop hardware service
        try:
            await stop_camera_via_hardware_service()
        except Exception as e:
            logger.warning(f"Error stopping hardware service during force stop: {e}")
        
        logger.info(f"Force stopped optimized camera {camera_id}")
        return {
            "message": f"Camera {camera_id} force stopped (optimized)",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error force stopping optimized camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Force stop failed: {e}")

@video_router.get("/status/{camera_id}")
async def get_optimized_stream_status(camera_id: int):
    """Get detailed streaming status for optimized streams."""
    try:
        stream_info = stream_manager.get_stream_info(camera_id)
        
        # Add optimized stream info
        if camera_id in active_stream_states:
            stream_state = active_stream_states[camera_id]
            stream_info.update({
                "frame_count": getattr(stream_state, 'frame_count', 0),
                "last_frame_time": getattr(stream_state, 'last_frame_time', 0),
                "consecutive_failures": getattr(stream_state, 'consecutive_failures', 0),
                "stream_active": getattr(stream_state, 'is_active', False),
                "consumer_count": len(getattr(stream_state, 'consumers', [])),
                "consumers": list(getattr(stream_state, 'consumers', [])),
                "has_connection_task": getattr(stream_state, 'connection_task', None) is not None,
                "connection_task_done": getattr(stream_state, 'connection_task', None) and stream_state.connection_task.done(),
                "buffer_size": len(getattr(stream_state, 'frame_buffer', [])),
                "optimization_enabled": True
            })
    except Exception as e:
        logger.error(f"Error getting optimized stream info for camera {camera_id}: {e}")
        stream_info = {"is_active": False, "error": str(e), "optimization_enabled": True}
    
    return {
        **stream_info,
        "active_streams": list(active_stream_states.keys()),
        "timestamp": time.time()
    }

@video_router.get("/status")
async def get_all_optimized_streams_status():
    """Get status of all optimized video streams with detailed information."""
    active_cameras = {}
    total_consumers = 0
    
    for camera_id in list(active_stream_states.keys()):
        try:
            stream_info = stream_manager.get_stream_info(camera_id)
            
            # Add optimized stream info
            if camera_id in active_stream_states:
                stream_state = active_stream_states[camera_id]
                consumer_count = len(getattr(stream_state, 'consumers', []))
                total_consumers += consumer_count
                
                stream_info.update({
                    "frame_count": getattr(stream_state, 'frame_count', 0),
                    "last_frame_time": getattr(stream_state, 'last_frame_time', 0),
                    "consecutive_failures": getattr(stream_state, 'consecutive_failures', 0),
                    "stream_active": getattr(stream_state, 'is_active', False),
                    "consumer_count": consumer_count,
                    "consumers": list(getattr(stream_state, 'consumers', [])),
                    "has_connection_task": getattr(stream_state, 'connection_task', None) is not None,
                    "connection_task_done": getattr(stream_state, 'connection_task', None) and stream_state.connection_task.done(),
                    "buffer_size": len(getattr(stream_state, 'frame_buffer', [])),
                    "optimization_enabled": True
                })
            
            active_cameras[camera_id] = stream_info
        except Exception as e:
            logger.error(f"Error getting optimized stream info for camera {camera_id}: {e}")
            active_cameras[camera_id] = {"is_active": False, "error": str(e), "optimization_enabled": True}
    
    return {
        "active_streams": list(active_stream_states.keys()),
        "total_active": len(active_stream_states),
        "total_consumers": total_consumers,
        "detailed_status": active_cameras,
        "optimization_enabled": True,
        "timestamp": time.time()
    }

@video_router.post("/cleanup")
async def cleanup_all_optimized_streams():
    """
    OPTIMIZED emergency cleanup that properly handles shared resources.
    """
    try:
        logger.info("Performing optimized emergency cleanup of all streams")
        
        cleanup_count = 0
        total_consumers = 0
        
        # Clean up optimized stream states
        for camera_id in list(active_stream_states.keys()):
            try:
                stream_state = active_stream_states[camera_id]
                consumers_count = len(stream_state.consumers)
                total_consumers += consumers_count
                
                await stream_state.cleanup()
                cleanup_count += 1
                
                logger.info(f"Cleaned up camera {camera_id} with {consumers_count} consumers")
                        
            except Exception as e:
                logger.error(f"Error cleaning up optimized stream state for camera {camera_id}: {e}")
        
        # Clear the dict
        active_stream_states.clear()
        
        # Use stream manager's cleanup
        manager_cleanup_count = await stream_manager.cleanup_all_streams()
        
        # Stop hardware service
        try:
            await stop_camera_via_hardware_service()
        except Exception as e:
            logger.warning(f"Error stopping hardware service during cleanup: {e}")
        
        logger.info(f"Optimized emergency cleanup completed. Cleaned up {cleanup_count} streams with {total_consumers} total consumers")
        return {
            "message": "All optimized streams cleaned up",
            "streams_cleaned": cleanup_count,
            "consumers_cleaned": total_consumers,
            "manager_streams_cleaned": manager_cleanup_count,
            "optimization_enabled": True,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error during optimized cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Optimized cleanup failed: {e}")

@video_router.get("/health")
async def video_health_check_optimized():
    """Enhanced health check for optimized video streaming service."""
    try:
        from video_streaming.app.service.video_streaming_service import check_camera_status_safe
        
        hardware_status = await check_camera_status_safe()
        hardware_accessible = hardware_status is not None
        
        active_count = len(active_stream_states)
        healthy_streams = 0
        total_consumers = 0
        active_producers = 0
        
        # Check health of optimized streams
        for camera_id, stream_state in active_stream_states.items():
            consumer_count = len(getattr(stream_state, 'consumers', []))
            total_consumers += consumer_count
            
            # Check if producer task is running
            if (hasattr(stream_state, 'connection_task') and 
                stream_state.connection_task and not stream_state.connection_task.done()):
                active_producers += 1
            
            if (hasattr(stream_state, 'is_active') and stream_state.is_active and 
                hasattr(stream_state, 'last_frame_time') and 
                (time.time() - stream_state.last_frame_time) < 5.0):
                healthy_streams += 1
        
        overall_health = "healthy"
        if not hardware_accessible:
            overall_health = "degraded"
        elif active_count > 0 and healthy_streams == 0:
            overall_health = "unhealthy"
        
        return {
            "status": overall_health,
            "service": "video_streaming_optimized",
            "hardware_service": "accessible" if hardware_accessible else "unavailable",
            "active_streams": active_count,
            "healthy_streams": healthy_streams,
            "total_consumers": total_consumers,
            "active_producers": active_producers,
            "optimization_enabled": True,
            "hardware_status": hardware_status,
            "performance_benefit": f"Reduced from {total_consumers} connections to {active_producers} connections",
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "video_streaming_optimized",
            "error": str(e),
            "optimization_enabled": True,
            "timestamp": time.time()
        }
