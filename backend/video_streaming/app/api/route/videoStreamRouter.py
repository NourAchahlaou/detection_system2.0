from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import time
import logging
import asyncio

from video_streaming.app.db.session import get_session
from video_streaming.app.service.video_streaming_service import (
    generate_video_frames, 
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
async def video_stream(camera_id: int, db: Session = Depends(get_session)):
    """
    High-performance video streaming endpoint with improved stop handling.
    """
    logger.info(f"Starting video stream for camera {camera_id}")
    
    # Check if camera is already streaming
    if stream_manager.is_camera_active(camera_id):
        raise HTTPException(status_code=409, detail="Camera is already streaming")
    
    return StreamingResponse(
        generate_video_frames(camera_id, db), 
        media_type='multipart/x-mixed-replace; boundary=frame',
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close"  # Ensure connection closes properly
        }
    )

@video_router.post("/stop/{camera_id}")
async def stop_video_stream(camera_id: int):
    """
    Stop video streaming for a specific camera with immediate response.
    This addresses the delay issue you're experiencing.
    """
    try:
        logger.info(f"Received stop request for camera {camera_id}")
        
        if not stream_manager.is_camera_active(camera_id):
            logger.info(f"Camera {camera_id} is not currently streaming")
            return {"message": f"Camera {camera_id} is not currently streaming"}
        
        # Use the async stop_camera_stream method from VideoStreamManager
        stop_success = await stream_manager.stop_camera_stream(camera_id)
        
        if stop_success:
            # Also clean up from active_stream_states
            if camera_id in active_stream_states:
                stream_state = active_stream_states[camera_id]
                stream_state.is_active = False
                stream_state.stop_event.set()
                
                # Close session if available
                if hasattr(stream_state, 'session') and stream_state.session and not stream_state.session.closed:
                    try:
                        await stream_state.session.close()
                    except Exception as session_error:
                        logger.warning(f"Error closing session: {session_error}")
                
                del active_stream_states[camera_id]
            
            # Stop the hardware service (this can be done in parallel)
            # Create a task to avoid blocking the response
            asyncio.create_task(stop_camera_via_hardware_service())
            
            logger.info(f"Video stream stop initiated for camera {camera_id}")
            return {
                "message": f"Video stream stopped for camera {camera_id}",
                "stopped_immediately": True,
                "timestamp": time.time()
            }
        else:
            return {
                "message": f"Failed to stop camera {camera_id}",
                "stopped_immediately": False
            }
        
    except Exception as e:
        logger.error(f"Error stopping video stream for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An error occurred: {e}")

@video_router.post("/force-stop/{camera_id}")
async def force_stop_video_stream(camera_id: int):
    """
    Force stop a video stream - useful for stuck streams.
    """
    try:
        logger.info(f"Force stopping camera {camera_id}")
        
        # Remove from all tracking immediately
        if camera_id in active_stream_states:
            stream_state = active_stream_states[camera_id]
            stream_state.is_active = False
            stream_state.stop_event.set()
            
            # Close session if available
            if hasattr(stream_state, 'session') and stream_state.session and not stream_state.session.closed:
                try:
                    await stream_state.session.close()
                except Exception as session_error:
                    logger.warning(f"Error closing session during force stop: {session_error}")
            
            del active_stream_states[camera_id]
        
        # Use stream manager's async method
        await stream_manager.stop_camera_stream(camera_id)
        
        # Stop hardware service
        try:
            await stop_camera_via_hardware_service()
        except Exception as e:
            logger.warning(f"Error stopping hardware service during force stop: {e}")
        
        logger.info(f"Force stopped camera {camera_id}")
        return {
            "message": f"Camera {camera_id} force stopped",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error force stopping camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Force stop failed: {e}")

@video_router.get("/status/{camera_id}")
async def get_stream_status(camera_id: int):
    """Get detailed streaming status for a specific camera."""
    try:
        stream_info = stream_manager.get_stream_info(camera_id)
        
        # Add additional info from active_stream_states if available
        if camera_id in active_stream_states:
            stream_state = active_stream_states[camera_id]
            stream_info.update({
                "frame_count": getattr(stream_state, 'frame_count', 0),
                "last_frame_time": getattr(stream_state, 'last_frame_time', 0),
                "consecutive_failures": getattr(stream_state, 'consecutive_failures', 0),
                "stream_active": getattr(stream_state, 'is_active', False)
            })
    except Exception as e:
        logger.error(f"Error getting stream info for camera {camera_id}: {e}")
        stream_info = {"is_active": False, "error": str(e)}
    
    return {
        **stream_info,
        "active_streams": list(active_stream_states.keys()),
        "timestamp": time.time()
    }

@video_router.get("/status")
async def get_all_streams_status():
    """Get status of all active video streams with detailed information."""
    active_cameras = {}
    
    for camera_id in list(active_stream_states.keys()):
        try:
            stream_info = stream_manager.get_stream_info(camera_id)
            
            # Add additional info from active_stream_states
            if camera_id in active_stream_states:
                stream_state = active_stream_states[camera_id]
                stream_info.update({
                    "frame_count": getattr(stream_state, 'frame_count', 0),
                    "last_frame_time": getattr(stream_state, 'last_frame_time', 0),
                    "consecutive_failures": getattr(stream_state, 'consecutive_failures', 0),
                    "stream_active": getattr(stream_state, 'is_active', False)
                })
            
            active_cameras[camera_id] = stream_info
        except Exception as e:
            logger.error(f"Error getting stream info for camera {camera_id}: {e}")
            active_cameras[camera_id] = {"is_active": False, "error": str(e)}
    
    return {
        "active_streams": list(active_stream_states.keys()),
        "total_active": len(active_stream_states),
        "detailed_status": active_cameras,
        "timestamp": time.time()
    }

@video_router.post("/cleanup")
async def cleanup_all_streams():
    """
    Emergency cleanup of all streams - useful for debugging.
    """
    try:
        logger.info("Performing emergency cleanup of all streams")
        
        # Use stream manager's async cleanup method
        cleanup_count = await stream_manager.cleanup_all_streams()
        
        # Also clear active_stream_states
        for camera_id in list(active_stream_states.keys()):
            try:
                stream_state = active_stream_states[camera_id]
                stream_state.is_active = False
                stream_state.stop_event.set()
                
                # Close session if available
                if hasattr(stream_state, 'session') and stream_state.session and not stream_state.session.closed:
                    try:
                        await stream_state.session.close()
                    except Exception as session_error:
                        logger.warning(f"Error closing session during cleanup: {session_error}")
                        
            except Exception as e:
                logger.error(f"Error cleaning up stream state for camera {camera_id}: {e}")
        
        # Clear the dict
        active_stream_states.clear()
        
        # Stop hardware service
        try:
            await stop_camera_via_hardware_service()
        except Exception as e:
            logger.warning(f"Error stopping hardware service during cleanup: {e}")
        
        logger.info(f"Emergency cleanup completed. Cleaned up {cleanup_count} streams")
        return {
            "message": "All streams cleaned up",
            "streams_cleaned": cleanup_count,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error during cleanup: {e}")
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {e}")

@video_router.get("/health")
async def video_health_check():
    """Enhanced health check for video streaming service."""
    try:
        from video_streaming.app.service.video_streaming_service import check_camera_status_safe
        
        # check_camera_status_safe is already async
        hardware_status = await check_camera_status_safe()
        hardware_accessible = hardware_status is not None
        
        active_count = len(active_stream_states)
        healthy_streams = 0
        
        # Check health of active streams
        for camera_id, stream_state in active_stream_states.items():
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
            "service": "video_streaming",
            "hardware_service": "accessible" if hardware_accessible else "unavailable",
            "active_streams": active_count,
            "healthy_streams": healthy_streams,
            "hardware_status": hardware_status,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "service": "video_streaming",
            "error": str(e),
            "timestamp": time.time()
        }