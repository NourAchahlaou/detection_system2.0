# basic_video_streaming_router.py - Router with freeze/unfreeze capability

import asyncio
import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query,File, UploadFile
from fastapi.responses import StreamingResponse,Response
from sqlalchemy.orm import Session

# Import existing dependencies (adapt these to your actual imports)
from video_streaming.app.service.alternative.basic_video_streaming_service import BasicStreamConfig, basic_stream_manager
from video_streaming.app.db.session import get_session
from video_streaming.app.service.camera import CameraService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services
camera_service = CameraService()

# Create the router
basic_router = APIRouter(
    prefix="/video/basic",
    tags=["Basic Video Streaming"],
    responses={404: {"description": "Not found"}},
)

@basic_router.get("/stream/{camera_id}")
async def basic_video_stream(
    camera_id: int,
    stream_quality: int = Query(85, description="JPEG quality", ge=50, le=95),
    db: Session = Depends(get_session)
):
    """
    Basic video streaming with freeze capability for detection
    
    This endpoint provides continuous video streaming that can be frozen
    for on-demand detection:
    - Efficient frame processing
    - Stream freeze/unfreeze capability
    - Lightweight resource usage
    """
    try:
        # Validate camera exists
        camera = camera_service.get_camera_by_id(db, camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        # Generate unique consumer ID
        consumer_id = f"basic_consumer_{uuid.uuid4().hex[:8]}_{camera_id}"
        
        logger.info(f"Starting basic stream for camera {camera_id}")
        
        # Create stream config
        config = BasicStreamConfig(
            camera_id=camera_id,
            stream_quality=stream_quality
        )
        
        return StreamingResponse(
            generate_basic_video_frames(config, consumer_id),
            media_type='multipart/x-mixed-replace; boundary=frame',
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Connection": "close",
                "X-Consumer-ID": consumer_id,
                "X-Stream-Type": "basic-streaming-with-freeze",
                "X-Stream-Quality": str(stream_quality)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting basic stream for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start basic stream: {str(e)}")

async def generate_basic_video_frames(config: BasicStreamConfig, consumer_id: str):
    """Generate video frames for basic streaming with freeze capability"""
    frame_count = 0
    stream_key = None
    
    try:
        logger.info(f"🚀 Starting basic frame generation for camera {config.camera_id}, consumer: {consumer_id}")
        
        # Create stream
        stream_key = await basic_stream_manager.create_stream(config)
        stream_state = basic_stream_manager.get_stream(stream_key)
        
        if not stream_state:
            logger.error(f"❌ Failed to create basic stream for camera {config.camera_id}")
            return
        
        # Add consumer
        await stream_state.add_consumer(consumer_id)
        
        # Stream frames
        target_fps = config.target_fps
        frame_time = 1.0 / target_fps
        no_frame_count = 0
        max_no_frame = 100
        
        logger.info(f"🎬 Starting basic frame streaming for camera {config.camera_id}")
        
        while stream_state.is_active and not stream_state.stop_event.is_set():
            frame_start_time = time.time()
            
            try:
                # Get frame (will be frozen frame if stream is frozen)
                frame_bytes = await stream_state.get_latest_frame()
                
                if frame_bytes is not None:
                    no_frame_count = 0
                    
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                    frame_count += 1
                    
                    # Update manager stats
                    basic_stream_manager.stats['total_frames_streamed'] += 1
                    
                    # Log progress
                    if frame_count % 100 == 0:
                        freeze_status = "🧊 FROZEN" if stream_state.is_frozen else "🎬 LIVE"
                        logger.info(f"📊 Basic stream - Camera {config.camera_id}: {frame_count} frames streamed ({freeze_status})")
                else:
                    no_frame_count += 1
                    if no_frame_count > max_no_frame:
                        logger.error(f"❌ No frames received for {max_no_frame} attempts, stopping basic stream for camera {config.camera_id}")
                        break
                    await asyncio.sleep(0.05)
                    continue
                
                # Frame rate control (but don't sleep if frozen to maintain responsiveness)
                if not stream_state.is_frozen:
                    elapsed = time.time() - frame_start_time
                    sleep_time = max(0, frame_time - elapsed)
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                else:
                    # When frozen, update less frequently to save resources
                    await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                logger.info(f"🛑 Basic stream cancelled for camera {config.camera_id}, consumer {consumer_id}")
                break
            except Exception as e:
                logger.error(f"❌ Error in basic frame generation: {e}")
                await asyncio.sleep(0.1)

    except Exception as e:
        logger.error(f"❌ Error in generate_basic_video_frames: {e}")
    finally:
        # Cleanup
        if stream_key and stream_state:
            await stream_state.remove_consumer(consumer_id)
        
        logger.info(f"🏁 Basic video stream stopped for camera {config.camera_id}, consumer {consumer_id} (streamed {frame_count} frames)")

@basic_router.post("/stream/{camera_id}/freeze")
async def freeze_stream(camera_id: int):
    """
    Freeze the video stream for detection
    
    This stops updating frames and holds the current frame
    for detection processing
    """
    try:
        stream_state = basic_stream_manager.get_stream_by_camera_id(camera_id)
        if not stream_state:
            raise HTTPException(status_code=404, detail=f"No active stream found for camera {camera_id}")
        
        success = await stream_state.freeze_stream()
        
        if success:
            return {
                "camera_id": camera_id,
                "status": "frozen",
                "message": f"Stream frozen for camera {camera_id}",
                "timestamp": time.time()
            }
        else:
            raise HTTPException(status_code=400, detail=f"Failed to freeze stream for camera {camera_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error freezing stream for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to freeze stream: {str(e)}")

@basic_router.post("/stream/{camera_id}/unfreeze")
async def unfreeze_stream(camera_id: int):
    """
    Unfreeze the video stream to resume live feed
    
    This resumes normal streaming after detection is complete
    """
    try:
        stream_state = basic_stream_manager.get_stream_by_camera_id(camera_id)
        if not stream_state:
            raise HTTPException(status_code=404, detail=f"No active stream found for camera {camera_id}")
        
        await stream_state.unfreeze_stream()
        
        return {
            "camera_id": camera_id,
            "status": "unfrozen",
            "message": f"Stream resumed for camera {camera_id}",
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unfreezing stream for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unfreeze stream: {str(e)}")

@basic_router.get("/stream/{camera_id}/status")
async def get_stream_status(camera_id: int):
    """Get the current status of a video stream"""
    try:
        stream_state = basic_stream_manager.get_stream_by_camera_id(camera_id)
        if not stream_state:
            return {
                "camera_id": camera_id,
                "stream_active": False,
                "is_frozen": False,
                "consumers_count": 0,
                "message": "No active stream found"
            }
        
        stats = stream_state.get_stats()
        
        return {
            "camera_id": camera_id,
            "stream_active": stats['is_active'],
            "is_frozen": stats['is_frozen'],
            "consumers_count": stats['consumers_count'],
            "frames_processed": stats['frames_processed'],
            "last_frame_time": stats['last_frame_time'],
            "stream_quality": stats['stream_quality'],
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting stream status for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stream status: {str(e)}")

@basic_router.get("/stats")
async def get_basic_streaming_stats():
    """Get basic streaming statistics"""
    try:
        stream_stats = []
        for stream_key, stream_state in basic_stream_manager.active_streams.items():
            try:
                stats = stream_state.get_stats()
                stream_stats.append(stats)
            except Exception as e:
                logger.error(f"Error getting stats for stream {stream_key}: {e}")
        
        total_consumers = sum(stat.get('consumers_count', 0) for stat in stream_stats)
        frozen_streams = sum(1 for stat in stream_stats if stat.get('is_frozen', False))
        
        return {
            "manager_stats": basic_stream_manager.stats,
            "stream_stats": stream_stats,
            "total_active_streams": len(stream_stats),
            "total_consumers": total_consumers,
            "frozen_streams": frozen_streams,
            "live_streams": len(stream_stats) - frozen_streams
        }
        
    except Exception as e:
        logger.error(f"Error getting basic streaming stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@basic_router.post("/stream/{camera_id}/stop")
async def stop_basic_stream(camera_id: int):
    """Stop all basic streams for a specific camera"""
    try:
        matching_streams = [
            key for key, stream in basic_stream_manager.active_streams.items()
            if stream.config.camera_id == camera_id
        ]
        
        if not matching_streams:
            raise HTTPException(status_code=404, detail=f"No active basic stream found for camera {camera_id}")
        
        stopped_count = 0
        for stream_key in matching_streams:
            try:
                await basic_stream_manager.remove_stream(stream_key)
                stopped_count += 1
            except Exception as e:
                logger.error(f"Error stopping basic stream {stream_key}: {e}")
        
        return {
            "camera_id": camera_id,
            "stopped_streams": stopped_count,
            "message": f"Stopped {stopped_count} basic streams for camera {camera_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping basic streams for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop streams: {str(e)}")

@basic_router.get("/health")
async def basic_streaming_health():
    """Health check for basic streaming service"""
    try:
        stats = basic_stream_manager.stats
        
        # Get detailed health info
        active_streams = len(basic_stream_manager.active_streams)
        frozen_count = 0
        total_consumers = 0
        
        for stream_state in basic_stream_manager.active_streams.values():
            if stream_state.is_frozen:
                frozen_count += 1
            total_consumers += len(stream_state.consumers)
        
        return {
            "status": "healthy",
            "active_streams": active_streams,
            "frozen_streams": frozen_count,
            "live_streams": active_streams - frozen_count,
            "total_consumers": total_consumers,
            "total_frames_streamed": stats.get('total_frames_streamed', 0),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Basic streaming health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }
@basic_router.get("/stream/{camera_id}/current-frame")
async def get_current_frame(camera_id: int):
    """
    Get current frame as JPEG for detection service
    
    This endpoint allows the detection service to retrieve
    the current frame from the video stream
    """
    try:
        stream_state = basic_stream_manager.get_stream_by_camera_id(camera_id)
        if not stream_state:
            raise HTTPException(status_code=404, detail=f"No active stream found for camera {camera_id}")
        
        frame_bytes = await stream_state.get_latest_frame()
        if frame_bytes is None:
            raise HTTPException(status_code=404, detail=f"No frame available for camera {camera_id}")
        
        return Response(
            content=frame_bytes,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current frame for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get current frame: {str(e)}")

@basic_router.post("/stream/{camera_id}/update-frozen-frame")
async def update_frozen_frame(camera_id: int, frame: UploadFile = File(...)):
    """
    Update the frozen frame with detection results
    
    This allows the detection service to update the frozen frame
    with overlay annotations after processing
    """
    try:
        stream_state = basic_stream_manager.get_stream_by_camera_id(camera_id)
        if not stream_state:
            raise HTTPException(status_code=404, detail=f"No active stream found for camera {camera_id}")
        
        if not stream_state.is_frozen:
            raise HTTPException(status_code=400, detail=f"Stream for camera {camera_id} is not frozen")
        
        # Read the uploaded frame
        frame_bytes = await frame.read()
        
        # Update the frozen frame
        await stream_state.update_frozen_frame(frame_bytes)
        
        return {
            "camera_id": camera_id,
            "message": "Frozen frame updated successfully",
            "frame_size": len(frame_bytes),
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating frozen frame for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update frozen frame: {str(e)}")    