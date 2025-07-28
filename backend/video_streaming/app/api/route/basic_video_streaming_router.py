# basic_video_streaming_service.py - Simple streaming for low-performance mode

import asyncio
import logging


import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

# Import existing dependencies (adapt these to your actual imports)
from video_streaming.app.service.alternative.basic_video_streaming_service import BasicStreamConfig,basic_stream_manager
from video_streaming.app.db.session import get_session
from video_streaming.app.service.camera import CameraService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration



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
    Basic video streaming without real-time detection
    
    This endpoint provides simple video streaming for low-performance systems:
    - Efficient frame processing
    - No detection overhead
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
                "X-Stream-Type": "basic-streaming-only",
                "X-Stream-Quality": str(stream_quality)
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting basic stream for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start basic stream: {str(e)}")

async def generate_basic_video_frames(config: BasicStreamConfig, consumer_id: str):
    """Generate video frames for basic streaming"""
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
                        logger.info(f"📊 Basic stream - Camera {config.camera_id}: {frame_count} frames streamed")
                else:
                    no_frame_count += 1
                    if no_frame_count > max_no_frame:
                        logger.error(f"❌ No frames received for {max_no_frame} attempts, stopping basic stream for camera {config.camera_id}")
                        break
                    await asyncio.sleep(0.05)
                    continue
                
                # Frame rate control
                elapsed = time.time() - frame_start_time
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
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
        
        return {
            "manager_stats": basic_stream_manager.stats,
            "stream_stats": stream_stats,
            "total_active_streams": len(stream_stats),
            "total_consumers": total_consumers
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
        
        return {
            "status": "healthy",
            "active_streams": stats.get('active_streams_count', 0),
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