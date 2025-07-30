# redis_basic_video_streaming_router.py - Router with Redis coordination and freeze/unfreeze capability

import asyncio
import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session

# Import Redis-based services (update these imports to match your actual structure)
from video_streaming.app.service.alternative.basic_video_streaming_service import (
    BasicStreamConfig, 
    redis_basic_stream_manager
)
from video_streaming.app.db.session import get_session
from video_streaming.app.service.camera import CameraService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services
camera_service = CameraService()

# Create the router
redis_basic_router = APIRouter(
    prefix="/video/redis-basic",
    tags=["Redis Basic Video Streaming"],
    responses={404: {"description": "Not found"}},
)

@redis_basic_router.get("/stream/{camera_id}")
async def redis_basic_video_stream(
    camera_id: int,
    stream_quality: int = Query(85, description="JPEG quality", ge=50, le=95),
    db: Session = Depends(get_session)
):
    """
    Redis-coordinated basic video streaming with freeze capability for detection
    
    This endpoint provides continuous video streaming that can be frozen
    for on-demand detection via Redis coordination:
    - Efficient frame processing with Redis coordination
    - Stream freeze/unfreeze capability via Redis commands
    - Cross-container communication support
    - Lightweight resource usage
    """
    try:
        # Validate camera exists
        camera = camera_service.get_camera_by_id(db, camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        # Generate unique consumer ID
        consumer_id = f"redis_basic_consumer_{uuid.uuid4().hex[:8]}_{camera_id}"
        
        logger.info(f"Starting Redis basic stream for camera {camera_id}")
        
        # Create stream config
        config = BasicStreamConfig(
            camera_id=camera_id,
            stream_quality=stream_quality
        )
        
        return StreamingResponse(
            generate_redis_basic_video_frames(config, consumer_id),
            media_type='multipart/x-mixed-replace; boundary=frame',
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Connection": "close",
                "X-Consumer-ID": consumer_id,
                "X-Stream-Type": "redis-basic-streaming-with-freeze",
                "X-Stream-Quality": str(stream_quality),
                "X-Redis-Coordinated": "true"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting Redis basic stream for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start Redis basic stream: {str(e)}")

async def generate_redis_basic_video_frames(config: BasicStreamConfig, consumer_id: str):
    """Generate video frames for Redis-coordinated basic streaming with freeze capability"""
    frame_count = 0
    stream_key = None
    
    try:
        logger.info(f"🚀 Starting Redis basic frame generation for camera {config.camera_id}, consumer: {consumer_id}")
        
        # Initialize Redis stream manager if not already done
        if not hasattr(redis_basic_stream_manager, 'redis_coordinator') or not redis_basic_stream_manager.redis_coordinator.is_running:
            await redis_basic_stream_manager.initialize()
        
        # Create stream
        stream_key = await redis_basic_stream_manager.create_stream(config)
        stream_state = redis_basic_stream_manager.get_stream(stream_key)
        
        if not stream_state:
            logger.error(f"❌ Failed to create Redis basic stream for camera {config.camera_id}")
            return
        
        # Add consumer
        await stream_state.add_consumer(consumer_id)
        
        # Stream frames
        target_fps = config.target_fps
        frame_time = 1.0 / target_fps
        no_frame_count = 0
        max_no_frame = 100
        
        logger.info(f"🎬 Starting Redis basic frame streaming for camera {config.camera_id}")
        
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
                    redis_basic_stream_manager.stats['total_frames_streamed'] += 1
                    
                    # Log progress with Redis coordination status
                    if frame_count % 100 == 0:
                        freeze_status = "🧊 FROZEN" if stream_state.is_frozen else "🎬 LIVE"
                        redis_status = "📡 REDIS-COORDINATED"
                        logger.info(f"📊 Redis basic stream - Camera {config.camera_id}: {frame_count} frames streamed ({freeze_status}) ({redis_status})")
                else:
                    no_frame_count += 1
                    if no_frame_count > max_no_frame:
                        logger.error(f"❌ No frames received for {max_no_frame} attempts, stopping Redis basic stream for camera {config.camera_id}")
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
                logger.info(f"🛑 Redis basic stream cancelled for camera {config.camera_id}, consumer {consumer_id}")
                break
            except Exception as e:
                logger.error(f"❌ Error in Redis basic frame generation: {e}")
                await asyncio.sleep(0.1)

    except Exception as e:
        logger.error(f"❌ Error in generate_redis_basic_video_frames: {e}")
    finally:
        # Cleanup
        if stream_key and stream_state:
            await stream_state.remove_consumer(consumer_id)
        
        logger.info(f"🏁 Redis basic video stream stopped for camera {config.camera_id}, consumer {consumer_id} (streamed {frame_count} frames)")

@redis_basic_router.post("/stream/{camera_id}/freeze")
async def freeze_redis_stream(camera_id: int):
    """
    Freeze the Redis-coordinated video stream for detection
    
    This stops updating frames and holds the current frame
    for detection processing via Redis coordination
    """
    try:
        stream_state = redis_basic_stream_manager.get_stream_by_camera_id(camera_id)
        if not stream_state:
            raise HTTPException(status_code=404, detail=f"No active Redis stream found for camera {camera_id}")
        
        success = await stream_state.freeze_stream()
        
        if success:
            return {
                "camera_id": camera_id,
                "status": "frozen",
                "message": f"Redis stream frozen for camera {camera_id}",
                "coordination": "redis",
                "timestamp": time.time()
            }
        else:
            raise HTTPException(status_code=400, detail=f"Failed to freeze Redis stream for camera {camera_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error freezing Redis stream for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to freeze Redis stream: {str(e)}")

@redis_basic_router.post("/stream/{camera_id}/unfreeze")
async def unfreeze_redis_stream(camera_id: int):
    """
    Unfreeze the Redis-coordinated video stream to resume live feed
    
    This resumes normal streaming after detection is complete
    """
    try:
        stream_state = redis_basic_stream_manager.get_stream_by_camera_id(camera_id)
        if not stream_state:
            raise HTTPException(status_code=404, detail=f"No active Redis stream found for camera {camera_id}")
        
        await stream_state.unfreeze_stream()
        
        return {
            "camera_id": camera_id,
            "status": "unfrozen",
            "message": f"Redis stream resumed for camera {camera_id}",
            "coordination": "redis",
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unfreezing Redis stream for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unfreeze Redis stream: {str(e)}")

@redis_basic_router.get("/stream/{camera_id}/status")
async def get_redis_stream_status(camera_id: int):
    """Get the current status of a Redis-coordinated video stream"""
    try:
        stream_state = redis_basic_stream_manager.get_stream_by_camera_id(camera_id)
        if not stream_state:
            return {
                "camera_id": camera_id,
                "stream_active": False,
                "is_frozen": False,
                "consumers_count": 0,
                "coordination": "redis",
                "message": "No active Redis stream found"
            }
        
        stats = stream_state.get_stats()
        
        # Add Redis coordination info
        redis_connected = False
        try:
            if redis_basic_stream_manager.redis_coordinator.sync_redis_client:
                redis_basic_stream_manager.redis_coordinator.sync_redis_client.ping()
                redis_connected = True
        except:
            redis_connected = False
        
        return {
            "camera_id": camera_id,
            "stream_active": stats['is_active'],
            "is_frozen": stats['is_frozen'],
            "consumers_count": stats['consumers_count'],
            "frames_processed": stats['frames_processed'],
            "last_frame_time": stats['last_frame_time'],
            "stream_quality": stats['stream_quality'],
            "coordination": "redis",
            "redis_connected": redis_connected,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting Redis stream status for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Redis stream status: {str(e)}")

@redis_basic_router.get("/stats")
async def get_redis_streaming_stats():
    """Get Redis-coordinated basic streaming statistics"""
    try:
        stream_stats = []
        for stream_key, stream_state in redis_basic_stream_manager.active_streams.items():
            try:
                stats = stream_state.get_stats()
                stats['stream_key'] = stream_key
                stream_stats.append(stats)
            except Exception as e:
                logger.error(f"Error getting stats for Redis stream {stream_key}: {e}")
        
        total_consumers = sum(stat.get('consumers_count', 0) for stat in stream_stats)
        frozen_streams = sum(1 for stat in stream_stats if stat.get('is_frozen', False))
        
        # Check Redis coordinator status
        redis_coordinator_status = {
            "is_running": redis_basic_stream_manager.redis_coordinator.is_running if redis_basic_stream_manager.redis_coordinator else False,
            "connected": False
        }
        
        try:
            if redis_basic_stream_manager.redis_coordinator and redis_basic_stream_manager.redis_coordinator.sync_redis_client:
                redis_basic_stream_manager.redis_coordinator.sync_redis_client.ping()
                redis_coordinator_status["connected"] = True
        except:
            pass
        
        return {
            "manager_stats": redis_basic_stream_manager.stats,
            "stream_stats": stream_stats,
            "total_active_streams": len(stream_stats),
            "total_consumers": total_consumers,
            "frozen_streams": frozen_streams,
            "live_streams": len(stream_stats) - frozen_streams,
            "redis_coordinator": redis_coordinator_status,
            "coordination_type": "redis"
        }
        
    except Exception as e:
        logger.error(f"Error getting Redis streaming stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get Redis stats: {str(e)}")

@redis_basic_router.post("/stream/{camera_id}/stop")
async def stop_redis_basic_stream(camera_id: int):
    """Stop all Redis-coordinated basic streams for a specific camera"""
    try:
        matching_streams = [
            key for key, stream in redis_basic_stream_manager.active_streams.items()
            if stream.config.camera_id == camera_id
        ]
        
        if not matching_streams:
            raise HTTPException(status_code=404, detail=f"No active Redis basic stream found for camera {camera_id}")
        
        stopped_count = 0
        for stream_key in matching_streams:
            try:
                await redis_basic_stream_manager.remove_stream(stream_key)
                stopped_count += 1
            except Exception as e:
                logger.error(f"Error stopping Redis basic stream {stream_key}: {e}")
        
        return {
            "camera_id": camera_id,
            "stopped_streams": stopped_count,
            "coordination": "redis",
            "message": f"Stopped {stopped_count} Redis basic streams for camera {camera_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping Redis basic streams for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop Redis streams: {str(e)}")

@redis_basic_router.get("/health")
async def redis_streaming_health():
    """Health check for Redis-coordinated basic streaming service"""
    try:
        stats = redis_basic_stream_manager.stats
        
        # Get detailed health info
        active_streams = len(redis_basic_stream_manager.active_streams)
        frozen_count = 0
        total_consumers = 0
        
        for stream_state in redis_basic_stream_manager.active_streams.values():
            if stream_state.is_frozen:
                frozen_count += 1
            total_consumers += len(stream_state.consumers)
        
        # Check Redis coordinator health
        redis_healthy = False
        redis_error = None
        
        try:
            if redis_basic_stream_manager.redis_coordinator:
                if redis_basic_stream_manager.redis_coordinator.sync_redis_client:
                    redis_basic_stream_manager.redis_coordinator.sync_redis_client.ping()
                    redis_healthy = True
                else:
                    redis_error = "Redis client not initialized"
            else:
                redis_error = "Redis coordinator not initialized"
        except Exception as e:
            redis_error = str(e)
        
        overall_status = "healthy" if redis_healthy else "degraded"
        
        response = {
            "status": overall_status,
            "active_streams": active_streams,
            "frozen_streams": frozen_count,
            "live_streams": active_streams - frozen_count,
            "total_consumers": total_consumers,
            "total_frames_streamed": stats.get('total_frames_streamed', 0),
            "coordination": "redis",
            "redis_coordinator": {
                "healthy": redis_healthy,
                "is_running": redis_basic_stream_manager.redis_coordinator.is_running if redis_basic_stream_manager.redis_coordinator else False,
                "error": redis_error
            },
            "timestamp": time.time()
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Redis streaming health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "coordination": "redis",
            "timestamp": time.time()
        }

@redis_basic_router.post("/initialize")
async def initialize_redis_streaming():
    """Initialize the Redis-coordinated streaming system"""
    try:
        logger.info("🚀 Initializing Redis basic streaming system...")
        
        await redis_basic_stream_manager.initialize()
        
        return {
            "success": True,
            "message": "Redis basic streaming system initialized successfully",
            "coordination": "redis",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Error initializing Redis streaming system: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to initialize Redis streaming: {str(e)}"
        )

@redis_basic_router.post("/cleanup")
async def cleanup_redis_streaming():
    """Cleanup Redis-coordinated streaming resources"""
    try:
        logger.info("🧹 Cleaning up Redis basic streaming system...")
        
        await redis_basic_stream_manager.cleanup()
        
        return {
            "success": True,
            "message": "Redis basic streaming system cleaned up successfully",
            "coordination": "redis",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"❌ Error cleaning up Redis streaming system: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to cleanup Redis streaming: {str(e)}"
        )