# hybrid_video_streaming_router.py - Router with adaptive streaming for both camera types

import asyncio
import logging
import time
from typing import Annotated, Dict
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, File, UploadFile
from fastapi.responses import StreamingResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

# Import the hybrid streaming service
from video_streaming.app.service.alternative.basic_video_streaming_service import (
    HybridStreamConfig, hybrid_stream_manager, CameraType
)
from video_streaming.app.db.session import get_session
from video_streaming.app.service.camera import CameraService

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize services
camera_service = CameraService()

class CameraIDRequest(BaseModel):
    camera_id: int = Field(..., description="The database ID of the camera")

class CameraTypeRequest(BaseModel):
    camera_id: int = Field(..., description="The database ID of the camera")
    camera_type: str = Field("auto", description="Camera type: 'regular', 'basler', or 'auto'")

# Create the router
hybrid_router = APIRouter(
    prefix="/video/basic",
    tags=["Hybrid Video Streaming"],
    responses={404: {"description": "Not found"}},
)

# Database dependency
db_dependency = Annotated[Session, Depends(get_session)]

@hybrid_router.post("/start", response_model=Dict[str, str])
async def start_camera(request: CameraIDRequest, db: db_dependency):
    """Start a camera by its ID using hybrid streaming approach"""
    return camera_service.start_camera(db, request.camera_id)

@hybrid_router.get("/stream/{camera_id}")
async def hybrid_video_stream(
    camera_id: int,
    stream_quality: int = Query(85, description="JPEG quality", ge=50, le=95),
    camera_type: str = Query("auto", description="Camera type: 'regular', 'basler', or 'auto'"),
    target_fps: int = Query(25, description="Target FPS for regular cameras", ge=10, le=60),
    polling_fps: int = Query(15, description="Polling FPS for Basler cameras", ge=5, le=30),
    db: Session = Depends(get_session)
):
    """
    Hybrid video streaming that adapts to camera type
    
    This endpoint automatically selects the best streaming approach:
    - Regular cameras: High-throughput MJPEG streaming
    - Basler cameras: Reliable polling-based frame capture
    - Auto-detection: Automatically determines camera type
    
    Features:
    - Automatic camera type detection
    - Optimized streaming method per camera type
    - Stream freeze/unfreeze capability
    - Efficient resource usage
    """
    try:
        # Validate camera exists
        camera = camera_service.get_camera_by_id(db, camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")
        
        # Generate unique consumer ID
        consumer_id = f"hybrid_consumer_{uuid.uuid4().hex[:8]}_{camera_id}"
        
        # Determine camera type
        if camera_type == "auto":
            detected_type = CameraType.UNKNOWN  # Will be auto-detected
        elif camera_type == "basler":
            detected_type = CameraType.BASLER
        elif camera_type == "regular":
            detected_type = CameraType.REGULAR
        else:
            detected_type = CameraType.UNKNOWN
        
        logger.info(f"Starting hybrid stream for camera {camera_id} (type: {camera_type})")
        
        # Create stream config
        config = HybridStreamConfig(
            camera_id=camera_id,
            stream_quality=stream_quality,
            target_fps=target_fps,
            polling_fps=polling_fps,
            camera_type=detected_type
        )
        
        return StreamingResponse(
            generate_hybrid_video_frames(config, consumer_id),
            media_type='multipart/x-mixed-replace; boundary=frame',
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "Connection": "close",
                "X-Consumer-ID": consumer_id,
                "X-Stream-Type": "hybrid-adaptive-streaming",
                "X-Stream-Quality": str(stream_quality),
                "X-Camera-Type": camera_type
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting hybrid stream for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start hybrid stream: {str(e)}")

async def generate_hybrid_video_frames(config: HybridStreamConfig, consumer_id: str):
    """Generate video frames using adaptive streaming approach"""
    frame_count = 0
    stream_key = None
    
    try:
        logger.info(f"🚀 Starting hybrid frame generation for camera {config.camera_id}, consumer: {consumer_id}")
        
        # Create stream (will auto-detect camera type if needed)
        stream_key = await hybrid_stream_manager.create_stream(config)
        stream_state = hybrid_stream_manager.get_stream(stream_key)
        
        if not stream_state:
            logger.error(f"❌ Failed to create hybrid stream for camera {config.camera_id}")
            return
        
        # Add consumer
        await stream_state.add_consumer(consumer_id)
        
        # Get the detected camera type for adaptive frame timing
        detected_type = stream_state.config.camera_type
        if detected_type == CameraType.BASLER:
            frame_time = 1.0 / config.polling_fps
            logger.info(f"🎬 Using Basler-optimized streaming (polling at {config.polling_fps} FPS)")
        else:
            frame_time = 1.0 / config.target_fps
            logger.info(f"🎬 Using regular camera streaming (target {config.target_fps} FPS)")
        
        # Stream frames with adaptive timing
        no_frame_count = 0
        max_no_frame = 150 if detected_type == CameraType.BASLER else 100
        
        logger.info(f"🎬 Starting hybrid frame streaming for {detected_type.value} camera {config.camera_id}")
        
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
                    hybrid_stream_manager.stats['total_frames_streamed'] += 1
                    
                    # Log progress with camera type info
                    if frame_count % 100 == 0:
                        freeze_status = "🧊 FROZEN" if stream_state.is_frozen else "🎬 LIVE"
                        logger.info(f"📊 Hybrid stream - {detected_type.value} camera {config.camera_id}: {frame_count} frames streamed ({freeze_status})")
                else:
                    no_frame_count += 1
                    if no_frame_count > max_no_frame:
                        logger.error(f"❌ No frames received for {max_no_frame} attempts, stopping hybrid stream for camera {config.camera_id}")
                        break
                    
                    # Different sleep patterns based on camera type
                    if detected_type == CameraType.BASLER:
                        await asyncio.sleep(0.1)  # Longer wait for Basler
                    else:
                        await asyncio.sleep(0.05)  # Shorter wait for regular
                    continue
                
                # Adaptive frame rate control
                if not stream_state.is_frozen:
                    elapsed = time.time() - frame_start_time
                    sleep_time = max(0, frame_time - elapsed)
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                else:
                    # When frozen, update less frequently to save resources
                    # But Basler cameras need slightly more frequent updates
                    if detected_type == CameraType.BASLER:
                        await asyncio.sleep(0.15)
                    else:
                        await asyncio.sleep(0.1)
                
            except asyncio.CancelledError:
                logger.info(f"🛑 Hybrid stream cancelled for camera {config.camera_id}, consumer {consumer_id}")
                break
            except Exception as e:
                logger.error(f"❌ Error in hybrid frame generation: {e}")
                await asyncio.sleep(0.1)

    except Exception as e:
        logger.error(f"❌ Error in generate_hybrid_video_frames: {e}")
    finally:
        # Cleanup
        if stream_key and stream_state:
            await stream_state.remove_consumer(consumer_id)
        
        camera_type_name = stream_state.config.camera_type.value if stream_state else "unknown"
        logger.info(f"🏁 Hybrid video stream stopped for {camera_type_name} camera {config.camera_id}, consumer {consumer_id} (streamed {frame_count} frames)")

@hybrid_router.post("/stream/{camera_id}/freeze")
async def freeze_stream(camera_id: int):
    """Freeze the video stream for detection (works with both camera types)"""
    try:
        stream_state = hybrid_stream_manager.get_stream_by_camera_id(camera_id)
        if not stream_state:
            raise HTTPException(status_code=404, detail=f"No active hybrid stream found for camera {camera_id}")
        
        success = await stream_state.freeze_stream()
        
        if success:
            return {
                "camera_id": camera_id,
                "camera_type": stream_state.config.camera_type.value,
                "status": "frozen",
                "message": f"Stream frozen for {stream_state.config.camera_type.value} camera {camera_id}",
                "timestamp": time.time()
            }
        else:
            raise HTTPException(status_code=400, detail=f"Failed to freeze stream for camera {camera_id}")
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error freezing stream for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to freeze stream: {str(e)}")

@hybrid_router.post("/stream/{camera_id}/unfreeze")
async def unfreeze_stream(camera_id: int):
    """Unfreeze the video stream to resume live feed (works with both camera types)"""
    try:
        stream_state = hybrid_stream_manager.get_stream_by_camera_id(camera_id)
        if not stream_state:
            raise HTTPException(status_code=404, detail=f"No active hybrid stream found for camera {camera_id}")
        
        await stream_state.unfreeze_stream()
        
        return {
            "camera_id": camera_id,
            "camera_type": stream_state.config.camera_type.value,
            "status": "unfrozen",
            "message": f"Stream resumed for {stream_state.config.camera_type.value} camera {camera_id}",
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error unfreezing stream for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to unfreeze stream: {str(e)}")

@hybrid_router.get("/stream/{camera_id}/status")
async def get_stream_status(camera_id: int):
    """Get the current status of a hybrid video stream"""
    try:
        stream_state = hybrid_stream_manager.get_stream_by_camera_id(camera_id)
        if not stream_state:
            return {
                "camera_id": camera_id,
                "stream_active": False,
                "camera_type": "unknown",
                "is_frozen": False,
                "consumers_count": 0,
                "message": "No active stream found"
            }
        
        stats = stream_state.get_stats()
        
        return {
            "camera_id": camera_id,
            "camera_type": stats['camera_type'],
            "stream_active": stats['is_active'],
            "is_frozen": stats['is_frozen'],
            "consumers_count": stats['consumers_count'],
            "frames_processed": stats['frames_processed'],
            "last_frame_time": stats['last_frame_time'],
            "stream_quality": stats['stream_quality'],
            "target_fps": stats['target_fps'],
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error getting stream status for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stream status: {str(e)}")

@hybrid_router.get("/stats")
async def get_hybrid_streaming_stats():
    """Get comprehensive hybrid streaming statistics"""
    try:
        stream_stats = []
        for stream_key, stream_state in hybrid_stream_manager.active_streams.items():
            try:
                stats = stream_state.get_stats()
                stream_stats.append(stats)
            except Exception as e:
                logger.error(f"Error getting stats for stream {stream_key}: {e}")
        
        total_consumers = sum(stat.get('consumers_count', 0) for stat in stream_stats)
        frozen_streams = sum(1 for stat in stream_stats if stat.get('is_frozen', False))
        basler_streams = sum(1 for stat in stream_stats if stat.get('camera_type') == 'basler')
        regular_streams = sum(1 for stat in stream_stats if stat.get('camera_type') == 'regular')
        
        return {
            "manager_stats": hybrid_stream_manager.stats,
            "stream_stats": stream_stats,
            "total_active_streams": len(stream_stats),
            "total_consumers": total_consumers,
            "frozen_streams": frozen_streams,
            "live_streams": len(stream_stats) - frozen_streams,
            "basler_streams": basler_streams,
            "regular_streams": regular_streams,
            "camera_type_breakdown": {
                "basler": basler_streams,
                "regular": regular_streams,
                "unknown": len(stream_stats) - basler_streams - regular_streams
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting hybrid streaming stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@hybrid_router.post("/stream/{camera_id}/stop")
async def stop_hybrid_stream(camera_id: int):
    """Stop all hybrid streams for a specific camera"""
    try:
        matching_streams = [
            key for key, stream in hybrid_stream_manager.active_streams.items()
            if stream.config.camera_id == camera_id
        ]
        
        if not matching_streams:
            raise HTTPException(status_code=404, detail=f"No active hybrid stream found for camera {camera_id}")
        
        stopped_count = 0
        camera_types = []
        for stream_key in matching_streams:
            try:
                stream_state = hybrid_stream_manager.active_streams[stream_key]
                camera_types.append(stream_state.config.camera_type.value)
                await hybrid_stream_manager.remove_stream(stream_key)
                stopped_count += 1
            except Exception as e:
                logger.error(f"Error stopping hybrid stream {stream_key}: {e}")
        
        return {
            "camera_id": camera_id,
            "stopped_streams": stopped_count,
            "camera_types": camera_types,
            "message": f"Stopped {stopped_count} hybrid streams for camera {camera_id}"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error stopping hybrid streams for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop streams: {str(e)}")

@hybrid_router.get("/health")
async def hybrid_streaming_health():
    """Health check for hybrid streaming service"""
    try:
        stats = hybrid_stream_manager.stats
        
        # Get detailed health info
        active_streams = len(hybrid_stream_manager.active_streams)
        frozen_count = 0
        total_consumers = 0
        basler_count = 0
        regular_count = 0
        
        for stream_state in hybrid_stream_manager.active_streams.values():
            if stream_state.is_frozen:
                frozen_count += 1
            total_consumers += len(stream_state.consumers)
            
            if stream_state.config.camera_type == CameraType.BASLER:
                basler_count += 1
            elif stream_state.config.camera_type == CameraType.REGULAR:
                regular_count += 1
        
        return {
            "status": "healthy",
            "active_streams": active_streams,
            "frozen_streams": frozen_count,
            "live_streams": active_streams - frozen_count,
            "total_consumers": total_consumers,
            "camera_types": {
                "basler": basler_count,
                "regular": regular_count,
                "unknown": active_streams - basler_count - regular_count
            },
            "total_frames_streamed": stats.get('total_frames_streamed', 0),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Hybrid streaming health check error: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": time.time()
        }

@hybrid_router.get("/stream/{camera_id}/current-frame")
async def get_current_frame(camera_id: int):
    """Get current frame as JPEG for detection service (works with both camera types)"""
    try:
        stream_state = hybrid_stream_manager.get_stream_by_camera_id(camera_id)
        if not stream_state:
            raise HTTPException(status_code=404, detail=f"No active hybrid stream found for camera {camera_id}")
        
        frame_bytes = await stream_state.get_latest_frame()
        if frame_bytes is None:
            raise HTTPException(status_code=404, detail=f"No frame available for camera {camera_id}")
        
        return Response(
            content=frame_bytes,
            media_type="image/jpeg",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
                "X-Camera-Type": stream_state.config.camera_type.value
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current frame for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get current frame: {str(e)}")

@hybrid_router.post("/stream/{camera_id}/update-frozen-frame")
async def update_frozen_frame(camera_id: int, frame: UploadFile = File(...)):
    """Update the frozen frame with detection results (works with both camera types)"""
    try:
        stream_state = hybrid_stream_manager.get_stream_by_camera_id(camera_id)
        if not stream_state:
            raise HTTPException(status_code=404, detail=f"No active hybrid stream found for camera {camera_id}")
        
        if not stream_state.is_frozen:
            raise HTTPException(status_code=400, detail=f"Stream for camera {camera_id} is not frozen")
        
        # Read the uploaded frame
        frame_bytes = await frame.read()
        
        # Update the frozen frame
        await stream_state.update_frozen_frame(frame_bytes)
        
        return {
            "camera_id": camera_id,
            "camera_type": stream_state.config.camera_type.value,
            "message": "Frozen frame updated successfully",
            "frame_size": len(frame_bytes),
            "timestamp": time.time()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating frozen frame for camera {camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to update frozen frame: {str(e)}")

@hybrid_router.post("/configure-camera-type")
async def configure_camera_type(request: CameraTypeRequest):
    """Manually configure camera type for better streaming optimization"""
    try:
        # Map string to enum
        type_mapping = {
            "basler": CameraType.BASLER,
            "regular": CameraType.REGULAR,
            "auto": CameraType.UNKNOWN
        }
        
        camera_type = type_mapping.get(request.camera_type.lower(), CameraType.UNKNOWN)
        
        # Check if there's already an active stream for this camera
        existing_stream = hybrid_stream_manager.get_stream_by_camera_id(request.camera_id)
        if existing_stream:
            # Update the existing stream's configuration
            existing_stream.config.camera_type = camera_type
            message = f"Updated existing stream configuration for camera {request.camera_id}"
        else:
            message = f"Camera type configuration stored for future streams of camera {request.camera_id}"
        
        return {
            "camera_id": request.camera_id,
            "camera_type": camera_type.value,
            "message": message,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Error configuring camera type for camera {request.camera_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to configure camera type: {str(e)}")

@hybrid_router.get("/supported-camera-types")
async def get_supported_camera_types():
    """Get list of supported camera types and their characteristics"""
    return {
        "supported_types": {
            "regular": {
                "name": "Regular/OpenCV Camera",
                "streaming_method": "MJPEG streaming",
                "optimal_fps": "15-30 FPS",
                "characteristics": [
                    "High-throughput streaming",
                    "Continuous frame capture",
                    "Lower latency",
                    "Good for real-time applications"
                ]
            },
            "basler": {
                "name": "Basler Industrial Camera",
                "streaming_method": "Polling-based capture",
                "optimal_fps": "10-20 FPS",
                "characteristics": [
                    "Reliable frame capture",
                    "Better error handling",
                    "Stable operation",
                    "Good for detection applications"
                ]
            },
            "auto": {
                "name": "Automatic Detection",
                "streaming_method": "Adaptive based on detected type",
                "optimal_fps": "Variable",
                "characteristics": [
                    "Automatic camera type detection",
                    "Optimal streaming method selection",
                    "Fallback to regular if detection fails"
                ]
            }
        },
        "default_type": "auto",
        "recommendation": "Use 'auto' for automatic detection, or specify 'basler' or 'regular' if known"
    }