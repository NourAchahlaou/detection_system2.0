from typing import AsyncGenerator

import aiohttp
import cv2
import numpy as np
from video_streaming.app.service.camera import CameraService
from video_streaming.app.service.video_streaming_service_with_detection import OptimizedStreamStateWithDetection
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import time
import logging
import asyncio
import uuid

from video_streaming.app.db.session import get_session
from video_streaming.app.service.video_streaming_service import (
    generate_video_frames_optimized,
    start_camera_via_hardware_service, 
    stop_camera_via_hardware_service,
    stream_manager,
    active_stream_states,
    wait_for_camera_ready
)

# Set up logging
from fastapi import APIRouter

HARDWARE_SERVICE_URL = "http://host.docker.internal:8003"
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)
camera_service = CameraService()
detection_streaming_router = APIRouter(
    prefix="/video",
    tags=["Video Streaming"],
    responses={404: {"description": "Not found"}},
)

@detection_streaming_router.get("/stream_with_detection/{camera_id}")
async def video_stream_with_detection(
    camera_id: int, 
    target_label: str = Query(..., description="Target object to detect"),
    db: Session = Depends(get_session)
):
    """
    Video streaming endpoint with real-time detection overlay
    """
    logger.info(f"Starting detection-enabled stream for camera {camera_id}, target: {target_label}")
    
    # Generate unique consumer ID
    consumer_id = f"detection_consumer_{uuid.uuid4().hex[:8]}_{camera_id}"
    
    return StreamingResponse(
        generate_video_frames_with_detection(camera_id, target_label, db, consumer_id),
        media_type='multipart/x-mixed-replace; boundary=frame',
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "Connection": "close",
            "X-Consumer-ID": consumer_id,
            "X-Detection-Target": target_label
        }
    )

async def generate_video_frames_with_detection(
    camera_id: int, 
    target_label: str, 
    db: Session, 
    consumer_id: str = None
) -> AsyncGenerator[bytes, None]:
    """Generate video frames with detection overlay"""
    
    if consumer_id is None:
        consumer_id = f"detection_consumer_{time.time()}_{id(asyncio.current_task())}"
    
    # Get or create stream state (using your existing logic but with detection)
    if camera_id not in active_stream_states:
        try:
            camera = camera_service.get_camera_by_id(db, camera_id)
            camera_index = camera.camera_index
        except Exception as e:
            logger.error(f"Failed to get camera details for ID {camera_id}: {e}")
            return
        
        # Create new stream state with detection
        stream_state = OptimizedStreamStateWithDetection(camera_id)
        active_stream_states[camera_id] = stream_state
        
        # Enable detection
        await stream_state.enable_detection(target_label)
        
        # Your existing camera startup logic...
        camera_started = await start_camera_via_hardware_service(camera_index)
        if not camera_started:
            logger.error("Failed to start camera via hardware service")
            del active_stream_states[camera_id]
            return
        
        camera_ready = await wait_for_camera_ready(camera_index)
        if not camera_ready:
            logger.error("Camera did not become ready in time")
            await stop_camera_via_hardware_service()
            del active_stream_states[camera_id]
            return
        
        stream_state.is_active = True
        stream_manager.add_camera_stream(camera_id, stream_state.stop_event)
        
        # Start frame producer with detection
        stream_state.connection_task = asyncio.create_task(
            single_connection_frame_producer_with_detection(stream_state)
        )
        
        logger.info(f"Created new detection-enabled stream for camera {camera_id}")
    
    stream_state = active_stream_states[camera_id]
    await stream_state.add_consumer(consumer_id)
    
    # Your existing frame generation logic remains the same
    # The detection happens in the frame producer, so consumers just get processed frames
    try:
        target_fps = 25
        frame_time = 1.0 / target_fps
        
        while stream_state.is_active and not stream_state.stop_event.is_set():
            frame_start_time = time.time()
            
            try:
                frame_bytes = await stream_state.get_latest_frame()
                
                if frame_bytes is not None:
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    await asyncio.sleep(0.05)
                    continue
                
                elapsed = time.time() - frame_start_time
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    try:
                        await asyncio.wait_for(asyncio.sleep(sleep_time), timeout=sleep_time + 0.01)
                    except asyncio.TimeoutError:
                        pass
                
            except asyncio.CancelledError:
                logger.info(f"Detection stream cancelled for camera {camera_id}, consumer {consumer_id}")
                break
            except Exception as e:
                logger.error(f"Error in detection frame generation loop: {e}")
                await asyncio.sleep(0.1)

    except Exception as e:
        logger.error(f"Error in generate_video_frames_with_detection: {e}")
    finally:
        if camera_id in active_stream_states:
            await active_stream_states[camera_id].remove_consumer(consumer_id)
        
        logger.info(f"Detection-enabled video stream stopped for camera ID {camera_id}, consumer {consumer_id}")

async def single_connection_frame_producer_with_detection(stream_state: OptimizedStreamStateWithDetection):
    """Frame producer that integrates with detection service"""
    logger.info(f"Starting detection-enabled frame producer for camera {stream_state.camera_id}")
    
    # Your existing frame producer logic, but replace add_frame with add_frame_with_detection
    timeout = aiohttp.ClientTimeout(total=None)
    session = aiohttp.ClientSession(timeout=timeout)
    stream_state.session = session
    
    try:
        while stream_state.is_active and not stream_state.stop_event.is_set():
            # Your existing frame capture logic...
            try:
                async with session.get(f"{HARDWARE_SERVICE_URL}/camera/video_feed") as response:
                    if response.status != 200:
                        logger.error(f"Frame stream failed with status {response.status}")
                        await asyncio.sleep(1.0)
                        continue
                    
                    buffer = bytearray()
                    async for chunk in response.content.iter_chunked(8192):
                        if stream_state.stop_event.is_set() or not stream_state.consumers:
                            break
                            
                        buffer.extend(chunk)
                        
                        # Process complete JPEG frames
                        while True:
                            jpeg_start = buffer.find(b'\xff\xd8')
                            if jpeg_start == -1:
                                break
                            
                            jpeg_end = buffer.find(b'\xff\xd9', jpeg_start + 2)
                            if jpeg_end == -1:
                                break
                            
                            jpeg_data = bytes(buffer[jpeg_start:jpeg_end + 2])
                            buffer = buffer[jpeg_end + 2:]
                            
                            try:
                                nparr = np.frombuffer(jpeg_data, np.uint8)
                                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                
                                if frame is not None and frame.size > 0:
                                    # Resize frame
                                    from video_streaming.app.service.video_streaming_service import resize_frame_optimized
                                    frame = resize_frame_optimized(frame, (640, 480))
                                    
                                    # Add frame with detection processing
                                    await stream_state.add_frame_with_detection(frame)
                                    stream_state.consecutive_failures = 0
                                    
                            except Exception as e:
                                logger.debug(f"Frame decode error: {e}")
                                
                        if len(buffer) > 100000:
                            buffer = buffer[-50000:]
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in detection frame producer: {e}")
                stream_state.consecutive_failures += 1
                if stream_state.consecutive_failures > 5:
                    logger.error("Too many consecutive failures, stopping producer")
                    break
                await asyncio.sleep(1.0)
    
    finally:
        logger.info(f"Detection frame producer stopped for camera {stream_state.camera_id}")
        if session and not session.closed:
            await session.close()