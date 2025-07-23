import cv2
import aiohttp
import asyncio
import logging
import threading
import numpy as np
from typing import AsyncGenerator, Optional
from fastapi import APIRouter
from sqlalchemy.orm import Session
import time
import weakref

from video_streaming.app.service.videoStremManager import VideoStreamManager
from video_streaming.app.service.hardwareServiceClient import CameraClient
from video_streaming.app.service.camera import CameraService

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger(__name__)

# Configuration for direct hardware connection
HARDWARE_SERVICE_URL = "http://host.docker.internal:8003"

# Initialize hardware client and camera service
hardware_client = CameraClient(base_url=HARDWARE_SERVICE_URL)
camera_service = CameraService()

# Global stream manager with improved tracking
stream_manager = VideoStreamManager()

class StreamState:
    """Track individual stream state more precisely"""
    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self.is_active = False
        self.stop_event = asyncio.Event()
        self.frame_count = 0
        self.last_frame_time = 0
        self.consecutive_failures = 0
        self.session = None
        
    async def cleanup(self):
        """Clean up stream resources"""
        self.is_active = False
        self.stop_event.set()
        if self.session and not self.session.closed:
            await self.session.close()

# Global stream states tracking
active_stream_states = {}

def resize_frame_optimized(frame: np.ndarray, target_size=(640, 480)) -> np.ndarray:
    """Optimized frame resizing with better interpolation."""
    if frame.shape[:2] != target_size[::-1]:
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    return frame

async def check_camera_status_safe() -> Optional[dict]:
    """Safely check camera status with proper error handling."""
    try:
        timeout = aiohttp.ClientTimeout(total=2.0)  # Reduced timeout
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(f"{HARDWARE_SERVICE_URL}/camera/check_camera") as response:
                if response.status == 200:
                    return await response.json()
                else:
                    logger.warning(f"Camera check returned status {response.status}")
                    return None
    except asyncio.TimeoutError:
        logger.warning("Camera check timed out")
        return None
    except Exception as e:
        logger.warning(f"Error checking camera status: {e}")
        return None

async def start_camera_via_hardware_service(camera_index: int) -> bool:
    """Start camera using the hardware service with better error handling."""
    try:
        # Cleanup first - FIX: Use sync method in async context properly
        try:
            # Since cleanup_temp_photos is synchronous, run it in a thread
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, hardware_client.cleanup_temp_photos)
        except Exception as e:
            logger.warning(f"Cleanup failed: {e}")
        
        timeout = aiohttp.ClientTimeout(total=8.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            payload = {"camera_index": camera_index}
            
            async with session.post(f"{HARDWARE_SERVICE_URL}/camera/opencv/start", json=payload) as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Camera start response: {result}")
                    
                    if result.get("status") == "success" or "started successfully" in result.get("message", "").lower():
                        logger.info(f"Camera index {camera_index} started successfully")
                        return True
                    else:
                        logger.error(f"Camera start failed: {result}")
                        return False
                else:
                    logger.error(f"Camera start request failed with status {response.status}")
                    return False
            
    except Exception as e:
        logger.error(f"Failed to start camera via hardware service: {e}")
        return False

async def stop_camera_via_hardware_service():
    """Stop camera using the hardware service with immediate response."""
    try:
        timeout = aiohttp.ClientTimeout(total=3.0)  # Reduced timeout for faster response
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{HARDWARE_SERVICE_URL}/camera/stop") as response:
                if response.status == 200:
                    result = await response.json()
                    logger.info(f"Camera stop response: {result}")
                else:
                    logger.warning(f"Camera stop request returned status {response.status}")
        
        # Cleanup in background to avoid blocking
        asyncio.create_task(cleanup_resources())
        
        return True
    except Exception as e:
        logger.error(f"Failed to stop camera via hardware service: {e}")
        return False

async def cleanup_resources():
    """Background cleanup task - FIX: Handle sync method properly"""
    try:
        await asyncio.sleep(0.1)  # Small delay
        # Run synchronous cleanup in executor
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, hardware_client.cleanup_temp_photos)
    except Exception as e:
        logger.warning(f"Background cleanup failed: {e}")

async def capture_frame_from_hardware_single_request(session: aiohttp.ClientSession, stream_state: StreamState) -> Optional[np.ndarray]:
    """
    Optimized frame capture using a single persistent session to avoid multiple concurrent requests.
    This prevents the multiple GET /camera/video_feed requests you're seeing.
    """
    if stream_state.stop_event.is_set():
        return None
        
    try:
        # Use the persistent session instead of creating new ones
        async with session.get(f"{HARDWARE_SERVICE_URL}/camera/video_feed") as response:
            if response.status != 200:
                logger.debug(f"Frame request failed with status {response.status}")
                return None
                
            buffer = bytearray()
            chunk_size = 16384
            max_read_time = 3.0  # Maximum time to spend reading
            start_time = time.time()
            
            async for chunk in response.content.iter_chunked(chunk_size):
                if stream_state.stop_event.is_set():
                    logger.debug("Stop event detected during frame capture")
                    return None
                    
                if time.time() - start_time > max_read_time:
                    logger.debug("Frame capture timeout")
                    break
                    
                buffer.extend(chunk)
                
                # Look for complete JPEG frame
                jpeg_start = buffer.find(b'\xff\xd8')
                if jpeg_start != -1:
                    jpeg_end = buffer.find(b'\xff\xd9', jpeg_start + 2)
                    if jpeg_end != -1:
                        # Extract and decode JPEG
                        jpeg_data = bytes(buffer[jpeg_start:jpeg_end + 2])
                        
                        try:
                            nparr = np.frombuffer(jpeg_data, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if frame is not None and frame.size > 0:
                                return frame
                        except Exception as e:
                            logger.debug(f"Frame decode error: {e}")
                            
                        # Remove processed data
                        buffer = buffer[jpeg_end + 2:]
                
                # Prevent excessive memory usage
                if len(buffer) > 200000:  # 200KB
                    break
                    
    except asyncio.CancelledError:
        logger.debug("Frame capture cancelled")
        return None
    except Exception as e:
        logger.debug(f"Frame capture error: {e}")
        
    return None

async def wait_for_camera_ready(camera_index: int, max_wait_time: int = 10) -> bool:
    """Wait for camera to be ready with improved checking."""
    logger.info(f"Waiting for camera index {camera_index} to be ready...")
    
    start_time = time.time()
    
    while time.time() - start_time < max_wait_time:
        try:
            check_response = await check_camera_status_safe()
            
            if check_response and check_response.get("camera_opened", False):
                logger.info(f"Camera index {camera_index} reports as ready")
                return True
                    
            await asyncio.sleep(1.0)
            
        except Exception as e:
            logger.warning(f"Error during camera readiness check: {e}")
            await asyncio.sleep(1.0)
    
    logger.error(f"Camera index {camera_index} not ready after {max_wait_time} seconds")
    return False

async def generate_video_frames(camera_id: int, db: Session) -> AsyncGenerator[bytes, None]:
    """
    Generate high-performance video frames with improved stop handling and 
    single session to prevent multiple concurrent requests.
    """
    
    # Check if camera is already streaming
    if camera_id in active_stream_states:
        logger.warning(f"Camera {camera_id} is already streaming")
        return
    
    # Create stream state
    stream_state = StreamState(camera_id)
    active_stream_states[camera_id] = stream_state
    
    try:
        # Get camera details
        camera = camera_service.get_camera_by_id(db, camera_id)
        camera_index = camera.camera_index
        logger.info(f"Starting video stream for camera ID {camera_id} with camera index {camera_index}")
    except Exception as e:
        logger.error(f"Failed to get camera details for ID {camera_id}: {e}")
        return
    
    # Create persistent session for this stream
    timeout = aiohttp.ClientTimeout(total=10.0)
    session = aiohttp.ClientSession(timeout=timeout)
    stream_state.session = session
    
    try:
        # Start camera
        camera_started = await start_camera_via_hardware_service(camera_index)
        if not camera_started:
            logger.error("Failed to start camera via hardware service")
            return
        
        # Wait for camera readiness
        camera_ready = await wait_for_camera_ready(camera_index)
        if not camera_ready:
            logger.error("Camera did not become ready in time")
            await stop_camera_via_hardware_service()
            return
        
        # Mark stream as active and register with stream manager
        stream_state.is_active = True
        stream_manager.add_camera_stream(camera_id, stream_state.stop_event)
        
        # Optimized encoding settings
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        target_fps = 25  # Slightly reduced for stability
        frame_time = 1.0 / target_fps
        max_consecutive_failures = 3  # Reduced threshold
        
        logger.info(f"Starting frame generation loop for camera {camera_id}")
        
        while stream_state.is_active and not stream_state.stop_event.is_set():
            frame_start_time = time.time()
            
            try:
                # Use single session for frame capture
                frame = await capture_frame_from_hardware_single_request(session, stream_state)

                if frame is None:
                    stream_state.consecutive_failures += 1
                    logger.debug(f"Frame capture failed {stream_state.consecutive_failures} times")
                    
                    if stream_state.consecutive_failures >= max_consecutive_failures:
                        logger.error("Too many consecutive frame failures, stopping camera")
                        break
                        
                    await asyncio.sleep(0.05)  # Short delay before retry
                    continue
                else:
                    stream_state.consecutive_failures = 0
                
                # Process frame
                frame = resize_frame_optimized(frame, (640, 480))
                
                # Encode and yield frame
                if frame is not None and len(frame.shape) == 3:
                    try:
                        _, buffer = cv2.imencode('.jpg', frame, encode_params)
                        if buffer is not None:
                            frame_bytes = buffer.tobytes()
                            yield (b'--frame\r\n'
                                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                            
                            stream_state.frame_count += 1
                            stream_state.last_frame_time = time.time()
                    except Exception as e:
                        logger.error(f"Error encoding frame: {e}")

                # Frame rate control
                elapsed = time.time() - frame_start_time
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    try:
                        await asyncio.wait_for(asyncio.sleep(sleep_time), timeout=sleep_time + 0.01)
                    except asyncio.TimeoutError:
                        pass  # Continue if sleep is interrupted
                
            except asyncio.CancelledError:
                logger.info(f"Video stream cancelled for camera {camera_id}")
                break
            except Exception as e:
                logger.error(f"Error in frame generation loop: {e}")
                stream_state.consecutive_failures += 1
                if stream_state.consecutive_failures >= max_consecutive_failures:
                    break
                await asyncio.sleep(0.05)

    except Exception as e:
        logger.error(f"Error in generate_video_frames: {e}")
    finally:
        # Immediate cleanup
        logger.info(f"Cleaning up video stream for camera {camera_id}")
        
        # Mark as inactive immediately
        stream_state.is_active = False
        stream_state.stop_event.set()
        
        # Remove from tracking
        if camera_id in active_stream_states:
            del active_stream_states[camera_id]
        
        # Close session
        if session and not session.closed:
            await session.close()
        
        # Stop camera service
        await stop_camera_via_hardware_service()
        logger.info(f"Video stream stopped for camera ID {camera_id}")