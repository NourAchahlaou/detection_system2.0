import cv2
import aiohttp
import asyncio
import logging
import threading
import numpy as np
from typing import AsyncGenerator, Optional, Dict
from fastapi import APIRouter
from sqlalchemy.orm import Session
import time
import weakref
from collections import deque

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

class OptimizedStreamState:
    """Enhanced stream state with frame buffering and single connection management"""
    def __init__(self, camera_id: int):
        self.camera_id = camera_id
        self.is_active = False
        self.stop_event = asyncio.Event()
        self.frame_count = 0
        self.last_frame_time = 0
        self.consecutive_failures = 0
        self.session = None
        
        # Frame buffering for multiple consumers
        self.frame_buffer = deque(maxlen=5)  # Keep last 5 frames
        self.buffer_lock = asyncio.Lock()
        self.latest_frame = None
        self.frame_ready_event = asyncio.Event()
        
        # Single connection management
        self.connection_task = None
        self.consumers = set()  # Track active consumers
        
    async def add_consumer(self, consumer_id: str):
        """Add a consumer to this stream"""
        self.consumers.add(consumer_id)
        logger.info(f"Added consumer {consumer_id} to camera {self.camera_id}. Total consumers: {len(self.consumers)}")
        
    async def remove_consumer(self, consumer_id: str):
        """Remove a consumer from this stream"""
        self.consumers.discard(consumer_id)
        logger.info(f"Removed consumer {consumer_id} from camera {self.camera_id}. Total consumers: {len(self.consumers)}")
        
        # If no consumers left, stop the stream
        if not self.consumers and self.is_active:
            logger.info(f"No consumers left for camera {self.camera_id}, stopping stream")
            await self.cleanup()
        
    async def add_frame(self, frame: np.ndarray):
        """Add a frame to the buffer"""
        async with self.buffer_lock:
            # Encode frame once for all consumers
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
            _, buffer = cv2.imencode('.jpg', frame, encode_params)
            if buffer is not None:
                frame_bytes = buffer.tobytes()
                self.frame_buffer.append({
                    'data': frame_bytes,
                    'timestamp': time.time()
                })
                self.latest_frame = frame_bytes
                self.frame_ready_event.set()
                self.frame_count += 1
                self.last_frame_time = time.time()
        
    async def get_latest_frame(self) -> Optional[bytes]:
        """Get the latest frame for streaming"""
        if self.latest_frame is not None:
            return self.latest_frame
        
        # Wait for first frame if none available
        try:
            await asyncio.wait_for(self.frame_ready_event.wait(), timeout=1.0)
            return self.latest_frame
        except asyncio.TimeoutError:
            return None
        
    async def cleanup(self):
        """Clean up stream resources"""
        self.is_active = False
        self.stop_event.set()
        
        if self.connection_task and not self.connection_task.done():
            self.connection_task.cancel()
            try:
                await self.connection_task
            except asyncio.CancelledError:
                pass
        
        if self.session and not self.session.closed:
            await self.session.close()
            
        self.consumers.clear()
        self.frame_buffer.clear()

# Global stream states tracking
active_stream_states: Dict[int, OptimizedStreamState] = {}

def resize_frame_optimized(frame: np.ndarray, target_size=(640, 480)) -> np.ndarray:
    """Optimized frame resizing with better interpolation."""
    if frame.shape[:2] != target_size[::-1]:
        return cv2.resize(frame, target_size, interpolation=cv2.INTER_LINEAR)
    return frame

async def check_camera_status_safe() -> Optional[dict]:
    """Safely check camera status with proper error handling."""
    try:
        timeout = aiohttp.ClientTimeout(total=2.0)
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
        # Cleanup first
        try:
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
        timeout = aiohttp.ClientTimeout(total=3.0)
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
    """Background cleanup task"""
    try:
        await asyncio.sleep(0.1)
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, hardware_client.cleanup_temp_photos)
    except Exception as e:
        logger.warning(f"Background cleanup failed: {e}")

async def single_connection_frame_producer(stream_state: OptimizedStreamState):
    """
    SINGLE CONNECTION PRODUCER: This replaces multiple GET requests with one persistent connection
    that continuously reads frames and distributes them to all consumers.
    """
    logger.info(f"Starting single connection frame producer for camera {stream_state.camera_id}")
    
    timeout = aiohttp.ClientTimeout(total=None)  # No timeout for persistent connection
    session = aiohttp.ClientSession(timeout=timeout)
    stream_state.session = session
    
    try:
        while stream_state.is_active and not stream_state.stop_event.is_set():
            try:
                # Single persistent connection for continuous frame reading
                async with session.get(f"{HARDWARE_SERVICE_URL}/camera/video_feed") as response:
                    if response.status != 200:
                        logger.error(f"Frame stream failed with status {response.status}")
                        await asyncio.sleep(1.0)
                        continue
                    
                    buffer = bytearray()
                    async for chunk in response.content.iter_chunked(8192):
                        if stream_state.stop_event.is_set():
                            break
                            
                        if not stream_state.consumers:
                            # No consumers, break and restart connection when needed
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
                            
                            # Extract complete JPEG frame
                            jpeg_data = bytes(buffer[jpeg_start:jpeg_end + 2])
                            buffer = buffer[jpeg_end + 2:]
                            
                            try:
                                # Decode and process frame
                                nparr = np.frombuffer(jpeg_data, np.uint8)
                                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                
                                if frame is not None and frame.size > 0:
                                    # Resize frame once for all consumers
                                    frame = resize_frame_optimized(frame, (640, 480))
                                    
                                    # Add to buffer for all consumers
                                    await stream_state.add_frame(frame)
                                    stream_state.consecutive_failures = 0
                                    
                            except Exception as e:
                                logger.debug(f"Frame decode error: {e}")
                                
                        # Prevent buffer overflow
                        if len(buffer) > 100000:  # 100KB max buffer
                            buffer = buffer[-50000:]  # Keep last 50KB
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in frame producer: {e}")
                stream_state.consecutive_failures += 1
                if stream_state.consecutive_failures > 5:
                    logger.error("Too many consecutive failures, stopping producer")
                    break
                await asyncio.sleep(1.0)
    
    finally:
        logger.info(f"Frame producer stopped for camera {stream_state.camera_id}")
        if session and not session.closed:
            await session.close()

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

async def generate_video_frames_optimized(camera_id: int, db: Session, consumer_id: str = None) -> AsyncGenerator[bytes, None]:
    """
    OPTIMIZED FRAME GENERATOR: Uses shared stream state to eliminate multiple connections.
    Multiple consumers share the same frame producer.
    """
    
    if consumer_id is None:
        consumer_id = f"consumer_{time.time()}_{id(asyncio.current_task())}"
    
    # Get or create stream state
    if camera_id not in active_stream_states:
        # Get camera details
        try:
            camera = camera_service.get_camera_by_id(db, camera_id)
            camera_index = camera.camera_index
        except Exception as e:
            logger.error(f"Failed to get camera details for ID {camera_id}: {e}")
            return
        
        # Create new stream state
        stream_state = OptimizedStreamState(camera_id)
        active_stream_states[camera_id] = stream_state
        
        # Start camera hardware
        camera_started = await start_camera_via_hardware_service(camera_index)
        if not camera_started:
            logger.error("Failed to start camera via hardware service")
            del active_stream_states[camera_id]
            return
        
        # Wait for camera readiness
        camera_ready = await wait_for_camera_ready(camera_index)
        if not camera_ready:
            logger.error("Camera did not become ready in time")
            await stop_camera_via_hardware_service()
            del active_stream_states[camera_id]
            return
        
        # Mark stream as active and start the single connection producer
        stream_state.is_active = True
        stream_manager.add_camera_stream(camera_id, stream_state.stop_event)
        
        # Start the single connection frame producer
        stream_state.connection_task = asyncio.create_task(
            single_connection_frame_producer(stream_state)
        )
        
        logger.info(f"Created new optimized stream for camera {camera_id}")
    
    stream_state = active_stream_states[camera_id]
    
    # Add this consumer to the stream
    await stream_state.add_consumer(consumer_id)
    
    try:
        logger.info(f"Starting optimized frame generation for camera {camera_id}, consumer {consumer_id}")
        
        target_fps = 25
        frame_time = 1.0 / target_fps
        
        while stream_state.is_active and not stream_state.stop_event.is_set():
            frame_start_time = time.time()
            
            try:
                # Get latest frame from shared buffer
                frame_bytes = await stream_state.get_latest_frame()
                
                if frame_bytes is not None:
                    yield (b'--frame\r\n'
                          b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
                else:
                    # No frame available, small delay
                    await asyncio.sleep(0.05)
                    continue
                
                # Frame rate control
                elapsed = time.time() - frame_start_time
                sleep_time = max(0, frame_time - elapsed)
                if sleep_time > 0:
                    try:
                        await asyncio.wait_for(asyncio.sleep(sleep_time), timeout=sleep_time + 0.01)
                    except asyncio.TimeoutError:
                        pass
                
            except asyncio.CancelledError:
                logger.info(f"Video stream cancelled for camera {camera_id}, consumer {consumer_id}")
                break
            except Exception as e:
                logger.error(f"Error in optimized frame generation loop: {e}")
                await asyncio.sleep(0.1)

    except Exception as e:
        logger.error(f"Error in generate_video_frames_optimized: {e}")
    finally:
        # Remove consumer
        if camera_id in active_stream_states:
            await active_stream_states[camera_id].remove_consumer(consumer_id)
        
        logger.info(f"Optimized video stream stopped for camera ID {camera_id}, consumer {consumer_id}")

# Legacy function for backward compatibility
async def generate_video_frames(camera_id: int, db: Session) -> AsyncGenerator[bytes, None]:
    """Legacy wrapper - redirects to optimized version"""
    async for frame in generate_video_frames_optimized(camera_id, db):
        yield frame