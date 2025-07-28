# fixed_optimized_detection_service.py - FIXED pubsub communication
import cv2
import asyncio
import logging
import redis.asyncio as redis
import redis as sync_redis
import redis.exceptions as redis_exceptions
import pickle
import numpy as np
from typing import Optional, Dict, Any
import time
from datetime import datetime
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from collections import deque
import psutil
import torch
import weakref

from detection.app.service.detection_service import DetectionSystem

# Enhanced logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class FrameProcessingRequest:
    """Structured frame processing request"""
    camera_id: int
    frame_data: bytes
    target_label: str
    timestamp: float
    session_id: str
    priority: int = 1

@dataclass
class DetectionResult:
    """Structured detection result"""
    camera_id: int
    session_id: str
    detected_target: bool
    non_target_count: int
    processing_time_ms: float
    confidence: Optional[float] = None
    bounding_boxes: Optional[list] = None
    timestamp: float = None
    frame_processed: bool = True
    # Add processed frame data
    processed_frame_data: Optional[bytes] = None

class OptimizedDetectionProcessor:
    """High-performance detection processor with FIXED pubsub communication"""

    def __init__(self, 
                redis_host='redis', 
                redis_port=6379,
                max_workers=2,
                max_queue_size=50,
                frame_timeout=5.0):
        
        # Redis connection
        self.redis_host = redis_host
        self.redis_port = redis_port
        self.redis_client = None
        self.sync_redis_client = None  # Add dedicated sync client
        
        # Processing configuration
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.frame_timeout = frame_timeout
        
        # Initialize detection system
        self.detection_system = None
        self.device = None
        
        # FIXED: Complete performance tracking initialization
        self.processing_stats = {
            'frames_processed': 0,
            'total_processing_time': 0,
            'queue_overflows': 0,
            'timeouts': 0,
            'overlays_created': 0,  # ← This was missing!
            'frames_stored': 0,
            'pubsub_messages_sent': 0,
            'detection_results_published': 0
        }
        
        # Thread pool for CPU-bound operations
        self.executor = None
        
        # Frame memory pool
        self.frame_pool = deque(maxlen=20)
        self.pool_lock = threading.Lock()
        
        # Processing queues with thread-safe access
        self.high_priority_queue = deque()
        self.normal_priority_queue = deque()
        self.queue_lock = threading.Lock()
        
        # Control flags
        self.is_running = False
        self.processor_thread = None
        self.redis_listener_task = None
        
        # Event loop management
        self._main_loop = None
        self._result_queue = asyncio.Queue() if hasattr(asyncio, 'Queue') else None
        self._result_sender_task = None
        
        # Cleanup tracking
        self._cleanup_tasks = []
    
    async def initialize(self):
        """Initialize the detection system and start processing"""
        try:
            logger.info("🚀 Initializing detection system...")
            
            # Store reference to the main event loop
            self._main_loop = asyncio.get_event_loop()
            
            # Create result queue for async communication
            self._result_queue = asyncio.Queue()
            
            # Create new executor if needed
            if self.executor is None or self.executor._shutdown:
                self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
            
            # Initialize detection system in thread pool
            self.detection_system = await self._main_loop.run_in_executor(
                self.executor, DetectionSystem
            )
            self.device = self.detection_system.device
            logger.info(f"✅ Detection system initialized on device: {self.device}")
            
            # Initialize async Redis connection
            await self._initialize_redis()
            
            # Start result sender task
            self._result_sender_task = asyncio.create_task(self._result_sender_loop())
            
            # Start processing thread
            self.is_running = True
            self.processor_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processor_thread.start()
            
            # Start Redis listener
            self.redis_listener_task = asyncio.create_task(self._redis_listener())
            
            logger.info("✅ Optimized detection processor initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize detection processor: {e}")
            await self.cleanup_on_error()
            raise
    
    async def _initialize_redis(self):
        """Initialize async Redis connection with proper error handling"""
        max_retries = 3
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                # Async Redis client
                self.redis_client = redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    db=0,
                    retry_on_timeout=True,
                    socket_keepalive=True,
                    health_check_interval=30,
                    decode_responses=False,
                    socket_connect_timeout=5,
                    socket_timeout=10,
                    max_connections=20
                )
                
                # FIXED: Create dedicated sync Redis client for pubsub operations
                self.sync_redis_client = sync_redis.Redis(
                    host=self.redis_host,
                    port=self.redis_port,
                    db=0,
                    decode_responses=False,
                    socket_keepalive=True,
                    socket_connect_timeout=5,
                    socket_timeout=10
                )
                
                # Test both connections
                await self.redis_client.ping()
                self.sync_redis_client.ping()  # Sync ping
                logger.info("✅ Async and sync Redis connections established successfully")
                
                # Test pubsub functionality
                await self._test_pubsub_connection()
                return
                
            except Exception as e:
                logger.error(f"❌ Redis connection attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(retry_delay)
                else:
                    raise
    
    async def _test_pubsub_connection(self):
        """Test that pubsub is working correctly"""
        try:
            test_channel = "test_pubsub_connection"
            test_message = b"test_message"
            
            # FIXED: Use sync client for pubsub operations
            subscribers = self.sync_redis_client.publish(test_channel, test_message)
            logger.info(f"✅ Pubsub test: published message to {subscribers} subscribers")
            
        except Exception as e:
            logger.error(f"❌ Pubsub test failed: {e}")
            raise
    
    async def cleanup_on_error(self):
        """Cleanup resources on initialization error"""
        self.is_running = False
        
        # Cancel async tasks
        for task in [self.redis_listener_task, self._result_sender_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close Redis connections
        if self.redis_client:
            try:
                await self.redis_client.aclose()
            except:
                pass
            self.redis_client = None
            
        if self.sync_redis_client:
            try:
                self.sync_redis_client.close()
            except:
                pass
            self.sync_redis_client = None
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=False)
    
    async def _redis_listener(self):
        """ASYNC Redis listener with proper error handling"""
        logger.info("🔊 Starting async Redis listener for frame processing requests")
        
        try:
            while self.is_running:
                try:
                    # Check if Redis client is still valid
                    if not self.redis_client:
                        logger.error("❌ Redis client is None, stopping listener")
                        break
                    
                    # Non-blocking Redis operation with timeout
                    request_data = await asyncio.wait_for(
                        self.redis_client.brpop(['detection_queue'], timeout=1),
                        timeout=2.0
                    )
                    
                    if request_data:
                        _, serialized_request = request_data
                        
                        try:
                            request = pickle.loads(serialized_request)
                            await self._queue_frame_for_processing(request)
                            
                        except Exception as e:
                            logger.error(f"❌ Failed to deserialize frame request: {e}")
                
                except asyncio.TimeoutError:
                    # Normal timeout, continue listening
                    continue
                except redis_exceptions.ConnectionError as e:
                    logger.error(f"❌ Redis connection error: {e}")
                    await asyncio.sleep(1)
                    # Try to reconnect
                    try:
                        await self._initialize_redis()
                    except Exception as reconnect_error:
                        logger.error(f"❌ Failed to reconnect to Redis: {reconnect_error}")
                        await asyncio.sleep(5)
                except asyncio.CancelledError:
                    logger.info("🛑 Redis listener task cancelled")
                    break
                except Exception as e:
                    logger.error(f"❌ Error in Redis listener: {e}")
                    await asyncio.sleep(0.1)
                    
        except Exception as e:
            logger.error(f"❌ Fatal error in Redis listener: {e}")
        finally:
            logger.info("🛑 Redis listener stopped")
    
    async def _queue_frame_for_processing(self, request_dict: Dict[str, Any]):
        """Queue frame for processing with priority handling"""
        try:
            # Create structured request
            request = FrameProcessingRequest(
                camera_id=request_dict['camera_id'],
                frame_data=request_dict['frame_data'],
                target_label=request_dict['target_label'],
                timestamp=request_dict['timestamp'],
                session_id=request_dict.get('session_id', 'default'),
                priority=request_dict.get('priority', 1)
            )
            
            # Check if frame is too old
            if time.time() - request.timestamp > self.frame_timeout:
                logger.debug(f"⏰ Dropping expired frame from camera {request.camera_id}")
                self.processing_stats['timeouts'] += 1
                return
            
            # Add to appropriate queue
            with self.queue_lock:
                total_queue_size = len(self.high_priority_queue) + len(self.normal_priority_queue)
                
                if total_queue_size >= self.max_queue_size:
                    # Drop oldest normal priority frame
                    if self.normal_priority_queue:
                        dropped = self.normal_priority_queue.popleft()
                        logger.debug(f"💧 Queue overflow: dropped frame from camera {dropped.camera_id}")
                        self.processing_stats['queue_overflows'] += 1
                
                if request.priority > 1:
                    self.high_priority_queue.append(request)
                else:
                    self.normal_priority_queue.append(request)
                    
        except Exception as e:
            logger.error(f"❌ Error queuing frame for processing: {e}")
    
    def _processing_loop(self):
        """Main processing loop running in separate thread"""
        logger.info("🔄 Starting frame processing loop")
        
        while self.is_running:
            try:
                request = self._get_next_request()
                
                if request:
                    # Process frame
                    result = self._process_frame_sync(request)
                    
                    # Queue result for async sending
                    if self._main_loop and not self._main_loop.is_closed():
                        try:
                            # Use thread-safe method to queue result
                            asyncio.run_coroutine_threadsafe(
                                self._result_queue.put(result),
                                self._main_loop
                            )
                        except RuntimeError as e:
                            if self.is_running:
                                logger.warning(f"⚠️ Could not queue result - event loop issue: {e}")
                    
                else:
                    # No requests available, short sleep
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"❌ Error in processing loop: {e}")
                time.sleep(0.1)
        
        logger.info("🛑 Processing loop stopped")
    
    async def _result_sender_loop(self):
        """FIXED: Async loop to send results back via Redis with proper pubsub"""
        logger.info("📤 Starting result sender loop")
        
        try:
            while self.is_running:
                try:
                    # Wait for results with timeout
                    result = await asyncio.wait_for(self._result_queue.get(), timeout=1.0)
                    await self._send_result_to_redis(result)
                    
                except asyncio.TimeoutError:
                    # Normal timeout, continue
                    continue
                except Exception as e:
                    logger.error(f"❌ Error in result sender loop: {e}")
                    await asyncio.sleep(0.1)
                    
        except asyncio.CancelledError:
            logger.info("🛑 Result sender loop cancelled")
        except Exception as e:
            logger.error(f"❌ Fatal error in result sender loop: {e}")
        finally:
            logger.info("🛑 Result sender loop stopped")
    
    def _get_next_request(self) -> Optional[FrameProcessingRequest]:
        """Get next request with priority handling"""
        with self.queue_lock:
            # Prioritize high priority queue
            if self.high_priority_queue:
                return self.high_priority_queue.popleft()
            elif self.normal_priority_queue:
                return self.normal_priority_queue.popleft()
        return None
    
    def _process_frame_sync(self, request: FrameProcessingRequest) -> DetectionResult:
        """Synchronously process a single frame with optimizations"""
        start_time = time.time()
        
        try:
            # Decode frame from bytes
            nparr = np.frombuffer(request.frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.error(f"❌ Failed to decode frame from camera {request.camera_id}")
                return DetectionResult(
                    camera_id=request.camera_id,
                    session_id=request.session_id,
                    detected_target=False,
                    non_target_count=0,
                    processing_time_ms=0,
                    frame_processed=False,
                    timestamp=time.time()
                )
            
            # Ensure frame is contiguous for better performance
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            # Clear GPU cache periodically to prevent memory buildup
            if torch.cuda.is_available() and self.processing_stats['frames_processed'] % 50 == 0:
                torch.cuda.empty_cache()
            
            # Perform detection and get frame with overlays
            logger.debug(f"🔍 Processing frame for camera {request.camera_id}, target: '{request.target_label}'")
            
            detection_results = self.detection_system.detect_and_contour(frame, request.target_label)
            
            # Handle different return types from detection
            processed_frame = None
            detected_target = False
            non_target_count = 0
            
            if isinstance(detection_results, tuple):
                processed_frame = detection_results[0]  # Frame with overlays
                detected_target = detection_results[1] if len(detection_results) > 1 else False
                non_target_count = detection_results[2] if len(detection_results) > 2 else 0
            else:
                # If single return value, it's the processed frame
                processed_frame = detection_results
                detected_target = False
            
            processing_time = (time.time() - start_time) * 1000
            
            # ENCODE THE PROCESSED FRAME (with detection overlays)
            processed_frame_data = None
            overlay_created = False
            
            if processed_frame is not None:
                try:
                    # Always encode the processed frame
                    success, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                    if success:
                        processed_frame_data = buffer.tobytes()
                        overlay_created = True
                        self.processing_stats['overlays_created'] += 1
                        
                        logger.info(f"✅ Created overlay frame for camera {request.camera_id}, size: {len(processed_frame_data)} bytes, target detected: {detected_target}")
                    else:
                        logger.error(f"❌ Failed to encode processed frame for camera {request.camera_id}")
                except Exception as e:
                    logger.error(f"❌ Error encoding processed frame: {e}")
            else:
                logger.warning(f"⚠️ No processed frame returned from detection for camera {request.camera_id}")
            
            # Update stats
            self.processing_stats['frames_processed'] += 1
            self.processing_stats['total_processing_time'] += processing_time
            
            result = DetectionResult(
                camera_id=request.camera_id,
                session_id=request.session_id,
                detected_target=detected_target,
                non_target_count=non_target_count,
                processing_time_ms=processing_time,
                timestamp=time.time(),
                processed_frame_data=processed_frame_data,
                frame_processed=overlay_created
            )
            
            if detected_target:
                logger.info(f"🎯 TARGET DETECTED: '{request.target_label}' found in camera {request.camera_id}!")
            
            if overlay_created:
                logger.debug(f"🖼️ Overlay frame created for camera {request.camera_id}")
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Error processing frame from camera {request.camera_id}: {e}")
            return DetectionResult(
                camera_id=request.camera_id,
                session_id=request.session_id,
                detected_target=False,
                non_target_count=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                frame_processed=False,
                timestamp=time.time()
            )
    
    def _wait_for_subscribers_sync(self, channel: str, timeout: float = 5.0) -> int:
        """FIXED: Synchronous method to wait for subscribers using sync Redis client"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Use sync Redis client for pubsub_numsub
                subscribers_result = self.sync_redis_client.pubsub_numsub(channel)
                
                if subscribers_result and len(subscribers_result) > 0:
                    subscriber_count = subscribers_result[0][1] if len(subscribers_result[0]) > 1 else 0
                    
                    if subscriber_count > 0:
                        logger.info(f"✅ Found {subscriber_count} subscribers on {channel}")
                        return subscriber_count
                
                # Wait a bit before checking again
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"❌ Error checking subscribers on {channel}: {e}")
                time.sleep(0.5)
        
        logger.warning(f"⚠️ No subscribers found on {channel} after {timeout}s timeout")
        return 0

    async def _send_result_to_redis(self, result: DetectionResult):
        """ENHANCED: Result sending with subscriber verification and retry logic"""
        try:
            if not self.redis_client or not self.sync_redis_client:
                logger.error("❌ Redis clients not initialized")
                return
            
            # Create result dict for serialization
            result_dict = {
                'camera_id': result.camera_id,
                'session_id': result.session_id,
                'detected_target': result.detected_target,
                'non_target_count': result.non_target_count,
                'processing_time_ms': result.processing_time_ms,
                'timestamp': result.timestamp,
                'confidence': result.confidence,
                'bounding_boxes': result.bounding_boxes,
                'frame_processed': result.frame_processed
            }
            
            serialized_result = pickle.dumps(result_dict)
            
            # Store processed frame first (if available)
            if result.processed_frame_data:
                processed_frame_key = f"processed_frame:{result.camera_id}:{result.session_id}"
                await self.redis_client.setex(processed_frame_key, 30, result.processed_frame_data)
                self.processing_stats['frames_stored'] += 1
                
                logger.info(f"📦 STORED processed frame in Redis: {processed_frame_key}, size: {len(result.processed_frame_data)} bytes")
            
            # Store result with TTL
            result_key = f"detection_result:{result.camera_id}:{result.session_id}"
            await self.redis_client.setex(result_key, 30, serialized_result)
            
            # CRITICAL FIX: Check for subscribers before publishing
            channel = f"detection_results:{result.camera_id}"
            
            # FIXED: Use sync method to check subscribers
            subscriber_count = await asyncio.get_event_loop().run_in_executor(
                None, 
                self._wait_for_subscribers_sync, 
                channel, 
                3.0
            )
            
            total_subscribers = 0
            if subscriber_count > 0:
                # Publish with enhanced retry mechanism
                published_count = await self._publish_with_retry(channel, serialized_result, max_retries=3)
                total_subscribers += published_count
                
                if published_count > 0:
                    logger.info(f"📡 PUBLISHED detection result to {channel}: {published_count} subscribers received")
                else:
                    logger.warning(f"⚠️ Failed to publish to {channel} despite {subscriber_count} subscribers")
            else:
                logger.warning(f"⚠️ Skipping publish to {channel} - no subscribers available")
            
            # Update stats
            self.processing_stats['pubsub_messages_sent'] += 1
            self.processing_stats['detection_results_published'] += 1
            
            # Log success with subscriber count
            if total_subscribers > 0:
                logger.info(f"✅ Detection result delivered to {total_subscribers} total subscribers for camera {result.camera_id}")
            else:
                logger.warning(f"⚠️ Detection result processed and stored but no active subscribers for camera {result.camera_id}")
                
                # DIAGNOSTIC: Log detailed info when no subscribers
                logger.info(f"🔍 DIAGNOSTIC - Camera {result.camera_id}: "
                        f"frame_stored={result.processed_frame_data is not None}, "
                        f"result_stored=True, "
                        f"target_detected={result.detected_target}")
            
            # DIAGNOSTIC: Log pubsub stats periodically
            if self.processing_stats['detection_results_published'] % 5 == 0:
                logger.info(f"📊 Pubsub Stats: {self.processing_stats['detection_results_published']} results published, "
                        f"{self.processing_stats['pubsub_messages_sent']} messages sent")
            
        except Exception as e:
            logger.error(f"❌ Failed to send result to Redis: {e}")

    async def _publish_with_retry(self, channel: str, message: bytes, max_retries: int = 3) -> int:
        """Publish message with retry logic and return subscriber count"""
        for attempt in range(max_retries):
            try:
                # Publish with timeout
                subscribers = await asyncio.wait_for(
                    self.redis_client.publish(channel, message),
                    timeout=3.0
                )
                
                if subscribers > 0:
                    return subscribers
                else:
                    logger.warning(f"⚠️ Published to {channel} but 0 subscribers received (attempt {attempt + 1})")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.5)  # Brief delay before retry
                        
            except asyncio.TimeoutError:
                logger.warning(f"⚠️ Publish timeout for {channel}, attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"❌ Error publishing to {channel} on attempt {attempt + 1}: {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)
        
        logger.error(f"❌ Failed to publish to {channel} after {max_retries} attempts")
        return 0

    async def check_subscribers_for_camera(self, camera_id: int) -> Dict[str, Any]:
        """Check subscriber status for a specific camera"""
        try:
            channel = f"detection_results:{camera_id}"
            
            # FIXED: Use sync Redis client to check subscribers
            result = self.sync_redis_client.pubsub_numsub(channel)
            subscriber_count = result[0][1] if result and len(result) > 0 and len(result[0]) > 1 else 0
            
            # Test publish
            test_message = pickle.dumps({
                'camera_id': camera_id,
                'test': True,
                'timestamp': time.time()
            })
            
            published_count = await self.redis_client.publish(channel, test_message)
            
            return {
                'camera_id': camera_id,
                'channel': channel,
                'subscriber_count': subscriber_count,
                'test_published_to': published_count,
                'redis_connected': True,
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'camera_id': camera_id,
                'error': str(e),
                'status': 'error'
            }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics with safe key access"""
        with self.queue_lock:
            queue_depth = len(self.high_priority_queue) + len(self.normal_priority_queue)
        
        # Safe access to processing stats with defaults
        frames_processed = self.processing_stats.get('frames_processed', 0)
        total_processing_time = self.processing_stats.get('total_processing_time', 0)
        
        avg_processing_time = 0
        if frames_processed > 0:
            avg_processing_time = total_processing_time / frames_processed
        
        # Memory stats with error handling
        memory_usage_mb = 0
        memory_percent = 0
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            memory_usage_mb = memory_info.rss / 1024 / 1024
            memory_percent = process.memory_percent()
        except Exception as e:
            logger.warning(f"⚠️ Could not get memory stats: {e}")
        
        return {
            'queue_depth': queue_depth,
            'frames_processed': frames_processed,
            'overlays_created': self.processing_stats.get('overlays_created', 0),
            'frames_stored': self.processing_stats.get('frames_stored', 0),
            'pubsub_messages_sent': self.processing_stats.get('pubsub_messages_sent', 0),
            'detection_results_published': self.processing_stats.get('detection_results_published', 0),
            'avg_processing_time_ms': round(avg_processing_time, 2),
            'queue_overflows': self.processing_stats.get('queue_overflows', 0),
            'timeouts': self.processing_stats.get('timeouts', 0),
            'memory_usage_mb': round(memory_usage_mb, 2),
            'memory_percent': round(memory_percent, 2),
            'device': str(self.device) if self.device else "unknown",
            'is_running': self.is_running,
            'redis_connected': self.redis_client is not None,
            'sync_redis_connected': self.sync_redis_client is not None,
            'result_queue_size': self._result_queue.qsize() if self._result_queue else 0
        }
    def reset_processing_stats(self):
        """Reset all processing statistics to initial state"""
        self.processing_stats = {
            'frames_processed': 0,
            'total_processing_time': 0,
            'queue_overflows': 0,
            'timeouts': 0,
            'overlays_created': 0,
            'frames_stored': 0,
            'pubsub_messages_sent': 0,
            'detection_results_published': 0
        }
        logger.info("📊 Processing statistics reset to initial state")

    async def shutdown(self):
        """Graceful shutdown with proper cleanup"""
        logger.info("🛑 Shutting down detection processor...")
        self.is_running = False
        
        # Cancel all async tasks
        tasks_to_cancel = [
            self.redis_listener_task,
            self._result_sender_task
        ]
        
        for task in tasks_to_cancel:
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    logger.info(f"🛑 Task cancelled/timed out during shutdown")
        
        # Wait for processing thread to finish
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5)
            if self.processor_thread.is_alive():
                logger.warning("⚠️ Processing thread did not stop gracefully")
        
        # Shutdown executor
        if self.executor:
            self.executor.shutdown(wait=True)
            self.executor = None
        
        # Close Redis connections
        if self.redis_client:
            try:
                await self.redis_client.aclose()
            except Exception as e:
                logger.warning(f"⚠️ Error closing async Redis connection: {e}")
            finally:
                self.redis_client = None
        
        if self.sync_redis_client:
            try:
                self.sync_redis_client.close()
            except Exception as e:
                logger.warning(f"⚠️ Error closing sync Redis connection: {e}")
            finally:
                self.sync_redis_client = None
        
        # Clear queues
        with self.queue_lock:
            self.high_priority_queue.clear()
            self.normal_priority_queue.clear()
        
        # Clear result queue
        if self._result_queue:
            while not self._result_queue.empty():
                try:
                    self._result_queue.get_nowait()
                except:
                    break
        
        # ADDED: Reset processing stats on shutdown
        self.reset_processing_stats()
        
        logger.info("✅ Detection processor shutdown complete")

# Global processor instance
detection_processor = OptimizedDetectionProcessor()