# optimized_detection_service.py
import cv2
import asyncio
import logging
import redis
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
    priority: int = 1  # Higher number = higher priority

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

class OptimizedDetectionProcessor:
    """High-performance detection processor with queue management"""
    
    def __init__(self, 
                 redis_host='redis', 
                 redis_port=6379,
                 max_workers=2,
                 max_queue_size=50,
                 frame_timeout=5.0):
        
        # Redis connection with connection pooling
        self.redis_pool = redis.ConnectionPool(
            host=redis_host, 
            port=redis_port, 
            db=0,
            max_connections=10,
            retry_on_timeout=True,
            socket_keepalive=True,
            socket_keepalive_options={},
            health_check_interval=30
        )
        self.redis_client = redis.Redis(connection_pool=self.redis_pool)
        
        # Processing configuration
        self.max_workers = max_workers
        self.max_queue_size = max_queue_size
        self.frame_timeout = frame_timeout
        
        # Initialize detection system
        self.detection_system = None
        self.device = None
        
        # Performance tracking
        self.processing_stats = {
            'frames_processed': 0,
            'total_processing_time': 0,
            'queue_overflows': 0,
            'timeouts': 0
        }
        
        # Thread pool for CPU-bound operations
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        # Frame memory pool to reduce allocations
        self.frame_pool = deque(maxlen=20)
        self.pool_lock = threading.Lock()
        
        # Processing queues
        self.high_priority_queue = deque()
        self.normal_priority_queue = deque()
        self.queue_lock = threading.Lock()
        
        # Control flags
        self.is_running = False
        self.processor_thread = None
        
    async def initialize(self):
        """Initialize the detection system and start processing"""
        try:
            logger.info("Initializing detection system...")
            self.detection_system = DetectionSystem()
            self.device = self.detection_system.device
            logger.info(f"Detection system initialized on device: {self.device}")
            
            # Test Redis connection
            await self.test_redis_connection()
            
            # Start processing thread
            self.is_running = True
            self.processor_thread = threading.Thread(target=self._processing_loop, daemon=True)
            self.processor_thread.start()
            
            # Start Redis listener
            asyncio.create_task(self._redis_listener())
            
            logger.info("Optimized detection processor initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize detection processor: {e}")
            raise
    
    async def test_redis_connection(self):
        """Test Redis connection"""
        try:
            self.redis_client.ping()
            logger.info("Redis connection established successfully")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            raise
    
    def get_frame_from_pool(self) -> Optional[np.ndarray]:
        """Get a reusable frame array from the pool"""
        with self.pool_lock:
            if self.frame_pool:
                return self.frame_pool.popleft()
        return None
    
    def return_frame_to_pool(self, frame: np.ndarray):
        """Return frame array to pool for reuse"""
        if frame is not None and frame.size > 0:
            with self.pool_lock:
                if len(self.frame_pool) < 20:
                    self.frame_pool.append(frame)
    
    async def _redis_listener(self):
        """Listen for frame processing requests from Redis"""
        logger.info("Starting Redis listener for frame processing requests")
        
        while self.is_running:
            try:
                # Block for new requests with timeout
                request_data = self.redis_client.brpop(['detection_queue'], timeout=1)
                
                if request_data:
                    _, serialized_request = request_data
                    
                    try:
                        request = pickle.loads(serialized_request)
                        await self._queue_frame_for_processing(request)
                        
                    except Exception as e:
                        logger.error(f"Failed to deserialize frame request: {e}")
                        
            except redis.exceptions.ConnectionError as e:
                logger.error(f"Redis connection error: {e}")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"Error in Redis listener: {e}")
                await asyncio.sleep(0.1)
    
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
                logger.debug(f"Dropping expired frame from camera {request.camera_id}")
                self.processing_stats['timeouts'] += 1
                return
            
            # Add to appropriate queue
            with self.queue_lock:
                total_queue_size = len(self.high_priority_queue) + len(self.normal_priority_queue)
                
                if total_queue_size >= self.max_queue_size:
                    # Drop oldest normal priority frame
                    if self.normal_priority_queue:
                        dropped = self.normal_priority_queue.popleft()
                        logger.debug(f"Queue overflow: dropped frame from camera {dropped.camera_id}")
                        self.processing_stats['queue_overflows'] += 1
                
                if request.priority > 1:
                    self.high_priority_queue.append(request)
                else:
                    self.normal_priority_queue.append(request)
                    
        except Exception as e:
            logger.error(f"Error queuing frame for processing: {e}")
    
    def _processing_loop(self):
        """Main processing loop running in separate thread"""
        logger.info("Starting frame processing loop")
        
        while self.is_running:
            try:
                request = self._get_next_request()
                
                if request:
                    # Process frame
                    result = self._process_frame_sync(request)
                    
                    # Send result back via Redis
                    self._send_result_to_redis(result)
                    
                else:
                    # No requests available, short sleep
                    time.sleep(0.01)
                    
            except Exception as e:
                logger.error(f"Error in processing loop: {e}")
                time.sleep(0.1)
    
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
        """Synchronously process a single frame"""
        start_time = time.time()
        
        try:
            # Decode frame from bytes
            nparr = np.frombuffer(request.frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.error(f"Failed to decode frame from camera {request.camera_id}")
                return DetectionResult(
                    camera_id=request.camera_id,
                    session_id=request.session_id,
                    detected_target=False,
                    non_target_count=0,
                    processing_time_ms=0,
                    frame_processed=False,
                    timestamp=time.time()
                )
            
            # Ensure frame is contiguous
            if not frame.flags['C_CONTIGUOUS']:
                frame = np.ascontiguousarray(frame)
            
            # Perform detection
            detection_results = self.detection_system.detect_and_contour(frame, request.target_label)
            
            if isinstance(detection_results, tuple):
                processed_frame = detection_results[0]
                detected_target = detection_results[1] if len(detection_results) > 1 else False
                non_target_count = detection_results[2] if len(detection_results) > 2 else 0
            else:
                processed_frame = detection_results
                detected_target = False
                non_target_count = 0
            
            # Encode processed frame back to bytes for caching
            success, buffer = cv2.imencode('.jpg', processed_frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                # Cache processed frame in Redis with TTL
                cache_key = f"processed_frame:{request.camera_id}:{request.session_id}"
                self.redis_client.setex(cache_key, 2, buffer.tobytes())  # 2 second TTL
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update stats
            self.processing_stats['frames_processed'] += 1
            self.processing_stats['total_processing_time'] += processing_time
            
            return DetectionResult(
                camera_id=request.camera_id,
                session_id=request.session_id,
                detected_target=detected_target,
                non_target_count=non_target_count,
                processing_time_ms=processing_time,
                timestamp=time.time()
            )
            
        except Exception as e:
            logger.error(f"Error processing frame from camera {request.camera_id}: {e}")
            return DetectionResult(
                camera_id=request.camera_id,
                session_id=request.session_id,
                detected_target=False,
                non_target_count=0,
                processing_time_ms=(time.time() - start_time) * 1000,
                frame_processed=False,
                timestamp=time.time()
            )
    
    def _send_result_to_redis(self, result: DetectionResult):
        """Send detection result back via Redis"""
        try:
            result_key = f"detection_result:{result.camera_id}:{result.session_id}"
            serialized_result = pickle.dumps(result)
            
            # Store result with TTL
            self.redis_client.setex(result_key, 5, serialized_result)  # 5 second TTL
            
            # Also publish to channel for real-time subscribers
            channel = f"detection_results:{result.camera_id}"
            self.redis_client.publish(channel, serialized_result)
            
        except Exception as e:
            logger.error(f"Failed to send result to Redis: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        with self.queue_lock:
            queue_depth = len(self.high_priority_queue) + len(self.normal_priority_queue)
        
        frames_processed = self.processing_stats['frames_processed']
        avg_processing_time = 0
        if frames_processed > 0:
            avg_processing_time = self.processing_stats['total_processing_time'] / frames_processed
        
        return {
            'queue_depth': queue_depth,
            'frames_processed': frames_processed,
            'avg_processing_time_ms': round(avg_processing_time, 2),
            'queue_overflows': self.processing_stats['queue_overflows'],
            'timeouts': self.processing_stats['timeouts'],
            'memory_usage_mb': psutil.Process().memory_info().rss / 1024 / 1024,
            'device': str(self.device),
            'is_running': self.is_running
        }
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down detection processor...")
        self.is_running = False
        
        if self.processor_thread and self.processor_thread.is_alive():
            self.processor_thread.join(timeout=5)
        
        self.executor.shutdown(wait=True)
        
        if self.redis_client:
            self.redis_client.close()
        
        logger.info("Detection processor shutdown complete")

# Global processor instance
detection_processor = OptimizedDetectionProcessor()

# FastAPI endpoints (simplified)
from fastapi import APIRouter

redis_router = APIRouter(    
    prefix="/redis",
    tags=["Detection with redis"],
    responses={404: {"description": "Not found"}},
)

@redis_router.on_event("startup")
async def startup():
    """Initialize detection processor on startup"""
    await detection_processor.initialize()

@redis_router.on_event("shutdown") 
async def shutdown():
    """Cleanup on shutdown"""
    await detection_processor.shutdown()

@redis_router.get("/stats")
async def get_stats():
    """Get performance statistics"""
    return detection_processor.get_performance_stats()

@redis_router.get("/health")
async def health_check():
    """Enhanced health check"""
    stats = detection_processor.get_performance_stats()
    return {
        "status": "healthy" if stats['is_running'] else "unhealthy",
        "stats": stats,
        "timestamp": datetime.now().isoformat()
    }