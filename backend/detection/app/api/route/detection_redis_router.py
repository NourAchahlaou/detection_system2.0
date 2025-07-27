# fixed_detection_redis_router.py
import asyncio
import logging
from datetime import datetime
import pickle
import time
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

# Import your FIXED detection processor
from detection.app.service.optimized_detection_service import detection_processor

logger = logging.getLogger(__name__)

redis_router = APIRouter(    
    prefix="/redis",
    tags=["Detection with redis"],
    responses={404: {"description": "Not found"}},
)

# Track initialization state
_processor_initialized = False
_initialization_lock = asyncio.Lock()

@redis_router.post("/initialize")
async def initialize_processor():
    """Initialize the detection processor"""
    global _processor_initialized
    
    async with _initialization_lock:
        if _processor_initialized:
            return {"message": "Detection processor already initialized", "status": "already_running"}
        
        try:
            # This is now properly async and won't block
            await detection_processor.initialize()
            _processor_initialized = True
            logger.info("Detection processor initialized successfully via API")
            return {
                "message": "Detection processor initialized successfully",
                "status": "initialized",
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Failed to initialize detection processor: {e}")
            # Reset flag on failure
            _processor_initialized = False
            raise HTTPException(
                status_code=500, 
                detail=f"Failed to initialize detection processor: {str(e)}"
            )

@redis_router.post("/shutdown")
async def shutdown_processor():
    """Shutdown the detection processor - now properly async"""
    global _processor_initialized
    
    if not _processor_initialized:
        return {"message": "Detection processor not running", "status": "not_running"}
    
    try:
        # Direct async shutdown - no background task needed
        await detection_processor.shutdown()
        _processor_initialized = False
        logger.info("Detection processor shutdown completed via API")
        
        return {
            "message": "Detection processor shutdown completed",
            "status": "shutdown",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")
        # Even if shutdown fails, mark as not initialized to allow re-init
        _processor_initialized = False
        raise HTTPException(
            status_code=500,
            detail=f"Shutdown failed: {str(e)}"
        )

@redis_router.get("/status")
async def get_processor_status():
    """Get the current status of the detection processor"""
    global _processor_initialized
    
    if not _processor_initialized:
        return {
            "status": "not_initialized",
            "message": "Detection processor is not initialized",
            "initialized": False,
            "timestamp": datetime.now().isoformat()
        }
    
    try:
        stats = detection_processor.get_performance_stats()
        return {
            "status": "running" if stats.get('is_running', False) else "stopped",
            "initialized": _processor_initialized,
            "stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting processor status: {e}")
        return {
            "status": "error",
            "initialized": _processor_initialized,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@redis_router.get("/stats")
async def get_stats():
    """Get performance statistics"""
    if not _processor_initialized:
        raise HTTPException(
            status_code=503, 
            detail="Detection processor not initialized. Call /redis/initialize first."
        )
    
    try:
        return detection_processor.get_performance_stats()
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Error getting stats: {str(e)}")

@redis_router.get("/health")
async def health_check():
    """Enhanced health check"""
    global _processor_initialized
    
    if not _processor_initialized:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "message": "Detection processor not initialized",
                "initialized": False,
                "timestamp": datetime.now().isoformat()
            }
        )
    
    try:
        stats = detection_processor.get_performance_stats()
        is_healthy = stats.get('is_running', False)
        
        return JSONResponse(
            status_code=200 if is_healthy else 503,
            content={
                "status": "healthy" if is_healthy else "unhealthy",
                "initialized": _processor_initialized,
                "stats": stats,
                "timestamp": datetime.now().isoformat()
            }
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "initialized": _processor_initialized,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# Force shutdown endpoint for emergencies
@redis_router.post("/force-shutdown")
async def force_shutdown():
    """Force shutdown without waiting - enhanced error handling"""
    global _processor_initialized
    
    try:
        # Set flag to stop processes
        detection_processor.is_running = False
        
        # Cancel tasks if they exist
        if hasattr(detection_processor, 'redis_listener_task') and detection_processor.redis_listener_task:
            detection_processor.redis_listener_task.cancel()
        
        # Force close Redis connection if it exists
        if hasattr(detection_processor, 'redis_client') and detection_processor.redis_client:
            try:
                await detection_processor.redis_client.aclose()
            except:
                pass  # Ignore errors during force shutdown
        
        # Shutdown executor if it exists
        if hasattr(detection_processor, 'executor') and detection_processor.executor:
            try:
                detection_processor.executor.shutdown(wait=False)
            except:
                pass  # Ignore errors during force shutdown
        
        _processor_initialized = False
        
        return {
            "message": "Force shutdown initiated",
            "status": "force_shutdown",
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Force shutdown error: {e}")
        # Even if force shutdown fails, reset the flag
        _processor_initialized = False
        return {
            "message": f"Force shutdown completed with errors: {str(e)}",
            "status": "force_shutdown_with_errors",
            "timestamp": datetime.now().isoformat()
        }

@redis_router.post("/auto-start")
async def auto_start():
    """Auto-start the detection processor if not already running"""
    global _processor_initialized
    
    if _processor_initialized:
        try:
            stats = detection_processor.get_performance_stats()
            if stats.get('is_running', False):
                return {
                    "message": "Detection processor already running",
                    "status": "already_running",
                    "stats": stats,
                    "timestamp": datetime.now().isoformat()
                }
        except Exception:
            # If we can't get stats but flag says initialized, try to reinitialize
            logger.warning("Processor marked as initialized but can't get stats - reinitializing")
            _processor_initialized = False
    
    # Initialize if not already done
    return await initialize_processor()

@redis_router.post("/reset")
async def reset_processor():
    """Reset the processor completely - force shutdown and clear state"""
    global _processor_initialized
    
    try:
        # Force shutdown first
        await force_shutdown()
        
        # Clear any remaining state
        detection_processor.processing_stats = {
            'frames_processed': 0,
            'total_processing_time': 0,
            'queue_overflows': 0,
            'timeouts': 0
        }
        
        # Clear queues
        if hasattr(detection_processor, 'high_priority_queue'):
            detection_processor.high_priority_queue.clear()
        if hasattr(detection_processor, 'normal_priority_queue'):
            detection_processor.normal_priority_queue.clear()
        
        return {
            "message": "Processor reset completed",
            "status": "reset",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Reset error: {e}")
        return {
            "message": f"Reset completed with errors: {str(e)}",
            "status": "reset_with_errors",
            "timestamp": datetime.now().isoformat()
        }
# Add this to your detection router/API file
@redis_router.get("/pubsub/diagnostic/{camera_id}")
async def diagnostic_pubsub(camera_id: int):
    """Diagnostic endpoint to test pubsub communication"""
    try:
        import redis.asyncio as redis
        
        # Create test Redis client
        redis_client = redis.Redis(
            host='redis',
            port=6379,
            db=0,
            decode_responses=False
        )
        
        # Test basic connection
        await redis_client.ping()
        
        # Test publishing
        test_message = {
            'camera_id': camera_id,
            'test': True,
            'timestamp': time.time(),
            'message': 'Diagnostic test message'
        }
        
        channel = f"detection_results:{camera_id}"
        serialized_message = pickle.dumps(test_message)
        
        subscribers = await redis_client.publish(channel, serialized_message)
        
        await redis_client.aclose()
        
        return {
            "status": "success",
            "message": f"Published test message to {channel}",
            "subscribers": subscribers,
            "camera_id": camera_id
        }
        
    except Exception as e:
        return {
            "status": "error",
            "message": str(e),
            "camera_id": camera_id
        }    

@redis_router.get("/ready")
async def detection_service_ready():
    """
    Service readiness endpoint for circuit breaker pattern.
    Returns 200 only when detection service is fully operational.
    """
    global _processor_initialized
    
    try:
        # Check 1: Processor initialized
        if not _processor_initialized:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "processor_initialized": False,
                    "redis_connected": False,
                    "pubsub_ready": False,
                    "error": "Detection processor not initialized",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Check 2: Get processor stats to verify it's running
        try:
            stats = detection_processor.get_performance_stats()
            is_running = stats.get('is_running', False)
            redis_connected = stats.get('redis_connected', False)
            sync_redis_connected = stats.get('sync_redis_connected', False)
        except Exception as e:
            logger.error(f"Error getting processor stats: {e}")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "processor_initialized": _processor_initialized,
                    "redis_connected": False,
                    "pubsub_ready": False,
                    "error": f"Cannot get processor stats: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Check 3: Redis connections
        if not redis_connected or not sync_redis_connected:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "processor_initialized": _processor_initialized,
                    "redis_connected": redis_connected,
                    "sync_redis_connected": sync_redis_connected,
                    "pubsub_ready": False,
                    "error": "Redis connections not established",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # Check 4: Pubsub functionality
        pubsub_ready = False
        pubsub_error = None
        
        try:
            # Test pubsub by checking if we can publish a test message
            test_channel = "readiness_test"
            test_message = b"readiness_check"
            
            # Use the processor's sync Redis client
            if hasattr(detection_processor, 'sync_redis_client') and detection_processor.sync_redis_client:
                subscribers = detection_processor.sync_redis_client.publish(test_channel, test_message)
                pubsub_ready = True  # Publishing worked, even if no subscribers
            else:
                pubsub_error = "Sync Redis client not available"
                
        except Exception as e:
            pubsub_error = f"Pubsub test failed: {str(e)}"
            logger.error(f"Pubsub readiness test failed: {e}")
        
        # Check 5: Processor is running
        if not is_running:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready",
                    "processor_initialized": _processor_initialized,
                    "redis_connected": redis_connected,
                    "sync_redis_connected": sync_redis_connected,
                    "pubsub_ready": pubsub_ready,
                    "processor_running": False,
                    "error": "Detection processor is not running",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
        # All checks passed
        if pubsub_ready:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "ready",
                    "processor_initialized": True,
                    "redis_connected": True,
                    "sync_redis_connected": True,
                    "pubsub_ready": True,
                    "processor_running": True,
                    "stats": {
                        "frames_processed": stats.get('frames_processed', 0),
                        "avg_processing_time_ms": stats.get('avg_processing_time_ms', 0),
                        "queue_depth": stats.get('queue_depth', 0),
                        "memory_usage_mb": stats.get('memory_usage_mb', 0),
                        "device": stats.get('device', 'unknown')
                    },
                    "timestamp": datetime.now().isoformat()
                }
            )
        else:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "not_ready", 
                    "processor_initialized": _processor_initialized,
                    "redis_connected": redis_connected,
                    "sync_redis_connected": sync_redis_connected,
                    "pubsub_ready": False,
                    "processor_running": is_running,
                    "error": pubsub_error or "Pubsub not ready",
                    "timestamp": datetime.now().isoformat()
                }
            )
        
    except Exception as e:
        logger.error(f"Error in readiness check: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "processor_initialized": _processor_initialized,
                "redis_connected": False,
                "pubsub_ready": False,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )    