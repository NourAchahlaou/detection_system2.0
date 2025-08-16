# fixed_detection_redis_router.py - FIXED VERSION
import asyncio
import logging
from datetime import datetime
import pickle
import time
from typing import Dict, Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse

# Import your FIXED detection processor
from detection.app.service.detection.optimized_detection_service import detection_processor

logger = logging.getLogger(__name__)

redis_router = APIRouter(    
    prefix="/redis",
    tags=["Detection with redis"],
    responses={404: {"description": "Not found"}},
)

# Track initialization state - SHARED with shutdown endpoints
_processor_initialized = False
_initialization_lock = asyncio.Lock()

@redis_router.post("/initialize")
async def initialize_processor():
    """FIXED: Initialize the detection processor with proper state management"""
    global _processor_initialized
    
    async with _initialization_lock:
        if _processor_initialized:
            # Verify it's actually running
            try:
                stats = detection_processor.get_performance_stats()
                if stats.get('is_running', False):
                    return {
                        "message": "Detection processor already initialized and running", 
                        "status": "already_running"
                    }
                else:
                    # Marked as initialized but not running - reset and reinitialize
                    logger.warning("⚠️ Processor marked as initialized but not running - resetting...")
                    _processor_initialized = False
            except Exception as e:
                logger.warning(f"⚠️ Could not verify processor status - resetting: {e}")
                _processor_initialized = False
        
        if not _processor_initialized:
            try:
                logger.info("🚀 Initializing detection processor...")
                
                # FIXED: Ensure processor is in clean state before initialization
                await _ensure_clean_processor_state()
                
                # This is now properly async and won't block
                await detection_processor.initialize()
                _processor_initialized = True
                
                # Verify initialization was successful
                stats = detection_processor.get_performance_stats()
                if not stats.get('is_running', False):
                    raise Exception("Processor initialized but not running")
                
                logger.info("✅ Detection processor initialized successfully via API")
                return {
                    "message": "Detection processor initialized successfully",
                    "status": "initialized",
                    "timestamp": datetime.now().isoformat()
                }
            except Exception as e:
                logger.error(f"❌ Failed to initialize detection processor: {e}")
                # Reset flag on failure
                _processor_initialized = False
                raise HTTPException(
                    status_code=500, 
                    detail=f"Failed to initialize detection processor: {str(e)}"
                )

async def _ensure_clean_processor_state():
    """FIXED: Ensure processor is in clean state before initialization"""
    try:
        # Stop any running processes
        detection_processor.is_running = False
        
        # Cancel any existing tasks
        if hasattr(detection_processor, 'redis_listener_task') and detection_processor.redis_listener_task:
            detection_processor.redis_listener_task.cancel()
            try:
                await detection_processor.redis_listener_task
            except asyncio.CancelledError:
                pass
        
        if hasattr(detection_processor, '_result_sender_task') and detection_processor._result_sender_task:
            detection_processor._result_sender_task.cancel()
            try:
                await detection_processor._result_sender_task
            except asyncio.CancelledError:
                pass
        
        # Close existing connections
        if hasattr(detection_processor, 'redis_client') and detection_processor.redis_client:
            try:
                await detection_processor.redis_client.aclose()
            except:
                pass
            detection_processor.redis_client = None
        
        if hasattr(detection_processor, 'sync_redis_client') and detection_processor.sync_redis_client:
            try:
                detection_processor.sync_redis_client.close()
            except:
                pass
            detection_processor.sync_redis_client = None
        
        # Shutdown executor
        if hasattr(detection_processor, 'executor') and detection_processor.executor:
            try:
                detection_processor.executor.shutdown(wait=False)
            except:
                pass
            detection_processor.executor = None
        
        # Clear queues
        with detection_processor.queue_lock:
            detection_processor.high_priority_queue.clear()
            detection_processor.normal_priority_queue.clear()
        
        # Reset other attributes
        detection_processor.detection_system = None
        detection_processor.device = None
        detection_processor.processor_thread = None
        detection_processor.redis_listener_task = None
        detection_processor._main_loop = None
        detection_processor._result_queue = None
        detection_processor._result_sender_task = None
        
        # Reset stats
        detection_processor.processing_stats = {
            'frames_processed': 0,
            'total_processing_time': 0,
            'queue_overflows': 0,
            'timeouts': 0,
            'overlays_created': 0,
            'frames_stored': 0,
            'pubsub_messages_sent': 0,
            'detection_results_published': 0
        }
        
        logger.info("🔄 Processor state cleaned for fresh initialization")
        
    except Exception as e:
        logger.warning(f"⚠️ Error cleaning processor state: {e}")

@redis_router.get("/status")
async def get_processor_status():
    """FIXED: Get the current status of the detection processor"""
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
        is_running = stats.get('is_running', False)
        
        # FIXED: If marked as initialized but not running, there's an inconsistency
        if not is_running:
            logger.warning("⚠️ Processor marked as initialized but not running")
            # Don't reset here, let the user call reset or re-initialize
        
        return {
            "status": "running" if is_running else "stopped",
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
    """FIXED: Enhanced health check with better state verification"""
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
        is_running = stats.get('is_running', False)
        redis_connected = stats.get('redis_connected', False)
        sync_redis_connected = stats.get('sync_redis_connected', False)
        
        # FIXED: More comprehensive health check
        is_healthy = (
            is_running and 
            redis_connected and 
            sync_redis_connected and
            hasattr(detection_processor, 'detection_system') and
            detection_processor.detection_system is not None
        )
        
        health_details = {
            "is_running": is_running,
            "redis_connected": redis_connected,
            "sync_redis_connected": sync_redis_connected,
            "detection_system_loaded": detection_processor.detection_system is not None,
            "device": stats.get('device', 'unknown')
        }
        
        return JSONResponse(
            status_code=200 if is_healthy else 503,
            content={
                "status": "healthy" if is_healthy else "unhealthy",
                "initialized": _processor_initialized,
                "details": health_details,
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

@redis_router.post("/force-shutdown")
async def force_shutdown():
    """FIXED: Force shutdown with proper state reset"""
    global _processor_initialized
    
    try:
        logger.info("🛑 Force shutdown initiated...")
        
        # Use the clean state function
        await _ensure_clean_processor_state()
        
        # Reset global state
        async with _initialization_lock:
            _processor_initialized = False
        
        logger.info("✅ Force shutdown completed")
        
        return {
            "message": "Force shutdown completed successfully",
            "status": "force_shutdown",
            "processor_initialized": False,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Force shutdown error: {e}")
        # Even if force shutdown fails, reset the flag
        async with _initialization_lock:
            _processor_initialized = False
        return {
            "message": f"Force shutdown completed with errors: {str(e)}",
            "status": "force_shutdown_with_errors",
            "processor_initialized": False,
            "timestamp": datetime.now().isoformat()
        }

@redis_router.post("/auto-start")
async def auto_start():
    """FIXED: Auto-start with better state checking"""
    global _processor_initialized
    
    # Check current state first
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
            else:
                # Initialized but not running - need to reinitialize
                logger.info("🔄 Processor initialized but not running - reinitializing...")
                async with _initialization_lock:
                    _processor_initialized = False
        except Exception:
            # If we can't get stats, assume we need to reinitialize
            logger.warning("⚠️ Cannot get processor stats - reinitializing...")
            async with _initialization_lock:
                _processor_initialized = False
    
    # Initialize if not already done or if reinitialize needed
    return await initialize_processor()

@redis_router.post("/reset")
async def reset_processor():
    """FIXED: Reset the processor completely with proper cleanup"""
    global _processor_initialized
    
    try:
        logger.info("🔄 Resetting processor completely...")
        
        # Use the enhanced clean state function
        await _ensure_clean_processor_state()
        
        # Reset global state  
        async with _initialization_lock:
            _processor_initialized = False
        
        logger.info("✅ Detection processor completely reset")
        
        return {
            "message": "Processor reset completed successfully",
            "status": "reset",
            "processor_initialized": False,
            "stats_reset": True,
            "queues_cleared": True,
            "connections_closed": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Reset error: {e}")
        # Even on error, ensure we reset the global state
        async with _initialization_lock:
            _processor_initialized = False
        
        return {
            "message": f"Reset completed with errors: {str(e)}",
            "status": "reset_with_errors",
            "processor_initialized": False,
            "stats_reset": True,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

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
    FIXED: Service readiness endpoint with improved state checking
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
        
        # Check 3: All components running
        all_ready = (
            is_running and 
            redis_connected and 
            sync_redis_connected and
            hasattr(detection_processor, 'detection_system') and
            detection_processor.detection_system is not None
        )
        
        if all_ready:
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
                    "detection_system_loaded": detection_processor.detection_system is not None,
                    "error": "One or more components not ready",
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