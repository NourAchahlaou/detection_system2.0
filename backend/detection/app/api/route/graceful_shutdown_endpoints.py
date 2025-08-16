# graceful_shutdown_endpoints.py - FIXED VERSION
import asyncio
import logging
from datetime import datetime

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

detection_shutdown_router = APIRouter(
    prefix="/detection/shutdown",
    tags=["Detection Service Shutdown"],
    responses={404: {"description": "Not found"}},
)

# Import the shared state from the redis router
from detection.app.api.route.detection_redis_router import _processor_initialized, _initialization_lock

@detection_shutdown_router.post("/graceful")
async def graceful_shutdown_detection():
    """
    FIXED: Complete shutdown of all detection components with proper cleanup
    """
    global _processor_initialized
    
    try:
        from detection.app.service.detection.optimized_detection_service import detection_processor
        
        logger.info("🛑 Initiating COMPLETE detection system shutdown...")
        
        shutdown_results = {
            "detection_service": {"status": "not_attempted"},
            "redis_processor": {"status": "not_attempted"}, 
            "initialization_reset": {"status": "not_attempted"}
        }
        
        # Step 1: Graceful shutdown of detection service
        logger.info("📍 Step 1: Gracefully shutting down detection service...")
        try:
            if hasattr(detection_processor, 'is_running') and detection_processor.is_running:
                # Get pre-shutdown stats
                try:
                    pre_stats = detection_processor.get_performance_stats()
                    frames_in_queue = pre_stats.get('queue_depth', 0)
                    frames_processed = pre_stats.get('frames_processed', 0)
                except:
                    frames_in_queue = 0
                    frames_processed = 0
                
                logger.info(f"📊 Pre-shutdown: {frames_processed} processed, {frames_in_queue} in queue")
                
                # Stop accepting new requests
                detection_processor.is_running = False
                
                # Wait for queue to drain (with timeout)
                if frames_in_queue > 0:
                    logger.info(f"⏳ Waiting for {frames_in_queue} frames to complete...")
                    max_wait = 30
                    waited = 0
                    
                    while waited < max_wait:
                        try:
                            current_stats = detection_processor.get_performance_stats()
                            current_queue = current_stats.get('queue_depth', 0)
                            if current_queue == 0:
                                break
                            await asyncio.sleep(1)
                            waited += 1
                        except:
                            break
                    
                    if waited >= max_wait:
                        logger.warning(f"⚠️ Queue drain timeout, continuing shutdown")
                
                shutdown_results["detection_service"] = {
                    "status": "completed",
                    "frames_processed": frames_processed,
                    "frames_in_queue": frames_in_queue,
                    "wait_time_seconds": waited if 'waited' in locals() else 0
                }
            else:
                shutdown_results["detection_service"] = {
                    "status": "already_stopped",
                    "message": "Detection service was not running"
                }
                
        except Exception as e:
            logger.error(f"Error in detection service shutdown: {e}")
            shutdown_results["detection_service"] = {
                "status": "error",
                "error": str(e)
            }
        
        # Step 2: CRITICAL FIX - Actually call processor shutdown
        logger.info("📍 Step 2: Shutting down Redis processor...")
        try:
            async with _initialization_lock:
                if _processor_initialized:
                    # FIXED: Actually call the processor's shutdown method
                    logger.info("🔧 Calling detection_processor.shutdown()...")
                    await detection_processor.shutdown()
                    
                    # FIXED: Reset the global state after successful shutdown
                    _processor_initialized = False
                    
                    shutdown_results["redis_processor"] = {
                        "status": "completed",
                        "message": "Redis processor shutdown completed successfully"
                    }
                    logger.info("✅ Detection processor shutdown completed")
                else:
                    shutdown_results["redis_processor"] = {
                        "status": "already_stopped", 
                        "message": "Redis processor was not initialized"
                    }
        except Exception as e:
            logger.error(f"Error in Redis processor shutdown: {e}")
            shutdown_results["redis_processor"] = {
                "status": "error",
                "error": str(e)
            }
            # FIXED: Force reset state even on error
            async with _initialization_lock:
                _processor_initialized = False
        
        # Step 3: ENHANCED cleanup and state reset
        logger.info("📍 Step 3: Resetting initialization state and cleanup...")
        try:
            # FIXED: Ensure processor is completely reset
            try:
                # Reset processor attributes to initial state
                detection_processor.detection_system = None
                detection_processor.device = None
                detection_processor.redis_client = None
                detection_processor.sync_redis_client = None
                detection_processor.executor = None
                detection_processor.is_running = False
                detection_processor.processor_thread = None
                detection_processor.redis_listener_task = None
                detection_processor._main_loop = None
                detection_processor._result_queue = None
                detection_processor._result_sender_task = None
                
                # Clear queues safely
                with detection_processor.queue_lock:
                    detection_processor.high_priority_queue.clear()
                    detection_processor.normal_priority_queue.clear()
                
                # Reset stats to initial state
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
                
                logger.info("🔄 Detection processor completely reset to initial state")
                
            except Exception as reset_error:
                logger.error(f"Error during processor reset: {reset_error}")
                # Continue anyway
            
            # Final state verification
            async with _initialization_lock:
                _processor_initialized = False
            
            shutdown_results["initialization_reset"] = {
                "status": "completed",
                "message": "Initialization state and resources cleaned up successfully"
            }
            
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
            shutdown_results["initialization_reset"] = {
                "status": "error",
                "error": str(e)
            }
            # Force reset anyway
            async with _initialization_lock:
                _processor_initialized = False
        
        # Determine overall status
        all_completed = all(
            result["status"] in ["completed", "already_stopped"] 
            for result in shutdown_results.values()
        )
        
        logger.info("✅ Complete detection system shutdown finished")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "shutdown_complete" if all_completed else "shutdown_partial",
                "message": "Complete detection system shutdown finished",
                "processor_initialized": False,  # Always false after complete shutdown
                "results": shutdown_results,
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Critical error during complete shutdown: {e}")
        # FIXED: Force reset state even on critical error
        try:
            async with _initialization_lock:
                _processor_initialized = False
        except:
            pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Critical error during complete shutdown: {str(e)}"
        )

@detection_shutdown_router.get("/status")
async def get_detection_shutdown_status():
    """Get current status of detection service for shutdown planning"""
    try:
        from detection.app.service.detection.optimized_detection_service import detection_processor
        
        # FIXED: Check both global state and processor state
        global _processor_initialized
        
        if not _processor_initialized:
            return {
                "status": "not_initialized",
                "can_shutdown": False,
                "processor_initialized": _processor_initialized,
                "message": "Detection processor not initialized"
            }
        
        if not hasattr(detection_processor, 'is_running'):
            return {
                "status": "not_initialized",
                "can_shutdown": False,
                "processor_initialized": _processor_initialized,
                "message": "Detection processor not properly initialized"
            }
        
        try:
            stats = detection_processor.get_performance_stats()
            is_running = stats.get('is_running', False)
            queue_depth = stats.get('queue_depth', 0)
            
            # Estimate shutdown time based on queue depth
            estimated_shutdown_seconds = min(queue_depth * 2, 30)  # 2 seconds per frame, max 30 seconds
            
            return {
                "status": "running" if is_running else "stopped",
                "can_shutdown": True,  # Can always attempt shutdown
                "processor_initialized": _processor_initialized,
                "current_stats": {
                    "queue_depth": queue_depth,
                    "frames_processed": stats.get('frames_processed', 0),
                    "is_running": is_running,
                    "redis_connected": stats.get('redis_connected', False)
                },
                "estimated_shutdown_time_seconds": estimated_shutdown_seconds,
                "message": f"{'Ready for shutdown' if is_running else 'Already stopped'}",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as stats_error:
            logger.warning(f"Could not get processor stats: {stats_error}")
            return {
                "status": "unknown",
                "can_shutdown": True,  # Can always attempt shutdown
                "processor_initialized": _processor_initialized,
                "current_stats": {},
                "estimated_shutdown_time_seconds": 30,
                "message": "Processor state unknown, but shutdown available",
                "error": str(stats_error),
                "timestamp": datetime.now().isoformat()
            }
            
    except Exception as e:
        logger.error(f"Error getting shutdown status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting shutdown status: {str(e)}"
        )