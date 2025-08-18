# graceful_shutdown_endpoints.py - ENHANCED VERSION with Identification Support
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
    ENHANCED: Complete shutdown of all detection AND identification components
    """
    global _processor_initialized
    
    try:
        from detection.app.service.detection.optimized_detection_service import detection_processor
        from detection.app.service.identification.identification_detection_service import piece_identification_processor
        
        logger.info("🛑 Initiating COMPLETE system shutdown (detection + identification)...")
        
        shutdown_results = {
            "detection_service": {"status": "not_attempted"},
            "redis_processor": {"status": "not_attempted"}, 
            "identification_service": {"status": "not_attempted"},
            "initialization_reset": {"status": "not_attempted"}
        }
        
        # Step 1: Graceful shutdown of detection service (existing logic)
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
        
        # Step 2: Shutdown Redis processor (existing logic)
        logger.info("📍 Step 2: Shutting down Redis processor...")
        try:
            async with _initialization_lock:
                if _processor_initialized:
                    logger.info("🔧 Calling detection_processor.shutdown()...")
                    await detection_processor.shutdown()
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
            async with _initialization_lock:
                _processor_initialized = False
        
        # Step 3: NEW - Shutdown identification service
        logger.info("📍 Step 3: Shutting down identification service...")
        try:
            if piece_identification_processor.is_initialized:
                logger.info("🔧 Shutting down identification processor...")
                
                # Get pre-shutdown stats
                identification_stats = piece_identification_processor.get_stats()
                identifications_performed = identification_stats.get('identifications_performed', 0)
                
                # Cleanup identification resources
                await piece_identification_processor.cleanup()
                
                # Reset identification state
                piece_identification_processor.is_initialized = False
                piece_identification_processor.detection_system = None
                
                # Clear caches
                piece_identification_processor._label_cache.clear()
                piece_identification_processor._frame_cache.clear()
                
                # Reset stats (keep historical data but mark as stopped)
                piece_identification_processor.stats['is_initialized'] = False
                
                shutdown_results["identification_service"] = {
                    "status": "completed",
                    "identifications_performed": identifications_performed,
                    "message": "Identification service shutdown completed successfully"
                }
                logger.info("✅ Identification processor shutdown completed")
            else:
                shutdown_results["identification_service"] = {
                    "status": "already_stopped",
                    "message": "Identification service was not initialized"
                }
        except Exception as e:
            logger.error(f"Error in identification service shutdown: {e}")
            shutdown_results["identification_service"] = {
                "status": "error",
                "error": str(e)
            }
            # Force reset identification state
            try:
                piece_identification_processor.is_initialized = False
                piece_identification_processor.detection_system = None
            except:
                pass
        
        # Step 4: Enhanced cleanup and state reset (existing logic)
        logger.info("📍 Step 4: Resetting initialization state and cleanup...")
        try:
            # Reset detection processor attributes (existing logic)
            try:
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
                
                with detection_processor.queue_lock:
                    detection_processor.high_priority_queue.clear()
                    detection_processor.normal_priority_queue.clear()
                
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
            
            # Final state verification
            async with _initialization_lock:
                _processor_initialized = False
            
            shutdown_results["initialization_reset"] = {
                "status": "completed",
                "message": "All initialization states and resources cleaned up successfully"
            }
            
        except Exception as e:
            logger.error(f"Error in cleanup: {e}")
            shutdown_results["initialization_reset"] = {
                "status": "error",
                "error": str(e)
            }
            async with _initialization_lock:
                _processor_initialized = False
        
        # Determine overall status
        all_completed = all(
            result["status"] in ["completed", "already_stopped"] 
            for result in shutdown_results.values()
        )
        
        logger.info("✅ Complete system shutdown finished (detection + identification)")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "shutdown_complete" if all_completed else "shutdown_partial",
                "message": "Complete system shutdown finished (detection + identification)",
                "services_shutdown": ["detection", "redis_processor", "identification", "initialization_reset"],
                "processor_initialized": False,
                "identification_initialized": False,
                "results": shutdown_results,
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Critical error during complete shutdown: {e}")
        # Force reset all states
        try:
            async with _initialization_lock:
                _processor_initialized = False
            piece_identification_processor.is_initialized = False
        except:
            pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Critical error during complete shutdown: {str(e)}"
        )

@detection_shutdown_router.get("/status")
async def get_shutdown_status():
    """Enhanced status check for both detection and identification services"""
    try:
        from detection.app.service.detection.optimized_detection_service import detection_processor
        from detection.app.service.identification.identification_detection_service import piece_identification_processor
        
        global _processor_initialized
        
        # Check detection service status
        detection_status = {}
        if _processor_initialized and hasattr(detection_processor, 'is_running'):
            try:
                stats = detection_processor.get_performance_stats()
                detection_status = {
                    "initialized": True,
                    "running": stats.get('is_running', False),
                    "queue_depth": stats.get('queue_depth', 0),
                    "frames_processed": stats.get('frames_processed', 0)
                }
            except:
                detection_status = {"initialized": True, "running": False, "error": "Could not get stats"}
        else:
            detection_status = {"initialized": False, "running": False}
        
        # Check identification service status
        identification_status = {}
        try:
            id_stats = piece_identification_processor.get_stats()
            identification_status = {
                "initialized": id_stats.get('is_initialized', False),
                "identifications_performed": id_stats.get('identifications_performed', 0),
                "device": id_stats.get('device', 'unknown')
            }
        except Exception as e:
            identification_status = {"initialized": False, "error": str(e)}
        
        # Calculate estimated shutdown time
        queue_depth = detection_status.get('queue_depth', 0)
        estimated_shutdown_seconds = min(queue_depth * 2 + 5, 35)  # +5 for identification cleanup
        
        can_shutdown = True  # Can always attempt shutdown
        
        overall_status = "mixed"
        if detection_status.get('initialized') or identification_status.get('initialized'):
            overall_status = "services_running"
        else:
            overall_status = "all_stopped"
        
        return JSONResponse(
            status_code=200,
            content={
                "status": overall_status,
                "can_shutdown": can_shutdown,
                "services": {
                    "detection": detection_status,
                    "identification": identification_status
                },
                "estimated_shutdown_time_seconds": estimated_shutdown_seconds,
                "message": f"System status: {overall_status}",
                "timestamp": datetime.now().isoformat()
            }
        )
        
    except Exception as e:
        logger.error(f"Error getting shutdown status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting shutdown status: {str(e)}"
        )

# NEW: Identification-only shutdown endpoint
@detection_shutdown_router.post("/identification-only")
async def shutdown_identification_only():
    """
    Shutdown only the identification service, leaving detection running
    """
    try:
        from detection.app.service.identification.identification_detection_service import piece_identification_processor
        
        logger.info("🛑 Shutting down identification service only...")
        
        if piece_identification_processor.is_initialized:
            # Get stats before shutdown
            stats = piece_identification_processor.get_stats()
            identifications_performed = stats.get('identifications_performed', 0)
            
            # Cleanup identification resources
            await piece_identification_processor.cleanup()
            
            # Reset state
            piece_identification_processor.is_initialized = False
            piece_identification_processor.detection_system = None
            piece_identification_processor._label_cache.clear()
            piece_identification_processor._frame_cache.clear()
            
            return JSONResponse(
                status_code=200,
                content={
                    "status": "identification_shutdown_complete",
                    "message": "Identification service shutdown completed",
                    "identifications_performed": identifications_performed,
                    "detection_service": "still_running",
                    "timestamp": datetime.now().isoformat()
                }
            )
        else:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "identification_already_stopped",
                    "message": "Identification service was not running",
                    "detection_service": "unaffected",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
    except Exception as e:
        logger.error(f"❌ Error shutting down identification service: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Identification shutdown failed: {str(e)}"
        )