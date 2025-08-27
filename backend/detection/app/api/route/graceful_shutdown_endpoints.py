# enhanced_graceful_shutdown_endpoints.py - Updated with Basic Detection lot context support
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
    ENHANCED: Complete shutdown of all detection AND identification components + basic detection
    """
    global _processor_initialized
    
    try:
        from detection.app.service.detection.optimized_detection_service import detection_processor
        from detection.app.service.identification.identification_detection_service import piece_identification_processor
        from detection.app.service.detection.alternative.basic_detection_service import basic_detection_processor
        
        logger.info("🛑 Initiating COMPLETE system shutdown (detection + identification + basic detection)...")
        
        shutdown_results = {
            "detection_service": {"status": "not_attempted"},
            "redis_processor": {"status": "not_attempted"}, 
            "identification_service": {"status": "not_attempted"},
            "basic_detection_service": {"status": "not_attempted"},
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
        
        # Step 3: Shutdown identification service
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
        
        # Step 4: NEW - Shutdown basic detection service with lot context
        logger.info("📍 Step 4: Shutting down basic detection service...")
        try:
            if basic_detection_processor.is_initialized:
                logger.info("🔧 Shutting down basic detection processor...")
                
                # Get pre-shutdown stats and lot context
                basic_stats = basic_detection_processor.get_stats()
                detections_performed = basic_stats.get('detections_performed', 0)
                lots_created = basic_stats.get('lots_created', 0)
                current_lot_context = basic_stats.get('current_lot_context', {})
                
                logger.info(f"📊 Basic detection stats: {detections_performed} detections, {lots_created} lots created")
                if current_lot_context.get('lot_id'):
                    logger.info(f"📋 Current lot context: {current_lot_context}")
                
                # Cleanup basic detection resources
                await basic_detection_processor.cleanup()
                
                # Reset basic detection state INCLUDING lot context
                basic_detection_processor.is_initialized = False
                basic_detection_processor.detection_system = None
                basic_detection_processor.clear_lot_context()  # Clear lot context
                
                # Reset initialization flags
                basic_detection_processor.is_initialized_for_lot = False
                basic_detection_processor.lot_model_loaded = False
                
                shutdown_results["basic_detection_service"] = {
                    "status": "completed",
                    "detections_performed": detections_performed,
                    "lots_created": lots_created,
                    "lot_context_cleared": current_lot_context,
                    "message": "Basic detection service shutdown completed successfully"
                }
                logger.info("✅ Basic detection processor shutdown completed")
            else:
                shutdown_results["basic_detection_service"] = {
                    "status": "already_stopped",
                    "message": "Basic detection service was not initialized"
                }
        except Exception as e:
            logger.error(f"Error in basic detection service shutdown: {e}")
            shutdown_results["basic_detection_service"] = {
                "status": "error",
                "error": str(e)
            }
            # Force reset basic detection state
            try:
                basic_detection_processor.is_initialized = False
                basic_detection_processor.detection_system = None
                basic_detection_processor.clear_lot_context()
            except:
                pass
        
        # Step 5: Enhanced cleanup and state reset
        logger.info("📍 Step 5: Resetting initialization state and cleanup...")
        try:
            # Reset detection processor attributes
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
        
        logger.info("✅ Complete system shutdown finished (detection + identification + basic detection)")
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "shutdown_complete" if all_completed else "shutdown_partial",
                "message": "Complete system shutdown finished (detection + identification + basic detection)",
                "services_shutdown": ["detection", "redis_processor", "identification", "basic_detection", "initialization_reset"],
                "processor_initialized": False,
                "identification_initialized": False,
                "basic_detection_initialized": False,
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
            basic_detection_processor.is_initialized = False
            basic_detection_processor.clear_lot_context()
        except:
            pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Critical error during complete shutdown: {str(e)}"
        )

@detection_shutdown_router.post("/basic-detection-only")
async def shutdown_basic_detection_only():
    """
    NEW: Shutdown only the basic detection service with lot context, leaving other services running
    """
    try:
        from detection.app.service.detection.alternative.basic_detection_service import basic_detection_processor
        
        logger.info("🛑 Shutting down basic detection service only...")
        
        if basic_detection_processor.is_initialized:
            # Get stats before shutdown including lot context
            stats = basic_detection_processor.get_stats()
            detections_performed = stats.get('detections_performed', 0)
            lots_created = stats.get('lots_created', 0)
            lots_completed = stats.get('lots_completed', 0)
            current_lot_context = stats.get('current_lot_context', {})
            
            logger.info(f"📊 Pre-shutdown stats: {detections_performed} detections, {lots_created} lots created, {lots_completed} completed")
            
            # Log current lot context if exists
            if current_lot_context.get('lot_id'):
                logger.info(f"📋 Clearing lot context: Lot {current_lot_context.get('lot_id')} - {current_lot_context.get('piece_label')}")
            
            # Cleanup basic detection resources
            await basic_detection_processor.cleanup()
            
            # Reset state INCLUDING lot context
            basic_detection_processor.is_initialized = False
            basic_detection_processor.detection_system = None
            basic_detection_processor.clear_lot_context()  # Clear lot context
            
            # Reset lot-specific initialization flags
            basic_detection_processor.is_initialized_for_lot = False
            basic_detection_processor.lot_model_loaded = False
            
            return JSONResponse(
                status_code=200,
                content={
                    "status": "basic_detection_shutdown_complete",
                    "message": "Basic detection service shutdown completed",
                    "detections_performed": detections_performed,
                    "lots_created": lots_created,
                    "lots_completed": lots_completed,
                    "lot_context_cleared": current_lot_context,
                    "other_services": "still_running",
                    "timestamp": datetime.now().isoformat()
                }
            )
        else:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "basic_detection_already_stopped",
                    "message": "Basic detection service was not running",
                    "other_services": "unaffected",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
    except Exception as e:
        logger.error(f"❌ Error shutting down basic detection service: {e}")
        # Force reset including lot context
        try:
            basic_detection_processor.is_initialized = False
            basic_detection_processor.detection_system = None
            basic_detection_processor.clear_lot_context()
            basic_detection_processor.is_initialized_for_lot = False
            basic_detection_processor.lot_model_loaded = False
        except:
            pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Basic detection shutdown failed: {str(e)}"
        )

@detection_shutdown_router.get("/status")
async def get_shutdown_status():
    """Enhanced status check for detection, identification, and basic detection services"""
    try:
        from detection.app.service.detection.optimized_detection_service import detection_processor
        from detection.app.service.identification.identification_detection_service import piece_identification_processor
        from detection.app.service.detection.alternative.basic_detection_service import basic_detection_processor
        
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
        
        # Check basic detection service status
        basic_detection_status = {}
        try:
            basic_stats = basic_detection_processor.get_stats()
            current_lot_context = basic_stats.get('current_lot_context', {})
            
            basic_detection_status = {
                "initialized": basic_stats.get('is_initialized', False),
                "detections_performed": basic_stats.get('detections_performed', 0),
                "lots_created": basic_stats.get('lots_created', 0),
                "lots_completed": basic_stats.get('lots_completed', 0),
                "device": basic_stats.get('device', 'unknown'),
                "lot_context": {
                    "has_active_lot": bool(current_lot_context.get('lot_id')),
                    "lot_id": current_lot_context.get('lot_id'),
                    "piece_label": current_lot_context.get('piece_label'),
                    "lot_name": current_lot_context.get('lot_name'),
                    "is_initialized_for_lot": current_lot_context.get('is_initialized_for_lot', False)
                }
            }
        except Exception as e:
            basic_detection_status = {"initialized": False, "error": str(e)}
        
        # Calculate estimated shutdown time
        queue_depth = detection_status.get('queue_depth', 0)
        estimated_shutdown_seconds = min(queue_depth * 2 + 10, 45)  # +10 for basic detection + identification cleanup
        
        can_shutdown = True  # Can always attempt shutdown
        
        # Determine overall status
        services_running = []
        if detection_status.get('initialized'):
            services_running.append("detection")
        if identification_status.get('initialized'):
            services_running.append("identification")
        if basic_detection_status.get('initialized'):
            services_running.append("basic_detection")
        
        if len(services_running) == 0:
            overall_status = "all_stopped"
        elif len(services_running) == 3:
            overall_status = "all_services_running"
        else:
            overall_status = f"partial_running_{len(services_running)}_of_3"
        
        return JSONResponse(
            status_code=200,
            content={
                "status": overall_status,
                "can_shutdown": can_shutdown,
                "services_running": services_running,
                "services": {
                    "detection": detection_status,
                    "identification": identification_status,
                    "basic_detection": basic_detection_status
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

# Existing identification-only shutdown endpoint
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
            current_group = stats.get('current_group_name')
            
            # Cleanup identification resources
            await piece_identification_processor.cleanup()
            
            # Reset state - INCLUDING group-specific state
            piece_identification_processor.is_initialized = False
            piece_identification_processor.is_group_loaded = False
            piece_identification_processor.current_group_name = None
            piece_identification_processor.detection_system = None
            piece_identification_processor._label_cache.clear()
            piece_identification_processor._frame_cache.clear()
            
            # Clear group from stats
            piece_identification_processor.stats['current_group'] = None
            
            return JSONResponse(
                status_code=200,
                content={
                    "status": "identification_shutdown_complete",
                    "message": "Identification service shutdown completed",
                    "identifications_performed": identifications_performed,
                    "previous_group": current_group,
                    "other_services": "still_running",
                    "timestamp": datetime.now().isoformat()
                }
            )
        else:
            return JSONResponse(
                status_code=200,
                content={
                    "status": "identification_already_stopped",
                    "message": "Identification service was not running",
                    "other_services": "unaffected",
                    "timestamp": datetime.now().isoformat()
                }
            )
            
    except Exception as e:
        logger.error(f"❌ Error shutting down identification service: {e}")
        # Force reset including group state
        try:
            piece_identification_processor.is_initialized = False
            piece_identification_processor.is_group_loaded = False
            piece_identification_processor.current_group_name = None
            piece_identification_processor.detection_system = None
            piece_identification_processor.stats['current_group'] = None
        except:
            pass
        
        raise HTTPException(
            status_code=500,
            detail=f"Identification shutdown failed: {str(e)}"
        )