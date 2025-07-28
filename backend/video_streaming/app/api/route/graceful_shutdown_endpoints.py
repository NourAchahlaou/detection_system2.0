# fixed_detection_redis_router.py
import asyncio
import logging
from datetime import datetime


from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

# Import your FIXED detection processor


logger = logging.getLogger(__name__)
streaming_shutdown_router = APIRouter(
        prefix="/streaming/shutdown",
        tags=["Video Streaming Service Shutdown"],
        responses={404: {"description": "Not found"}},
    )
    
@streaming_shutdown_router.post("/graceful")
async def graceful_shutdown_streaming():
        """
        Gracefully shutdown the video streaming service
        
        This will:
        1. Stop accepting new streaming connections
        2. Gracefully disconnect existing consumers
        3. Clean up all active streams
        4. Close Redis connections
        5. Stop all background tasks
        """
        try:
            from video_streaming.app.service.videoStreamingRedis import optimized_stream_manager
            
            logger.info("🛑 Initiating graceful shutdown of video streaming service...")
            
            # Get current state before shutdown
            try:
                pre_shutdown_stats = optimized_stream_manager.get_performance_stats()
                active_streams = pre_shutdown_stats.get('active_streams_count', 0)
                total_consumers = sum(
                    len(stream.consumers) for stream in optimized_stream_manager.active_streams.values()
                )
            except Exception as e:
                logger.warning(f"Could not get pre-shutdown stats: {e}")
                active_streams = len(optimized_stream_manager.active_streams)
                total_consumers = 0
            
            logger.info(f"📊 Pre-shutdown: {active_streams} active streams, {total_consumers} consumers")
            
            if active_streams == 0:
                return JSONResponse(
                    status_code=200,
                    content={
                        "status": "already_stopped",
                        "message": "No active streams to shutdown",
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            shutdown_results = []
            
            # Step 1: Get list of all active streams
            stream_keys = list(optimized_stream_manager.active_streams.keys())
            logger.info(f"🔄 Step 1: Found {len(stream_keys)} streams to shutdown")
            
            # Step 2: Gracefully shutdown each stream
            for stream_key in stream_keys:
                try:
                    logger.info(f"🔄 Shutting down stream: {stream_key}")
                    
                    stream_state = optimized_stream_manager.active_streams.get(stream_key)
                    if stream_state:
                        # Get stream info before shutdown
                        stream_stats = stream_state.get_performance_stats()
                        camera_id = stream_stats.get('camera_id', 'unknown')
                        consumer_count = len(stream_state.consumers)
                        
                        # Notify consumers about shutdown (graceful disconnect)
                        if stream_state.consumers:
                            logger.info(f"👥 Notifying {consumer_count} consumers about shutdown for camera {camera_id}")
                            # Set stop event to signal consumers
                            stream_state.stop_event.set()
                            
                            # Give consumers a moment to disconnect gracefully
                            await asyncio.sleep(1.0)
                        
                        # Remove the stream
                        await optimized_stream_manager.remove_stream(stream_key)
                        
                        shutdown_results.append({
                            "stream_key": stream_key,
                            "camera_id": camera_id,
                            "consumers_disconnected": consumer_count,
                            "status": "success"
                        })
                        
                        logger.info(f"✅ Stream {stream_key} shutdown complete")
                    
                except Exception as e:
                    logger.error(f"❌ Error shutting down stream {stream_key}: {e}")
                    shutdown_results.append({
                        "stream_key": stream_key,
                        "status": "error",
                        "error": str(e)
                    })
            
            # Step 3: Close Redis connections
            logger.info("🔄 Step 3: Closing Redis connections...")
            try:
                if hasattr(optimized_stream_manager, 'redis_client') and optimized_stream_manager.redis_client:
                    optimized_stream_manager.redis_client.close()
                logger.info("✅ Redis connections closed")
            except Exception as e:
                logger.warning(f"⚠️ Error closing Redis connections: {e}")
            
            # Step 4: Reset manager state
            logger.info("🔄 Step 4: Resetting manager state...")
            optimized_stream_manager.active_streams.clear()
            optimized_stream_manager.stats = {
                'total_frames_streamed': 0,
                'total_frames_detected': 0,
                'active_streams_count': 0,
                'redis_operations': 0
            }
            
            successful_shutdowns = sum(1 for result in shutdown_results if result.get('status') == 'success')
            failed_shutdowns = len(shutdown_results) - successful_shutdowns
            
            logger.info("✅ Video streaming service graceful shutdown completed")
            
            return JSONResponse(
                status_code=200,
                content={
                    "status": "shutdown_complete",
                    "message": "Video streaming service has been gracefully shutdown",
                    "shutdown_stats": {
                        "streams_shutdown": successful_shutdowns,
                        "failed_shutdowns": failed_shutdowns,
                        "total_consumers_disconnected": sum(
                            result.get('consumers_disconnected', 0) 
                            for result in shutdown_results 
                            if result.get('status') == 'success'
                        )
                    },
                    "shutdown_details": shutdown_results,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Error during graceful shutdown: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error during graceful shutdown: {str(e)}"
            )
    
@streaming_shutdown_router.post("/graceful/{camera_id}")
async def graceful_shutdown_camera_stream(camera_id: int):
        """Gracefully shutdown streams for a specific camera"""
        try:
            from video_streaming.app.service.videoStreamingRedis import optimized_stream_manager
            
            logger.info(f"🛑 Initiating graceful shutdown for camera {camera_id} streams...")
            
            # Find streams for this camera
            camera_streams = [
                (key, stream) for key, stream in optimized_stream_manager.active_streams.items()
                if hasattr(stream, 'config') and stream.config.camera_id == camera_id
            ]
            
            if not camera_streams:
                return JSONResponse(
                    status_code=404,
                    content={
                        "status": "not_found",
                        "message": f"No active streams found for camera {camera_id}",
                        "timestamp": datetime.now().isoformat()
                    }
                )
            
            shutdown_results = []
            total_consumers = 0
            
            for stream_key, stream_state in camera_streams:
                try:
                    consumer_count = len(stream_state.consumers)
                    total_consumers += consumer_count
                    
                    logger.info(f"🔄 Shutting down stream {stream_key} ({consumer_count} consumers)")
                    
                    # Gracefully disconnect consumers
                    if stream_state.consumers:
                        stream_state.stop_event.set()
                        await asyncio.sleep(1.0)  # Give consumers time to disconnect
                    
                    # Remove stream
                    await optimized_stream_manager.remove_stream(stream_key)
                    
                    shutdown_results.append({
                        "stream_key": stream_key,
                        "consumers_disconnected": consumer_count,
                        "status": "success"
                    })
                    
                except Exception as e:
                    logger.error(f"❌ Error shutting down stream {stream_key}: {e}")
                    shutdown_results.append({
                        "stream_key": stream_key,
                        "status": "error",
                        "error": str(e)
                    })
            
            successful_shutdowns = sum(1 for result in shutdown_results if result.get('status') == 'success')
            
            return JSONResponse(
                status_code=200,
                content={
                    "status": "shutdown_complete",
                    "camera_id": camera_id,
                    "message": f"Camera {camera_id} streams have been gracefully shutdown",
                    "shutdown_stats": {
                        "streams_shutdown": successful_shutdowns,
                        "total_consumers_disconnected": total_consumers
                    },
                    "shutdown_details": shutdown_results,
                    "timestamp": datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            logger.error(f"❌ Error during camera stream shutdown: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error during camera stream shutdown: {str(e)}"
            )
    
@streaming_shutdown_router.get("/status")
async def get_streaming_shutdown_status():
        """Get current status of streaming service for shutdown planning"""
        try:
            from video_streaming.app.service.videoStreamingRedis import optimized_stream_manager
            
            stats = optimized_stream_manager.get_performance_stats()
            active_streams = stats.get('active_streams_count', 0)
            
            # Get detailed stream info
            stream_details = []
            total_consumers = 0
            
            for stream_key, stream_state in optimized_stream_manager.active_streams.items():
                try:
                    stream_stats = stream_state.get_performance_stats()
                    consumer_count = len(stream_state.consumers)
                    total_consumers += consumer_count
                    
                    stream_details.append({
                        "stream_key": stream_key,
                        "camera_id": stream_stats.get('camera_id'),
                        "consumers": consumer_count,
                        "is_active": stream_stats.get('is_active', False),
                        "detection_enabled": stream_stats.get('detection_enabled', False)
                    })
                except Exception as e:
                    logger.warning(f"Error getting stats for stream {stream_key}: {e}")
            
            return {
                "status": "running" if active_streams > 0 else "stopped",
                "can_shutdown": active_streams > 0,
                "current_stats": {
                    "active_streams": active_streams,
                    "total_consumers": total_consumers,
                    "total_frames_streamed": stats.get('total_frames_streamed', 0)
                },
                "stream_details": stream_details,
                "estimated_shutdown_time_seconds": min(active_streams * 2, 15),  # 2 seconds per stream, max 15
                "message": f"{'Ready for shutdown' if active_streams > 0 else 'No active streams'}",
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting streaming shutdown status: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error getting shutdown status: {str(e)}"
            )