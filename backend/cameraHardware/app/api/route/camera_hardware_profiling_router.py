# camera_hardware_profiling_router.py - Place in CameraHardware Microservice (External)
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import asyncio

from app.service.system_profiler import system_profiler

logger = logging.getLogger(__name__)

profiling_router = APIRouter(
    prefix="/system",
    tags=["System Profiling"],
    responses={404: {"description": "Not found"}},
)

# Global performance mode cache
_current_performance_mode = None
_last_profiling_time = None
_profiling_cache_duration = 300  # 5 minutes

@profiling_router.get("/specs")
async def get_system_specifications():
    """
    Get detailed system specifications
    This endpoint provides comprehensive hardware information
    """
    try:
        specs = system_profiler.get_system_specs()
        
        return {
            "status": "success",
            "system_specifications": {
                "cpu": {
                    "cores": specs.cpu_cores,
                    "frequency_mhz": specs.cpu_frequency_mhz,
                    "current_load_percent": specs.average_cpu_load
                },
                "memory": {
                    "total_gb": specs.total_ram_gb,
                    "available_gb": specs.available_ram_gb,
                    "usage_percent": round((specs.total_ram_gb - specs.available_ram_gb) / specs.total_ram_gb * 100, 2)
                },
                "gpu": {
                    "available": specs.gpu_available,
                    "name": specs.gpu_name,
                    "memory_gb": specs.gpu_memory_gb,
                    "cuda_support": specs.cuda_available
                },
                "system": {
                    "platform": specs.system_platform,
                    "timestamp": specs.timestamp.isoformat()
                }
            },
            "performance_score": specs.performance_score,
            "recommended_mode": specs.recommended_mode
        }
        
    except Exception as e:
        logger.error(f"Error getting system specs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system specifications: {str(e)}")

@profiling_router.get("/performance/recommendation")
async def get_performance_recommendation(include_runtime_test: bool = False):
    """
    Get performance mode recommendation
    
    Args:
        include_runtime_test: If True, performs a short runtime performance test
    """
    try:
        global _current_performance_mode, _last_profiling_time
        
        # Check if we have a recent cached result
        current_time = datetime.now()
        if (_last_profiling_time and _current_performance_mode and 
            (current_time - _last_profiling_time).seconds < _profiling_cache_duration and
            not include_runtime_test):
            
            logger.info("Returning cached performance recommendation")
            return {
                "status": "success",
                "source": "cached",
                "recommendation": _current_performance_mode,
                "cached_at": _last_profiling_time.isoformat(),
                "cache_valid_until": (_last_profiling_time.timestamp() + _profiling_cache_duration)
            }
        
        # Get fresh recommendation
        logger.info(f"Getting fresh performance recommendation (runtime_test: {include_runtime_test})")
        recommendation = system_profiler.get_performance_recommendation(include_runtime_test)
        
        # Cache the result
        _current_performance_mode = recommendation
        _last_profiling_time = current_time
        
        return {
            "status": "success",
            "source": "fresh",
            "recommendation": recommendation,
            "timestamp": current_time.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting performance recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance recommendation: {str(e)}")

@profiling_router.get("/performance/mode")
async def get_current_performance_mode():
    """
    Get the current recommended performance mode
    Returns either 'high' or 'low'
    """
    try:
        # Always get fresh recommendation for mode queries
        recommendation = system_profiler.get_performance_recommendation(include_runtime_test=False)
        
        mode = recommendation['final_recommendation']
        
        return {
            "status": "success",
            "performance_mode": mode,
            "performance_score": recommendation['performance_score'],
            "timestamp": datetime.now().isoformat(),
            "mode_details": {
                "high_performance": mode == "high",
                "real_time_detection": mode == "high",
                "on_demand_detection": mode == "low",
                "description": "Real-time video + detection" if mode == "high" else "Real-time video + on-demand detection"
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting current performance mode: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance mode: {str(e)}")

@profiling_router.post("/performance/test")
async def run_performance_test(duration_seconds: int = 10):
    """
    Run a runtime performance test
    
    Args:
        duration_seconds: How long to monitor system performance (default: 10 seconds)
    """
    try:
        if duration_seconds < 5:
            duration_seconds = 5
        elif duration_seconds > 30:
            duration_seconds = 30
            
        logger.info(f"Starting {duration_seconds}s performance test...")
        
        # Run the performance test
        test_results = system_profiler.monitor_runtime_performance(duration_seconds)
        
        return {
            "status": "success",
            "test_duration": duration_seconds,
            "results": test_results,
            "recommendation": test_results['recommended_mode'],
            "performance_sustainable": test_results['performance_sustainable'],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error running performance test: {e}")
        raise HTTPException(status_code=500, detail=f"Performance test failed: {str(e)}")

@profiling_router.get("/performance/monitor/{camera_id}")
async def monitor_camera_performance(camera_id: int, duration_seconds: int = 5):
    """
    Monitor performance specifically during camera operation
    This can be called when a camera stream starts to verify performance
    """
    try:
        logger.info(f"Monitoring performance for camera {camera_id} for {duration_seconds}s")
        
        # Get baseline system specs
        specs = system_profiler.get_system_specs()
        
        # Run performance monitoring
        monitor_results = system_profiler.monitor_runtime_performance(duration_seconds)
        
        # Camera-specific analysis
        camera_recommendation = "high" if (
            specs.performance_score >= 70 and
            monitor_results['performance_sustainable'] and
            monitor_results['average_cpu_percent'] < 75
        ) else "low"
        
        return {
            "status": "success",
            "camera_id": camera_id,
            "monitoring_duration": duration_seconds,
            "system_specs": {
                "performance_score": specs.performance_score,
                "cpu_cores": specs.cpu_cores,
                "available_ram_gb": specs.available_ram_gb,
                "gpu_available": specs.gpu_available
            },
            "runtime_performance": monitor_results,
            "camera_recommendation": camera_recommendation,
            "can_handle_realtime": camera_recommendation == "high",
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error monitoring camera performance: {e}")
        raise HTTPException(status_code=500, detail=f"Camera performance monitoring failed: {str(e)}")

@profiling_router.post("/performance/force-refresh")
async def force_refresh_performance_cache():
    """
    Force refresh the performance recommendation cache
    """
    try:
        global _current_performance_mode, _last_profiling_time
        
        logger.info("Force refreshing performance cache...")
        
        # Clear cache
        _current_performance_mode = None
        _last_profiling_time = None
        
        # Get fresh recommendation
        recommendation = system_profiler.get_performance_recommendation(include_runtime_test=True)
        
        # Update cache
        _current_performance_mode = recommendation
        _last_profiling_time = datetime.now()
        
        return {
            "status": "success",
            "message": "Performance cache refreshed",
            "new_recommendation": recommendation['final_recommendation'],
            "performance_score": recommendation['performance_score'],
            "timestamp": _last_profiling_time.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error force refreshing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh cache: {str(e)}")

@profiling_router.get("/health")
async def system_profiling_health():
    """
    Health check for system profiling service
    """
    try:
        # Quick system check
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        return {
            "status": "healthy",
            "service": "system_profiling",
            "quick_stats": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "available_memory_gb": round(memory.available / (1024**3), 2)
            },
            "profiling_available": True,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "system_profiling",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@profiling_router.get("/capabilities")
async def get_system_capabilities():
    """
    Get system capabilities for different performance modes
    """
    try:
        specs = system_profiler.get_system_specs()
        
        # Define capabilities for each mode
        high_performance_capabilities = {
            "real_time_video_streaming": True,
            "real_time_object_detection": True,
            "simultaneous_cameras": min(specs.cpu_cores // 2, 4) if specs.gpu_available else 1,
            "detection_fps": "15-25" if specs.gpu_available else "5-10",
            "max_resolution": "1080p" if specs.gpu_available else "720p",
            "ai_model_complexity": "YOLO v8/v9" if specs.gpu_available else "YOLO v5",
            "memory_usage_mb": "2000-4000"
        }
        
        low_performance_capabilities = {
            "real_time_video_streaming": True,
            "real_time_object_detection": False,
            "on_demand_detection": True,
            "simultaneous_cameras": 1,
            "detection_mode": "single_frame_on_click",
            "max_resolution": "720p",
            "ai_model_complexity": "YOLO v5 optimized",
            "memory_usage_mb": "500-1500"
        }
        
        return {
            "status": "success",
            "current_mode": specs.recommended_mode,
            "system_specs": {
                "cpu_cores": specs.cpu_cores,
                "ram_gb": specs.total_ram_gb,
                "gpu_available": specs.gpu_available,
                "gpu_name": specs.gpu_name,
                "performance_score": specs.performance_score
            },
            "capabilities": {
                "high_performance": high_performance_capabilities,
                "low_performance": low_performance_capabilities
            },
            "recommended_capabilities": high_performance_capabilities if specs.recommended_mode == "high" else low_performance_capabilities,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system capabilities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system capabilities: {str(e)}")

# Background task to periodically update performance mode
async def periodic_performance_check():
    """
    Background task that periodically checks system performance
    This can be used to detect if system conditions change
    """
    global _current_performance_mode, _last_profiling_time
    
    while True:
        try:
            await asyncio.sleep(300)  # Check every 5 minutes
            
            logger.info("Running periodic performance check...")
            recommendation = system_profiler.get_performance_recommendation(include_runtime_test=False)
            
            previous_mode = _current_performance_mode['final_recommendation'] if _current_performance_mode else None
            new_mode = recommendation['final_recommendation']
            
            if previous_mode and previous_mode != new_mode:
                logger.warning(f"Performance mode changed: {previous_mode} -> {new_mode}")
                # Here you could notify other services about the mode change
            
            _current_performance_mode = recommendation
            _last_profiling_time = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in periodic performance check: {e}")

@profiling_router.on_event("startup")
async def start_periodic_profiling():
    """Start background performance monitoring"""
    asyncio.create_task(periodic_performance_check())