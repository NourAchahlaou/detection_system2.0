# system_profiler_router.py - Complete router for AI system profiling
from fastapi import APIRouter, HTTPException, BackgroundTasks, Query
from fastapi.responses import JSONResponse
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import asyncio
import json

from app.service.system_profiler import system_profiler

logger = logging.getLogger(__name__)

profiling_router = APIRouter(
    prefix="/system",
    tags=["System Profiling & Performance"],
    responses={404: {"description": "Not found"}},
)

# Global cache for performance data
_performance_cache = {
    "recommendation": None,
    "specs": None,
    "last_update": None,
    "cache_duration": 300  # 5 minutes
}


@profiling_router.get("/specs", summary="Get detailed system specifications")
async def get_system_specifications():
    """
    Get comprehensive system specifications with AI workload analysis
    
    Returns detailed information about:
    - CPU cores, frequency, and current load
    - Memory (total, available, usage)
    - GPU availability, VRAM, CUDA support
    - Performance scoring and tier classification
    - Minimum requirements comparison
    """
    try:
        specs = system_profiler.get_system_specs()
        
        return {
            "status": "success",
            "timestamp": specs.timestamp.isoformat(),
            "system_specifications": {
                "cpu": {
                    "cores": specs.cpu_cores,
                    "frequency_mhz": specs.cpu_frequency_mhz,
                    "current_load_percent": specs.average_cpu_load,
                    "meets_minimum": specs.cpu_cores >= system_profiler.minimum_requirements['cpu_cores'],
                    "adequacy": "excellent" if specs.cpu_cores >= 8 else "adequate" if specs.cpu_cores >= 6 else "limited"
                },
                "memory": {
                    "total_gb": specs.total_ram_gb,
                    "available_gb": specs.available_ram_gb,
                    "used_gb": round(specs.total_ram_gb - specs.available_ram_gb, 2),
                    "usage_percent": round((specs.total_ram_gb - specs.available_ram_gb) / specs.total_ram_gb * 100, 2),
                    "meets_minimum": specs.total_ram_gb >= system_profiler.minimum_requirements['total_ram_gb'],
                    "adequacy": "excellent" if specs.total_ram_gb >= 16 else "adequate" if specs.total_ram_gb >= 12 else "limited"
                },
                "gpu": {
                    "available": specs.gpu_available,
                    "name": specs.gpu_name,
                    "memory_gb": specs.gpu_memory_gb,
                    "cuda_support": specs.cuda_available,
                    "meets_minimum": (
                        specs.gpu_available and 
                        specs.cuda_available and 
                        (specs.gpu_memory_gb or 0) >= system_profiler.minimum_requirements['gpu_memory_gb']
                    ),
                    "adequacy": (
                        "excellent" if specs.cuda_available and (specs.gpu_memory_gb or 0) >= 8 
                        else "adequate" if specs.cuda_available and (specs.gpu_memory_gb or 0) >= 6
                        else "limited"
                    )
                },
                "platform": {
                    "system": specs.system_platform,
                    "optimization_notes": {
                        "Linux": "Optimal for AI workloads",
                        "Windows": "Good performance with proper drivers",
                        "Darwin": "Limited GPU acceleration options"
                    }.get(specs.system_platform, "Unknown platform")
                }
            },
            "performance_analysis": {
                "score": specs.performance_score,
                "tier": specs.performance_tier,
                "recommended_mode": specs.recommended_mode,
                "meets_minimum_requirements": specs.meets_minimum_requirements,
                "score_breakdown": {
                    "excellent": specs.performance_score >= 80,
                    "good": 60 <= specs.performance_score < 80,
                    "adequate": 40 <= specs.performance_score < 60,
                    "limited": specs.performance_score < 40
                }
            },
            "requirements_analysis": {
                "minimum_requirements": system_profiler.minimum_requirements,
                "performance_thresholds": system_profiler.performance_thresholds,
                "system_meets_advanced_mode": specs.meets_minimum_requirements,
                "upgrade_recommendations": _get_upgrade_recommendations(specs)
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting system specs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system specifications: {str(e)}")


@profiling_router.get("/performance/recommendation", summary="Get performance mode recommendation")
async def get_performance_recommendation(
    include_runtime_test: bool = Query(False, description="Include runtime performance test"),
    force_refresh: bool = Query(False, description="Force refresh cached data")
):
    """
    Get comprehensive performance recommendation for AI detection system
    
    Args:
        include_runtime_test: Run actual performance monitoring (takes 10+ seconds)
        force_refresh: Bypass cache and get fresh analysis
    
    Returns:
        - Static analysis based on hardware specs
        - Optional runtime performance test results
        - Final recommendation (advanced/basic mode)
        - Detailed reasoning and limitations
    """
    try:
        global _performance_cache
        
        # Check cache validity
        current_time = datetime.now()
        cache_valid = (
            not force_refresh and
            _performance_cache["recommendation"] is not None and
            _performance_cache["last_update"] is not None and
            (current_time - _performance_cache["last_update"]).seconds < _performance_cache["cache_duration"] and
            not include_runtime_test
        )
        
        if cache_valid:
            logger.info("Returning cached performance recommendation")
            cached_result = _performance_cache["recommendation"].copy()
            cached_result["source"] = "cached"
            cached_result["cached_at"] = _performance_cache["last_update"].isoformat()
            return {
                "status": "success",
                **cached_result
            }
        
        # Get fresh recommendation
        logger.info(f"Getting fresh performance recommendation (runtime_test: {include_runtime_test})")
        recommendation = system_profiler.get_performance_recommendation(include_runtime_test)
        
        # Enhance the recommendation with additional analysis
        enhanced_recommendation = {
            **recommendation,
            "source": "fresh",
            "analysis_timestamp": current_time.isoformat(),
            "mode_descriptions": {
                "advanced": {
                    "description": "Real-time YOLO detection with GPU acceleration",
                    "features": [
                        "Simultaneous multi-camera support",
                        "15-25 FPS detection rate",
                        "Background model training",
                        "Redis-based streaming",
                        "Full resolution processing"
                    ],
                    "requirements": "6+ CPU cores, 12+ GB RAM, CUDA GPU with 6+ GB VRAM"
                },
                "basic": {
                    "description": "CPU-based detection with optimized models",
                    "features": [
                        "Single camera support",
                        "On-demand detection",
                        "Lightweight model variants",
                        "Basic streaming",
                        "Reduced resolution processing"
                    ],
                    "requirements": "4+ CPU cores, 8+ GB RAM"
                }
            },
            "expected_performance": _get_expected_performance(recommendation)
        }
        
        # Update cache
        _performance_cache["recommendation"] = enhanced_recommendation
        _performance_cache["last_update"] = current_time
        
        return {
            "status": "success",
            **enhanced_recommendation
        }
        
    except Exception as e:
        logger.error(f"Error getting performance recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance recommendation: {str(e)}")


@profiling_router.get("/performance/mode", summary="Get current optimal performance mode")
async def get_current_performance_mode():
    """
    Get the current recommended performance mode with detailed capabilities
    
    Returns:
        - Current mode (advanced/basic)
        - Mode-specific capabilities and limitations
        - System adequacy analysis
        - Expected performance metrics
    """
    try:
        recommendation = system_profiler.get_performance_recommendation(include_runtime_test=False)
        specs = recommendation['system_specs']
        mode = recommendation['final_recommendation']
        
        # Define mode-specific capabilities
        mode_capabilities = {
            "advanced": {
                "real_time_detection": True,
                "simultaneous_cameras": min(specs['cpu_cores'] // 2, 4),
                "detection_fps_range": "15-25",
                "max_resolution": "1080p",
                "model_complexity": "YOLO v8/v9",
                "background_training": True,
                "gpu_acceleration": True,
                "memory_usage_mb": "2000-4000",
                "streaming_protocol": "Redis + WebRTC"
            },
            "basic": {
                "real_time_detection": False,
                "simultaneous_cameras": 1,
                "detection_mode": "on-demand",
                "max_resolution": "720p",
                "model_complexity": "YOLO v5 optimized",
                "background_training": False,
                "gpu_acceleration": False,
                "memory_usage_mb": "500-1500",
                "streaming_protocol": "Basic HTTP streaming"
            }
        }
        
        return {
            "status": "success",
            "performance_mode": mode,
            "performance_score": recommendation['performance_score'],
            "performance_tier": recommendation['performance_tier'],
            "meets_minimum_requirements": recommendation['meets_minimum_requirements'],
            "timestamp": datetime.now().isoformat(),
            "capabilities": mode_capabilities[mode],
            "mode_comparison": {
                "current_mode": mode,
                "alternative_mode": "basic" if mode == "advanced" else "advanced",
                "mode_switch_possible": recommendation['meets_minimum_requirements'],
                "performance_difference": "Significant GPU acceleration and multi-camera support" if mode == "advanced" else "Lower resource usage but limited functionality"
            },
            "system_adequacy": {
                "cpu_adequate": specs['cpu_cores'] >= system_profiler.minimum_requirements['cpu_cores'],
                "memory_adequate": specs['total_ram_gb'] >= system_profiler.minimum_requirements['total_ram_gb'],
                "gpu_adequate": specs.get('cuda_available', False) and (specs.get('gpu_memory_gb', 0) >= system_profiler.minimum_requirements['gpu_memory_gb']),
                "overall_adequate": recommendation['meets_minimum_requirements']
            },
            "optimization_suggestions": _get_mode_optimization_suggestions(mode, specs)
        }
        
    except Exception as e:
        logger.error(f"Error getting current performance mode: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance mode: {str(e)}")


@profiling_router.post("/performance/test", summary="Run system performance test")
async def run_performance_test(
    duration_seconds: int = Query(10, ge=5, le=30, description="Test duration in seconds")
):
    """
    Run a comprehensive runtime performance test
    
    Args:
        duration_seconds: Duration of the test (5-30 seconds)
    
    Returns:
        - Real-time system performance metrics
        - CPU, memory, and GPU utilization
        - Performance sustainability analysis
        - Updated mode recommendation based on actual performance
    """
    try:
        logger.info(f"Starting {duration_seconds}s performance test...")
        
        # Get baseline specs first
        baseline_specs = system_profiler.get_system_specs()
        
        # Run the performance test
        test_results = system_profiler.monitor_runtime_performance(duration_seconds)
        
        # Enhanced analysis
        performance_analysis = {
            "baseline_performance_score": baseline_specs.performance_score,
            "runtime_performance_sustainable": test_results['performance_sustainable'],
            "performance_degradation": {
                "cpu_stress": test_results['average_cpu_percent'] > 80,
                "memory_pressure": test_results['average_memory_percent'] > 85,
                "resource_contention": test_results['min_available_memory_gb'] < 2.0
            },
            "recommendation_confidence": "high" if test_results['performance_sustainable'] else "medium",
            "mode_stability": "stable" if test_results['performance_sustainable'] else "variable"
        }
        
        return {
            "status": "success",
            "test_configuration": {
                "duration_seconds": duration_seconds,
                "samples_collected": test_results['samples_collected']
            },
            "baseline_specs": {
                "performance_score": baseline_specs.performance_score,
                "performance_tier": baseline_specs.performance_tier,
                "recommended_mode": baseline_specs.recommended_mode
            },
            "runtime_results": test_results,
            "performance_analysis": performance_analysis,
            "final_recommendation": {
                "mode": test_results['recommended_mode'],
                "confidence": performance_analysis["recommendation_confidence"],
                "reasoning": "Based on actual runtime performance monitoring"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error running performance test: {e}")
        raise HTTPException(status_code=500, detail=f"Performance test failed: {str(e)}")


@profiling_router.get("/performance/monitor/{camera_id}", summary="Monitor camera-specific performance")
async def monitor_camera_performance(
    camera_id: int,
    duration_seconds: int = Query(5, ge=3, le=15, description="Monitoring duration")
):
    """
    Monitor system performance during specific camera operation
    
    Args:
        camera_id: ID of the camera being monitored
        duration_seconds: Duration of monitoring (3-15 seconds)
    
    Returns:
        - Camera-specific performance metrics
        - Resource utilization during camera operation
        - Real-time capability assessment
        - Camera-specific recommendations
    """
    try:
        logger.info(f"Monitoring performance for camera {camera_id}")
        
        # Get system specs
        specs = system_profiler.get_system_specs()
        
        # Monitor runtime performance
        monitor_results = system_profiler.monitor_runtime_performance(duration_seconds)
        
        # Camera-specific analysis
        camera_performance = {
            "can_handle_realtime": (
                specs.performance_score >= 60 and
                monitor_results['performance_sustainable'] and
                monitor_results['average_cpu_percent'] < 75
            ),
            "recommended_resolution": "1080p" if specs.gpu_available else "720p",
            "expected_fps": {
                "with_gpu": "15-25" if specs.cuda_available else "N/A",
                "cpu_only": "5-10",
                "current_recommendation": "15-25" if specs.cuda_available and monitor_results['performance_sustainable'] else "5-10"
            },
            "resource_allocation": {
                "cpu_cores_recommended": min(2, specs.cpu_cores // 2),
                "memory_mb_estimated": 1500 if specs.gpu_available else 800,
                "gpu_memory_mb_estimated": 2000 if specs.cuda_available else 0
            }
        }
        
        return {
            "status": "success",
            "camera_id": camera_id,
            "monitoring_duration": duration_seconds,
            "system_baseline": {
                "performance_score": specs.performance_score,
                "performance_tier": specs.performance_tier,
                "gpu_available": specs.gpu_available,
                "cuda_available": specs.cuda_available
            },
            "runtime_monitoring": monitor_results,
            "camera_performance": camera_performance,
            "recommendations": {
                "mode": "realtime" if camera_performance["can_handle_realtime"] else "on_demand",
                "resolution": camera_performance["recommended_resolution"],
                "fps_target": camera_performance["expected_fps"]["current_recommendation"],
                "optimization_tips": _get_camera_optimization_tips(specs, camera_performance)
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error monitoring camera performance: {e}")
        raise HTTPException(status_code=500, detail=f"Camera performance monitoring failed: {str(e)}")


@profiling_router.get("/capabilities", summary="Get system capabilities overview")
async def get_system_capabilities():
    """
    Get comprehensive system capabilities for different performance modes
    
    Returns:
        - Capabilities for each performance mode
        - Current system limitations and strengths
        - Mode-specific feature availability
        - Performance expectations
    """
    try:
        specs = system_profiler.get_system_specs()
        recommendation = system_profiler.get_performance_recommendation(include_runtime_test=False)
        
        capabilities = {
            "advanced_mode": {
                "available": specs.meets_minimum_requirements,
                "features": {
                    "real_time_video_streaming": True,
                    "real_time_object_detection": True,
                    "simultaneous_cameras": min(specs.cpu_cores // 2, 4) if specs.gpu_available else 2,
                    "detection_fps": "15-25" if specs.gpu_available else "10-15",
                    "max_resolution": "1080p",
                    "ai_models": ["YOLO v8", "YOLO v9", "Custom models"],
                    "gpu_acceleration": specs.cuda_available,
                    "background_training": True,
                    "model_fine_tuning": specs.cuda_available,
                    "redis_streaming": True,
                    "multi_threaded_processing": True
                },
                "resource_requirements": {
                    "cpu_cores": "6+",
                    "ram_gb": "12+",
                    "gpu_memory_gb": "6+" if specs.cuda_available else "N/A",
                    "storage_gb": "50+ (for models and training data)"
                },
                "performance_expectations": {
                    "startup_time_seconds": "30-60",
                    "detection_latency_ms": "50-100" if specs.gpu_available else "200-500",
                    "memory_usage_mb": "2000-4000",
                    "cpu_usage_percent": "40-70"
                }
            },
            "basic_mode": {
                "available": True,
                "features": {
                    "video_streaming": True,
                    "on_demand_detection": True,
                    "single_camera": True,
                    "detection_fps": "on_demand",
                    "max_resolution": "720p",
                    "ai_models": ["YOLO v5 lite", "Optimized models"],
                    "cpu_only_processing": True,
                    "simple_streaming": True,
                    "basic_alerts": True
                },
                "resource_requirements": {
                    "cpu_cores": "4+",
                    "ram_gb": "8+",
                    "storage_gb": "20+ (for basic models)"
                },
                "performance_expectations": {
                    "startup_time_seconds": "10-20",
                    "detection_latency_ms": "500-2000",
                    "memory_usage_mb": "500-1500",
                    "cpu_usage_percent": "20-50"
                }
            }
        }
        
        return {
            "status": "success",
            "current_system": {
                "performance_score": specs.performance_score,
                "performance_tier": specs.performance_tier,
                "recommended_mode": specs.recommended_mode,
                "meets_advanced_requirements": specs.meets_minimum_requirements
            },
            "capabilities": capabilities,
            "current_limitations": _get_current_limitations(specs),
            "upgrade_path": _get_upgrade_path(specs),
            "feature_matrix": {
                "real_time_detection": capabilities["advanced_mode"]["available"],
                "multi_camera": capabilities["advanced_mode"]["available"],
                "gpu_acceleration": specs.cuda_available,
                "background_training": specs.cuda_available and specs.meets_minimum_requirements,
                "high_resolution": specs.meets_minimum_requirements
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting system capabilities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system capabilities: {str(e)}")


@profiling_router.post("/cache/refresh", summary="Force refresh performance cache")
async def force_refresh_cache():
    """
    Force refresh all cached performance data
    
    Returns:
        - Updated performance recommendation
        - New system analysis
        - Cache refresh confirmation
    """
    try:
        global _performance_cache
        
        logger.info("Force refreshing performance cache...")
        
        # Clear cache
        _performance_cache = {
            "recommendation": None,
            "specs": None,
            "last_update": None,
            "cache_duration": 300
        }
        
        # Get fresh data
        recommendation = system_profiler.get_performance_recommendation(include_runtime_test=True)
        specs = system_profiler.get_system_specs()
        
        # Update cache
        current_time = datetime.now()
        _performance_cache["recommendation"] = recommendation
        _performance_cache["specs"] = specs
        _performance_cache["last_update"] = current_time
        
        return {
            "status": "success",
            "message": "Performance cache refreshed successfully",
            "refresh_timestamp": current_time.isoformat(),
            "new_recommendation": {
                "mode": recommendation['final_recommendation'],
                "performance_score": recommendation['performance_score'],
                "performance_tier": recommendation['performance_tier']
            },
            "cache_valid_until": (current_time + timedelta(seconds=300)).isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error refreshing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh cache: {str(e)}")


@profiling_router.get("/health", summary="System profiling service health check")
async def health_check():
    """
    Health check for the system profiling service
    
    Returns:
        - Service status
        - Quick system metrics
        - Service availability confirmation
    """
    try:
        import psutil
        
        # Quick system check
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        # Test profiler functionality
        test_specs = system_profiler.get_system_specs()
        
        return {
            "status": "healthy",
            "service": "system_profiler",
            "version": "2.0",
            "timestamp": datetime.now().isoformat(),
            "quick_metrics": {
                "cpu_percent": round(cpu_percent, 2),
                "memory_percent": round(memory.percent, 2),
                "available_memory_gb": round(memory.available / (1024**3), 2)
            },
            "profiler_status": {
                "available": True,
                "last_analysis": test_specs.timestamp.isoformat(),
                "performance_tier": test_specs.performance_tier,
                "recommended_mode": test_specs.recommended_mode
            },
            "endpoints_available": [
                "/system/specs",
                "/system/performance/recommendation",
                "/system/performance/mode",
                "/system/performance/test",
                "/system/capabilities",
                "/system/health"
            ]
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "system_profiler",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


# Helper functions
def _get_upgrade_recommendations(specs) -> List[str]:
    """Get specific upgrade recommendations based on current specs"""
    recommendations = []
    
    if specs.cpu_cores < 6:
        recommendations.append(f"Upgrade CPU to at least 6 cores (current: {specs.cpu_cores})")
    
    if specs.total_ram_gb < 12:
        recommendations.append(f"Upgrade RAM to at least 12GB (current: {specs.total_ram_gb}GB)")
    
    if not specs.gpu_available:
        recommendations.append("Add NVIDIA GPU with CUDA support for AI acceleration")
    elif not specs.cuda_available:
        recommendations.append("Ensure CUDA drivers are properly installed")
    elif specs.gpu_memory_gb and specs.gpu_memory_gb < 6:
        recommendations.append(f"Upgrade GPU VRAM to at least 6GB (current: {specs.gpu_memory_gb}GB)")
    
    if specs.cpu_frequency_mhz < 2500:
        recommendations.append(f"Consider faster CPU (current: {specs.cpu_frequency_mhz}MHz)")
    
    return recommendations


def _get_expected_performance(recommendation) -> Dict[str, Any]:
    """Get expected performance metrics based on recommendation"""
    mode = recommendation['final_recommendation']
    specs = recommendation['system_specs']
    
    if mode == 'advanced':
        return {
            "detection_fps": "15-25" if specs.get('cuda_available') else "10-15",
            "startup_time_seconds": "30-60",
            "memory_usage_mb": "2000-4000",
            "cpu_usage_percent": "40-70",
            "simultaneous_cameras": min(specs['cpu_cores'] // 2, 4),
            "resolution_support": "up to 1080p",
            "real_time_capable": True
        }
    else:
        return {
            "detection_mode": "on-demand",
            "startup_time_seconds": "10-20",
            "memory_usage_mb": "500-1500",
            "cpu_usage_percent": "20-50",
            "simultaneous_cameras": 1,
            "resolution_support": "up to 720p",
            "real_time_capable": False
        }


def _get_mode_optimization_suggestions(mode: str, specs: Dict[str, Any]) -> List[str]:
    """Get optimization suggestions for the current mode"""
    suggestions = []
    
    if mode == 'advanced':
        suggestions.extend([
            "Close unnecessary background applications",
            "Ensure adequate cooling for sustained performance",
            "Monitor GPU temperature during intensive operations"
        ])
        
        if specs['available_ram_gb'] < 10:
            suggestions.append("Consider closing memory-intensive applications")
            
        if specs['cpu_cores'] < 8:
            suggestions.append("Limit simultaneous camera streams for optimal performance")
    else:
        suggestions.extend([
            "Use lower resolution for better performance",
            "Process detections in batches rather than continuously",
            "Consider scheduled detection runs during low-usage periods"
        ])
    
    return suggestions


def _get_camera_optimization_tips(specs, camera_performance: Dict[str, Any]) -> List[str]:
    """Get camera-specific optimization tips"""
    tips = []
    
    if not camera_performance["can_handle_realtime"]:
        tips.extend([
            "Use motion detection to trigger AI analysis",
            "Reduce frame rate to 5-10 FPS",
            "Consider lower resolution (720p or 480p)"
        ])
    
    if not specs.cuda_available:
        tips.extend([
            "Use CPU-optimized YOLO models",
            "Enable model quantization for faster inference",
            "Process every 2nd or 3rd frame"
        ])
    
    if specs.available_ram_gb < 8:
        tips.append("Close other applications to free memory")
    
    return tips


def _get_current_limitations(specs) -> List[str]:
    """Get current system limitations"""
    limitations = []
    
    if not specs.meets_minimum_requirements:
        limitations.append("System does not meet minimum requirements for advanced mode")
    
    if not specs.gpu_available:
        limitations.append("No GPU acceleration available - CPU-only processing")
    elif not specs.cuda_available:
        limitations.append("GPU available but CUDA not supported")
    
    if specs.cpu_cores < 8:
        limitations.append("Limited simultaneous camera support")
    
    if specs.total_ram_gb < 16:
        limitations.append("Memory may limit model complexity and simultaneous operations")
    
    return limitations


def _get_upgrade_path(specs) -> Dict[str, Any]:
    """Get suggested upgrade path"""
    if specs.meets_minimum_requirements:
        return {
            "current_tier": "adequate",
            "next_tier": "high_performance",
            "suggested_upgrades": [
                "Upgrade to higher-end GPU for better performance",
                "Add more RAM for larger models",
                "Consider faster storage for model loading"
            ]
        }
    else:
        return {
            "current_tier": "below_minimum",
            "next_tier": "adequate",
            "critical_upgrades": _get_upgrade_recommendations(specs),
            "estimated_cost": "Varies by component selection"
        }


# Background tasks
async def periodic_performance_monitoring():
    """Background task for periodic performance monitoring"""
    global _performance_cache
    
    while True:
        try:
            await asyncio.sleep(300)  # Every 5 minutes
            
            logger.info("Running periodic performance check...")
            
            # Update cache with fresh data
            recommendation = system_profiler.get_performance_recommendation(include_runtime_test=False)
            specs = system_profiler.get_system_specs()
            
            old_mode = _performance_cache["recommendation"]["final_recommendation"] if _performance_cache["recommendation"] else None
            new_mode = recommendation["final_recommendation"]
            
            if old_mode and old_mode != new_mode:
                logger.warning(f"Performance mode changed: {old_mode} -> {new_mode}")
                # Could trigger notifications or other services here
            
            _performance_cache["recommendation"] = recommendation
            _performance_cache["specs"] = specs
            _performance_cache["last_update"] = datetime.now()
            
        except Exception as e:
            logger.error(f"Error in periodic performance monitoring: {e}")


# Startup event
@profiling_router.on_event("startup")
async def startup_profiling_service():
    """Initialize the profiling service"""
    logger.info("Starting system profiling service...")
    
    # Initialize cache with first reading
    try:
        specs = system_profiler.get_system_specs()
        recommendation = system_profiler.get_performance_recommendation(include_runtime_test=False)
        
        global _performance_cache
        _performance_cache["recommendation"] = recommendation
        _performance_cache["specs"] = specs
        _performance_cache["last_update"] = datetime.now()
        
        logger.info(f"System profiling initialized - Mode: {specs.recommended_mode}, Score: {specs.performance_score}")
        
        # Start background monitoring
        asyncio.create_task(periodic_performance_monitoring())
        
    except Exception as e:
        logger.error(f"Failed to initialize system profiling: {e}")