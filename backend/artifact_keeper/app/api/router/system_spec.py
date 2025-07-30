# system_profiling_router.py - Complete Artifact Keeper Service Router
from fastapi import APIRouter, HTTPException, Depends, Query, BackgroundTasks
from fastapi.responses import JSONResponse
import logging
import requests
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import asyncio
from dataclasses import dataclass
import json

logger = logging.getLogger(__name__)

# Configuration - adjust these URLs to match your environment
HARDWARE_SERVICE_URL = "http://host.docker.internal:8003"  # Adjust as needed

system_router = APIRouter(
    prefix="/system",
    tags=["System Profiling"],
    responses={404: {"description": "Not found"}},
)

# Cache for performance data
_performance_cache = {
    "profile": None,
    "recommendation": None,
    "specs": None,
    "capabilities": None,
    "last_updated": None,
    "cache_duration": 30  # 30 seconds as per your JS service
}

class HardwareServiceClient:
    """Comprehensive client for communicating with the Hardware Service"""
    
    def __init__(self, base_url: str = HARDWARE_SERVICE_URL):
        self.base_url = base_url.rstrip('/')
        
    async def get_system_specs(self) -> Dict[str, Any]:
        """Get detailed system specifications from hardware service"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(f"{self.base_url}/system/specs")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get system specs: {e}")
            raise HTTPException(status_code=503, detail=f"Hardware service unavailable: {str(e)}")
    
    async def get_performance_recommendation(self, include_runtime_test: bool = False, force_refresh: bool = False) -> Dict[str, Any]:
        """Get performance recommendation from hardware service"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=45.0) as client:
                params = {
                    "include_runtime_test": include_runtime_test,
                    "force_refresh": force_refresh
                }
                response = await client.get(f"{self.base_url}/system/performance/recommendation", params=params)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get performance recommendation: {e}")
            raise HTTPException(status_code=503, detail=f"Hardware service unavailable: {str(e)}")
    
    async def get_performance_mode(self) -> Dict[str, Any]:
        """Get current performance mode from hardware service"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/system/performance/mode")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get performance mode: {e}")
            raise HTTPException(status_code=503, detail=f"Hardware service unavailable: {str(e)}")
    
    async def run_performance_test(self, duration_seconds: int = 10) -> Dict[str, Any]:
        """Run system performance test"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=60.0) as client:
                params = {"duration_seconds": min(max(duration_seconds, 5), 30)}
                response = await client.post(f"{self.base_url}/system/performance/test", params=params)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to run performance test: {e}")
            raise HTTPException(status_code=503, detail=f"Hardware service unavailable: {str(e)}")
    
    async def monitor_camera_performance(self, camera_id: int, duration_seconds: int = 5) -> Dict[str, Any]:
        """Monitor camera-specific performance"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                params = {"duration_seconds": min(max(duration_seconds, 3), 15)}
                response = await client.get(f"{self.base_url}/system/performance/monitor/{camera_id}", params=params)
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to monitor camera performance: {e}")
            raise HTTPException(status_code=503, detail=f"Hardware service unavailable: {str(e)}")
    
    async def get_system_capabilities(self) -> Dict[str, Any]:
        """Get comprehensive system capabilities"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(f"{self.base_url}/system/capabilities")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get system capabilities: {e}")
            raise HTTPException(status_code=503, detail=f"Hardware service unavailable: {str(e)}")
    
    async def force_refresh_cache(self) -> Dict[str, Any]:
        """Force refresh hardware service cache"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(f"{self.base_url}/system/cache/refresh")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to refresh hardware cache: {e}")
            raise HTTPException(status_code=503, detail=f"Hardware service unavailable: {str(e)}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check hardware service health"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/system/health")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Hardware service health check failed: {e}")
            raise HTTPException(status_code=503, detail=f"Hardware service unavailable: {str(e)}")

# Dependency to get hardware service client
def get_hardware_client() -> HardwareServiceClient:
    return HardwareServiceClient()

def transform_hardware_profile(hardware_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform hardware service response to match frontend expectations
    """
    try:
        if hardware_data.get("status") != "success":
            raise ValueError("Hardware service returned error status")
        
        specs = hardware_data.get("system_specifications", {})
        cpu = specs.get("cpu", {})
        memory = specs.get("memory", {})
        gpu = specs.get("gpu", {})
        performance = hardware_data.get("performance_analysis", {})
        
        # Calculate avg_fps based on system capabilities (estimation)
        avg_fps = 30  # Default assumption
        if gpu.get("cuda_support", False) and gpu.get("meets_minimum", False):
            avg_fps = 25
        elif gpu.get("available", False):
            avg_fps = 20
        elif cpu.get("cores", 2) >= 6:
            avg_fps = 15
        else:
            avg_fps = 10
            
        # Transform to match your frontend expectations
        profile = {
            "cpu_cores": cpu.get("cores", 2),
            "cpu_frequency_mhz": cpu.get("frequency_mhz", 2000),
            "total_memory_gb": memory.get("total_gb", 4.0),
            "available_memory_gb": memory.get("available_gb", 2.0),
            "gpu_available": gpu.get("available", False),
            "gpu_name": gpu.get("name"),
            "gpu_memory_gb": gpu.get("memory_gb"),
            "cuda_available": gpu.get("cuda_support", False),
            "performance_score": performance.get("score", 30),
            "performance_tier": performance.get("tier", "limited"),
            "avg_fps": avg_fps,
            "current_cpu_load": cpu.get("current_load_percent", 50),
            "meets_minimum_requirements": performance.get("meets_minimum_requirements", False),
            "recommended_mode": performance.get("recommended_mode", "basic"),
            "timestamp": datetime.now().isoformat()
        }
        
        return profile
        
    except Exception as e:
        logger.error(f"Error transforming hardware profile: {e}")
        # Return safe defaults
        return {
            "cpu_cores": 2,
            "cpu_frequency_mhz": 2000,
            "total_memory_gb": 4.0,
            "available_memory_gb": 2.0,
            "gpu_available": False,
            "gpu_name": None,
            "gpu_memory_gb": None,
            "cuda_available": False,
            "performance_score": 30,
            "performance_tier": "limited",
            "avg_fps": 15,
            "current_cpu_load": 50,
            "meets_minimum_requirements": False,
            "recommended_mode": "basic",
            "timestamp": datetime.now().isoformat()
        }

# === MAIN ENDPOINTS MATCHING HARDWARE SERVICE ===

@system_router.get("/specs", summary="Get detailed system specifications")
async def get_system_specifications(
    hardware_client: HardwareServiceClient = Depends(get_hardware_client)
):
    """
    Get comprehensive system specifications with AI workload analysis
    Proxies to: /system/specs from hardware service
    """
    try:
        specs_data = await hardware_client.get_system_specs()
        
        # Transform for consistency but keep original structure available
        transformed_profile = transform_hardware_profile(specs_data)
        
        return {
            "status": "success",
            "specifications": specs_data,  # Original detailed specs
            "profile": transformed_profile,  # Transformed for frontend compatibility
            "timestamp": datetime.now().isoformat(),
            "source": "hardware_service"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting system specifications: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system specifications: {str(e)}")

@system_router.get("/performance/recommendation", summary="Get performance mode recommendation")
async def get_performance_recommendation(
    include_runtime_test: bool = Query(False, description="Include runtime performance test"),
    force_refresh: bool = Query(False, description="Force refresh cached data"),
    hardware_client: HardwareServiceClient = Depends(get_hardware_client)
):
    """
    Get comprehensive performance recommendation for AI detection system
    Proxies to: /system/performance/recommendation from hardware service
    """
    try:
        current_time = datetime.now()
        
        # Check cache first (unless force refresh or runtime test)
        if (not force_refresh and not include_runtime_test and
            _performance_cache["recommendation"] and 
            _performance_cache["last_updated"]):
            
            time_diff = (current_time - _performance_cache["last_updated"]).total_seconds()
            if time_diff < _performance_cache["cache_duration"]:
                logger.info("Returning cached performance recommendation")
                cached_result = _performance_cache["recommendation"].copy()
                cached_result["source"] = "cached"
                cached_result["cached_at"] = _performance_cache["last_updated"].isoformat()
                return cached_result
        
        logger.info(f"Fetching fresh performance recommendation (runtime_test: {include_runtime_test})")
        
        recommendation_data = await hardware_client.get_performance_recommendation(
            include_runtime_test=include_runtime_test,
            force_refresh=force_refresh
        )
        
        # Update cache
        _performance_cache["recommendation"] = recommendation_data
        _performance_cache["last_updated"] = current_time
        
        return recommendation_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance recommendation: {str(e)}")

@system_router.get("/performance/mode", summary="Get current optimal performance mode")
async def get_current_performance_mode(
    hardware_client: HardwareServiceClient = Depends(get_hardware_client)
):
    """
    Get the current recommended performance mode with detailed capabilities
    Proxies to: /system/performance/mode from hardware service
    """
    try:
        mode_data = await hardware_client.get_performance_mode()
        return mode_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance mode: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance mode: {str(e)}")

@system_router.post("/performance/test", summary="Run system performance test")
async def run_performance_test(
    duration_seconds: int = Query(10, ge=5, le=30, description="Test duration in seconds"),
    hardware_client: HardwareServiceClient = Depends(get_hardware_client)
):
    """
    Run a comprehensive runtime performance test
    Proxies to: /system/performance/test from hardware service
    """
    try:
        test_data = await hardware_client.run_performance_test(duration_seconds)
        return test_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error running performance test: {e}")
        raise HTTPException(status_code=500, detail=f"Performance test failed: {str(e)}")

@system_router.get("/performance/monitor/{camera_id}", summary="Monitor camera-specific performance")
async def monitor_camera_performance(
    camera_id: int,
    duration_seconds: int = Query(5, ge=3, le=15, description="Monitoring duration"),
    hardware_client: HardwareServiceClient = Depends(get_hardware_client)
):
    """
    Monitor system performance during specific camera operation
    Proxies to: /system/performance/monitor/{camera_id} from hardware service
    """
    try:
        monitor_data = await hardware_client.monitor_camera_performance(camera_id, duration_seconds)
        return monitor_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error monitoring camera performance: {e}")
        raise HTTPException(status_code=500, detail=f"Camera performance monitoring failed: {str(e)}")

@system_router.get("/capabilities", summary="Get system capabilities overview")
async def get_system_capabilities(
    hardware_client: HardwareServiceClient = Depends(get_hardware_client)
):
    """
    Get comprehensive system capabilities for different performance modes
    Proxies to: /system/capabilities from hardware service
    """
    try:
        current_time = datetime.now()
        
        # Check cache first
        if (_performance_cache["capabilities"] and 
            _performance_cache["last_updated"]):
            
            time_diff = (current_time - _performance_cache["last_updated"]).total_seconds()
            if time_diff < _performance_cache["cache_duration"]:
                logger.info("Returning cached system capabilities")
                cached_result = _performance_cache["capabilities"].copy()
                cached_result["source"] = "cached"
                return cached_result
        
        logger.info("Fetching fresh system capabilities")
        capabilities_data = await hardware_client.get_system_capabilities()
        
        # Update cache
        _performance_cache["capabilities"] = capabilities_data
        _performance_cache["last_updated"] = current_time
        
        return capabilities_data
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting system capabilities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system capabilities: {str(e)}")

@system_router.post("/cache/refresh", summary="Force refresh performance cache")
async def force_refresh_cache(
    hardware_client: HardwareServiceClient = Depends(get_hardware_client)
):
    """
    Force refresh all cached performance data
    Proxies to: /system/cache/refresh from hardware service
    """
    try:
        global _performance_cache
        
        logger.info("Force refreshing all performance caches...")
        
        # Clear local cache
        _performance_cache = {
            "profile": None,
            "recommendation": None,
            "specs": None,
            "capabilities": None,
            "last_updated": None,
            "cache_duration": 30
        }
        
        # Refresh hardware service cache
        hardware_refresh = await hardware_client.force_refresh_cache()
        
        # Get fresh data to populate local cache
        current_time = datetime.now()
        fresh_profile = await get_system_profile(force_refresh=True, hardware_client=hardware_client)
        
        return {
            "status": "success",
            "message": "All caches refreshed successfully",
            "hardware_service_refresh": hardware_refresh,
            "local_cache_refreshed": True,
            "refresh_timestamp": current_time.isoformat(),
            "new_profile": fresh_profile
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh cache: {str(e)}")

@system_router.get("/health", summary="System profiling service health check")
async def health_check():
    """
    Health check for the system profiling service
    Also checks hardware service health
    """
    try:
        # Check hardware service health
        hardware_client = HardwareServiceClient()
        
        try:
            hardware_health = await hardware_client.health_check()
            hardware_service_healthy = hardware_health.get("status") == "healthy"
        except:
            hardware_service_healthy = False
            hardware_health = {"status": "unavailable"}
        
        # Check local cache status
        cache_status = {
            "has_cached_profile": _performance_cache["profile"] is not None,
            "has_cached_recommendation": _performance_cache["recommendation"] is not None,
            "has_cached_capabilities": _performance_cache["capabilities"] is not None,
            "last_updated": _performance_cache["last_updated"].isoformat() if _performance_cache["last_updated"] else None,
            "cache_age_seconds": (datetime.now() - _performance_cache["last_updated"]).total_seconds() if _performance_cache["last_updated"] else None
        }
        
        overall_status = "healthy" if hardware_service_healthy else "degraded"
        
        return {
            "status": overall_status,
            "service": "artifact_keeper_system_profiling",
            "version": "2.0",
            "hardware_service": {
                "available": hardware_service_healthy,
                "health": hardware_health,
                "url": HARDWARE_SERVICE_URL
            },
            "cache_status": cache_status,
            "endpoints_available": [
                "/system/specs",
                "/system/performance/recommendation", 
                "/system/performance/mode",
                "/system/performance/test",
                "/system/performance/monitor/{camera_id}",
                "/system/capabilities",
                "/system/cache/refresh",
                "/system/profile",  # Legacy endpoint
                "/system/health"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "service": "artifact_keeper_system_profiling",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )

# === LEGACY ENDPOINTS FOR BACKWARD COMPATIBILITY ===

@system_router.get("/profile")
async def get_system_profile(
    force_refresh: bool = False,
    hardware_client: HardwareServiceClient = Depends(get_hardware_client)
):
    """
    Get system performance profile - Legacy endpoint for frontend compatibility
    This matches the endpoint called in your JavaScript service: '/api/artifact_keeper/system/profile'
    """
    try:
        current_time = datetime.now()
        
        # Check cache first (unless force refresh)
        if (not force_refresh and 
            _performance_cache["profile"] and 
            _performance_cache["last_updated"]):
            
            time_diff = (current_time - _performance_cache["last_updated"]).total_seconds()
            if time_diff < _performance_cache["cache_duration"]:
                logger.info("Returning cached system profile")
                return _performance_cache["profile"]
        
        logger.info("Fetching fresh system profile from hardware service")
        
        # Get data from hardware service
        hardware_data = await hardware_client.get_system_specs()
        
        # Transform to expected format
        profile = transform_hardware_profile(hardware_data)
        
        # Update cache
        _performance_cache["profile"] = profile
        _performance_cache["last_updated"] = current_time
        
        logger.info(f"System profile updated: Performance score {profile['performance_score']}")
        return profile
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting system profile: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system profile: {str(e)}")

@system_router.get("/recommendation")
async def get_performance_recommendation_legacy(
    include_runtime_test: bool = False,
    hardware_client: HardwareServiceClient = Depends(get_hardware_client)
):
    """
    Get comprehensive performance recommendation - Legacy endpoint
    """
    try:
        recommendation_data = await hardware_client.get_performance_recommendation(include_runtime_test)
        
        if recommendation_data.get("status") != "success":
            raise HTTPException(status_code=503, detail="Hardware service returned error")
        
        # Transform to legacy format if needed
        return {
            "recommended_mode": recommendation_data.get("final_recommendation", "basic"),
            "performance_score": recommendation_data.get("performance_score", 30),
            "system_specs": recommendation_data.get("system_specs", {}),
            "reasoning": recommendation_data.get("reasoning", {}),
            "thresholds": recommendation_data.get("thresholds", {}),
            "runtime_performance": recommendation_data.get("runtime_performance") if include_runtime_test else None,
            "timestamp": recommendation_data.get("timestamp", datetime.now().isoformat())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting recommendation: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get recommendation: {str(e)}")

@system_router.post("/profile/refresh")
async def force_refresh_profile(
    hardware_client: HardwareServiceClient = Depends(get_hardware_client)
):
    """
    Force refresh the system profile cache - Legacy endpoint
    """
    try:
        logger.info("Force refreshing system profile")
        
        # Clear cache
        _performance_cache["profile"] = None
        _performance_cache["last_updated"] = None
        
        # Get fresh profile
        profile = await get_system_profile(force_refresh=True, hardware_client=hardware_client)
        
        return {
            "status": "success",
            "message": "System profile refreshed",
            "profile": profile,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error refreshing profile: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to refresh profile: {str(e)}")

# === ADDITIONAL UTILITY ENDPOINTS ===

@system_router.get("/performance/history")
async def get_performance_history():
    """
    Get performance history (placeholder for future implementation)
    """
    return {
        "status": "success",
        "message": "Performance history not yet implemented",
        "history": [],
        "timestamp": datetime.now().isoformat()
    }

@system_router.get("/performance/summary")
async def get_performance_summary(
    hardware_client: HardwareServiceClient = Depends(get_hardware_client)
):
    """
    Get a quick performance summary combining multiple endpoints
    """
    try:
        # Get basic profile and mode info
        profile = await get_system_profile(hardware_client=hardware_client)
        mode_data = await hardware_client.get_performance_mode()
        
        return {
            "status": "success",
            "summary": {
                "performance_score": profile["performance_score"],
                "performance_tier": profile["performance_tier"],
                "recommended_mode": profile["recommended_mode"],
                "current_cpu_load": profile["current_cpu_load"],
                "available_memory_gb": profile["available_memory_gb"],
                "gpu_available": profile["gpu_available"],
                "cuda_available": profile["cuda_available"],
                "meets_minimum_requirements": profile["meets_minimum_requirements"]
            },
            "quick_capabilities": {
                "real_time_detection": mode_data.get("capabilities", {}).get("real_time_detection", False),
                "simultaneous_cameras": mode_data.get("capabilities", {}).get("simultaneous_cameras", 1),
                "expected_fps": profile["avg_fps"]
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting performance summary: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance summary: {str(e)}")

# === BACKGROUND TASKS ===

async def background_cache_warmer():
    """
    Background task to periodically refresh the cache
    """
    while True:
        try:
            await asyncio.sleep(25)  # Refresh every 25 seconds (before 30s cache expires)
            
            if _performance_cache["last_updated"]:
                time_diff = (datetime.now() - _performance_cache["last_updated"]).total_seconds()
                if time_diff >= 25:  # Refresh if close to expiry
                    logger.info("Background cache refresh")
                    hardware_client = HardwareServiceClient()
                    await get_system_profile(force_refresh=True, hardware_client=hardware_client)
                    
        except Exception as e:
            logger.error(f"Background cache warmer error: {e}")

async def periodic_health_monitoring():
    """
    Background task for periodic health monitoring
    """
    while True:
        try:
            await asyncio.sleep(300)  # Every 5 minutes
            
            logger.info("Running periodic health check...")
            hardware_client = HardwareServiceClient()
            
            try:
                health = await hardware_client.health_check()
                if health.get("status") != "healthy":
                    logger.warning(f"Hardware service health degraded: {health}")
            except Exception as e:
                logger.error(f"Hardware service health check failed: {e}")
                
        except Exception as e:
            logger.error(f"Error in periodic health monitoring: {e}")

@system_router.on_event("startup")
async def start_background_tasks():
    """Start background tasks"""
    logger.info("Starting system profiling background tasks...")
    asyncio.create_task(background_cache_warmer())
    asyncio.create_task(periodic_health_monitoring())
    
    # Initialize cache with first reading
    try:
        hardware_client = HardwareServiceClient()
        initial_profile = await get_system_profile(force_refresh=True, hardware_client=hardware_client)
        logger.info(f"System profiling initialized - Mode: {initial_profile['recommended_mode']}, Score: {initial_profile['performance_score']}")
    except Exception as e:
        logger.error(f"Failed to initialize system profiling cache: {e}")

@system_router.on_event("shutdown")
async def shutdown_background_tasks():
    """Cleanup on shutdown"""
    logger.info("Shutting down system profiling service...")
    # Additional cleanup if needed