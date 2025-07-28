# system_profiling_router.py - Artifact Keeper Service Router
from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
import logging
import requests
from datetime import datetime
from typing import Dict, Any, Optional
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
    "last_updated": None,
    "cache_duration": 30  # 30 seconds as per your JS service
}

class HardwareServiceClient:
    """Client for communicating with the Hardware Service"""
    
    def __init__(self, base_url: str = HARDWARE_SERVICE_URL):
        self.base_url = base_url.rstrip('/')
        
    async def get_system_specs(self) -> Dict[str, Any]:
        """Get system specifications from hardware service"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.base_url}/system/specs")
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.error(f"Failed to get system specs: {e}")
            raise HTTPException(status_code=503, detail=f"Hardware service unavailable: {str(e)}")
    
    async def get_performance_recommendation(self, include_runtime_test: bool = False) -> Dict[str, Any]:
        """Get performance recommendation from hardware service"""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=30.0) as client:
                params = {"include_runtime_test": include_runtime_test}
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
        
        # Calculate avg_fps based on system capabilities (estimation)
        avg_fps = 30  # Default assumption
        if gpu.get("available", False):
            avg_fps = 30
        elif cpu.get("cores", 2) >= 4:
            avg_fps = 20
        else:
            avg_fps = 15
            
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
            "performance_score": hardware_data.get("performance_score", 30),
            "avg_fps": avg_fps,
            "current_cpu_load": cpu.get("current_load_percent", 50),
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
            "avg_fps": 15,
            "current_cpu_load": 50,
            "timestamp": datetime.now().isoformat()
        }

@system_router.get("/profile")
async def get_system_profile(
    force_refresh: bool = False,
    hardware_client: HardwareServiceClient = Depends(get_hardware_client)
):
    """
    Get system performance profile - Main endpoint used by frontend
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

@system_router.get("/performance/mode")
async def get_performance_mode_detailed(
    hardware_client: HardwareServiceClient = Depends(get_hardware_client)
):
    """
    Get detailed performance mode information
    """
    try:
        mode_data = await hardware_client.get_performance_mode()
        
        if mode_data.get("status") != "success":
            raise HTTPException(status_code=503, detail="Hardware service returned error")
        
        mode = mode_data.get("performance_mode", "low")
        mode_details = mode_data.get("mode_details", {})
        
        return {
            "mode": mode,
            "performance_score": mode_data.get("performance_score", 30),
            "high_performance": mode == "high",
            "real_time_detection": mode_details.get("real_time_detection", False),
            "on_demand_detection": mode_details.get("on_demand_detection", True),
            "description": mode_details.get("description", "Basic detection mode"),
            "timestamp": mode_data.get("timestamp", datetime.now().isoformat())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting performance mode: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get performance mode: {str(e)}")

@system_router.get("/recommendation")
async def get_performance_recommendation(
    include_runtime_test: bool = False,
    hardware_client: HardwareServiceClient = Depends(get_hardware_client)
):
    """
    Get comprehensive performance recommendation
    """
    try:
        recommendation_data = await hardware_client.get_performance_recommendation(include_runtime_test)
        
        if recommendation_data.get("status") != "success":
            raise HTTPException(status_code=503, detail="Hardware service returned error")
        
        recommendation = recommendation_data.get("recommendation", {})
        
        return {
            "recommended_mode": recommendation.get("final_recommendation", "low"),
            "performance_score": recommendation.get("performance_score", 30),
            "system_specs": recommendation.get("system_specs", {}),
            "reasoning": recommendation.get("reasoning", {}),
            "thresholds": recommendation.get("thresholds", {}),
            "runtime_performance": recommendation.get("runtime_performance") if include_runtime_test else None,
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
    Force refresh the system profile cache
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

@system_router.get("/capabilities")
async def get_system_capabilities(
    hardware_client: HardwareServiceClient = Depends(get_hardware_client)
):
    """
    Get system capabilities based on current profile
    """
    try:
        profile = await get_system_profile(hardware_client=hardware_client)
        
        # Determine capabilities based on profile
        is_high_performance = (
            profile["cpu_cores"] >= 4 and
            profile["total_memory_gb"] >= 8 and
            (profile["gpu_available"] or profile["performance_score"] >= 70) and
            profile["avg_fps"] >= 20
        )
        
        if is_high_performance:
            capabilities = {
                "mode": "high",
                "real_time_detection": True,
                "continuous_processing": True,
                "max_concurrent_streams": min(profile["cpu_cores"] // 2, 4),
                "detection_fps": "20-30",
                "max_resolution": "1080p" if profile["gpu_available"] else "720p",
                "ai_models": ["YOLO v8", "YOLO v9"] if profile["gpu_available"] else ["YOLO v5"],
                "memory_requirement_mb": "2000-4000"
            }
        else:
            capabilities = {
                "mode": "basic",
                "real_time_detection": False,
                "on_demand_detection": True,
                "max_concurrent_streams": 1,
                "detection_mode": "manual_trigger",
                "max_resolution": "720p",
                "ai_models": ["YOLO v5 optimized"],
                "memory_requirement_mb": "500-1500"
            }
        
        return {
            "status": "success",
            "system_profile": profile,
            "capabilities": capabilities,
            "performance_mode": capabilities["mode"],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error getting capabilities: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get capabilities: {str(e)}")

@system_router.get("/health")
async def system_health_check():
    """
    Health check for the system profiling service
    """
    try:
        # Check if we can reach the hardware service
        hardware_client = HardwareServiceClient()
        
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{hardware_client.base_url}/system/health")
                hardware_service_healthy = response.status_code == 200
        except:
            hardware_service_healthy = False
        
        return {
            "status": "healthy" if hardware_service_healthy else "degraded",
            "service": "artifact_keeper_system_profiling",
            "hardware_service_available": hardware_service_healthy,
            "cache_status": {
                "has_cached_profile": _performance_cache["profile"] is not None,
                "last_updated": _performance_cache["last_updated"].isoformat() if _performance_cache["last_updated"] else None
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "artifact_keeper_system_profiling",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

# Background task to keep cache warm
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

@system_router.on_event("startup")
async def start_background_tasks():
    """Start background tasks"""
    asyncio.create_task(background_cache_warmer())

# Additional endpoints that might be useful for your frontend

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

@system_router.post("/performance/test")
async def run_performance_test(
    duration_seconds: int = 10,
    hardware_client: HardwareServiceClient = Depends(get_hardware_client)
):
    """
    Run a performance test through the hardware service
    """
    try:
        import httpx
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{hardware_client.base_url}/system/performance/test",
                params={"duration_seconds": min(duration_seconds, 30)}
            )
            response.raise_for_status()
            return response.json()
            
    except Exception as e:
        logger.error(f"Performance test failed: {e}")
        raise HTTPException(status_code=500, detail=f"Performance test failed: {str(e)}")