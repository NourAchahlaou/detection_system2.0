# app/response/camera.py
from pydantic import BaseModel
from typing import List, Optional, Dict, Any


class CameraResponse(BaseModel):
    """Response model for camera information"""
    camera_id: str
    camera_type: str  # "opencv" or "basler"
    name: Optional[str] = None
    status: str  # "available", "in_use", "error"
    additional_info: Optional[Dict[str, Any]] = None


class CameraStopResponse(BaseModel):
    """Response model for stopping a camera"""
    message: str


class CameraStatusResponse(BaseModel):
    """Response model for checking camera status"""
    camera_opened: bool
    circuit_breaker_active: Optional[bool] = None


class CircuitBreakerStatusResponse(BaseModel):
    """Response model for circuit breaker status"""
    state: str  # "CLOSED", "OPEN", or "HALF_OPEN"
    failure_count: int
    last_failure_time: float