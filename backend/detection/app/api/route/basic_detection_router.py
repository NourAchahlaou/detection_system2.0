# basic_detection_router.py - Complete router for on-demand detection
import cv2
import logging
import numpy as np
import aiohttp
from typing import Optional, Dict, Any
import time
import base64
from dataclasses import dataclass, asdict
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
import asyncio

from detection.app.service.alternative.basic_detection_service import (
    DetectionRequest, 
    DetectionResponse,
    basic_detection_processor
)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the router
basic_detection_router = APIRouter(
    prefix="/basic",
    tags=["Basic Detection"],
    responses={404: {"description": "Not found"}},
)

@basic_detection_router.post("/detect/{camera_id}")
async def detect_single_frame(
    camera_id: int,
    request: Request
):
    """
    Perform detection on a single frame from the specified camera
    
    This endpoint is designed for low-performance systems:
    - Captures one frame from the camera
    - Runs detection on that frame
    - Returns the frame with overlays and detection results
    
    Body should contain:
    {
        "target_label": "person",
        "quality": 85  // optional, JPEG quality for response
    }
    """
    try:
        # Parse request body
        body = await request.json()
        target_label = body.get('target_label')
        quality = body.get('quality', 85)
        
        if not target_label:
            raise HTTPException(status_code=400, detail="target_label is required")
        
        # Validate camera_id
        if camera_id < 0:
            raise HTTPException(status_code=400, detail="Invalid camera_id")
        
        # Validate quality
        if not isinstance(quality, int) or quality < 50 or quality > 100:
            quality = 85
        
        logger.info(f"🎯 Detection request for camera {camera_id}, target: '{target_label}'")
        
        # Create detection request
        detection_request = DetectionRequest(
            camera_id=camera_id,
            target_label=target_label,
            timestamp=time.time(),
            quality=quality
        )
        
        # Perform detection
        response = await basic_detection_processor.detect_on_frame(detection_request)
        
        # Convert dataclass to dict for JSON response
        response_dict = asdict(response)
        
        # Return JSON response
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "data": response_dict,
                "message": "Detection completed successfully"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing detection request: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@basic_detection_router.get("/health")
async def health_check():
    """
    Check if the basic detection service is healthy and ready
    """
    try:
        # Check if processor is initialized
        stats = basic_detection_processor.get_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy" if stats['is_initialized'] else "initializing",
                "is_initialized": stats['is_initialized'],
                "device": stats.get('device', 'unknown'),
                "detections_performed": stats.get('detections_performed', 0),
                "avg_processing_time": stats.get('avg_processing_time', 0),
                "message": "Basic detection service is operational"
            }
        )
    except Exception as e:
        logger.error(f"❌ Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "message": "Basic detection service is not operational"
            }
        )

@basic_detection_router.post("/initialize")
async def initialize_processor():
    """
    Initialize the basic detection processor
    """
    try:
        logger.info("🚀 Initializing basic detection processor...")
        
        await basic_detection_processor.initialize()
        
        stats = basic_detection_processor.get_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "message": "Basic detection processor initialized successfully",
                "device": stats.get('device', 'unknown'),
                "is_initialized": stats['is_initialized']
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error initializing processor: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Failed to initialize processor: {str(e)}"
        )

@basic_detection_router.get("/stats")
async def get_detection_stats():
    """
    Get basic detection service statistics
    """
    try:
        stats = basic_detection_processor.get_stats()
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "stats": stats,
                "timestamp": time.time()
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")

@basic_detection_router.post("/test/{camera_id}")
async def test_camera_frame(camera_id: int):
    """
    Test endpoint to capture a frame from camera without detection
    Useful for testing camera connectivity
    """
    try:
        if camera_id < 0:
            raise HTTPException(status_code=400, detail="Invalid camera_id")
        
        logger.info(f"📸 Testing frame capture from camera {camera_id}")
        
        # Get frame without detection
        frame = await basic_detection_processor.get_frame_from_camera(camera_id)
        
        if frame is None:
            raise HTTPException(
                status_code=404, 
                detail=f"Could not capture frame from camera {camera_id}"
            )
        
        # Encode frame as base64
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, 85]
        success, buffer = cv2.imencode('.jpg', frame, encode_params)
        
        if not success:
            raise HTTPException(
                status_code=500, 
                detail="Failed to encode captured frame"
            )
        
        frame_b64 = base64.b64encode(buffer.tobytes()).decode('utf-8')
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "camera_id": camera_id,
                "frame_shape": frame.shape,
                "frame_data": frame_b64,
                "message": f"Successfully captured frame from camera {camera_id}"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error testing camera frame: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Camera test failed: {str(e)}"
        )

@basic_detection_router.post("/detect/batch")
async def detect_multiple_frames(request: Request):
    """
    Perform detection on multiple cameras/targets in a single request
    
    Body should contain:
    {
        "detections": [
            {
                "camera_id": 0,
                "target_label": "person",
                "quality": 85
            },
            {
                "camera_id": 1,
                "target_label": "car",
                "quality": 90
            }
        ]
    }
    """
    try:
        body = await request.json()
        detections = body.get('detections', [])
        
        if not detections or not isinstance(detections, list):
            raise HTTPException(
                status_code=400, 
                detail="detections array is required"
            )
        
        if len(detections) > 5:  # Limit batch size
            raise HTTPException(
                status_code=400, 
                detail="Maximum 5 detections per batch"
            )
        
        logger.info(f"🎯 Batch detection request for {len(detections)} items")
        
        results = []
        
        for i, detection_data in enumerate(detections):
            try:
                camera_id = detection_data.get('camera_id')
                target_label = detection_data.get('target_label')
                quality = detection_data.get('quality', 85)
                
                if camera_id is None or not target_label:
                    results.append({
                        "index": i,
                        "success": False,
                        "error": "camera_id and target_label are required"
                    })
                    continue
                
                # Create detection request
                detection_request = DetectionRequest(
                    camera_id=camera_id,
                    target_label=target_label,
                    timestamp=time.time(),
                    quality=quality
                )
                
                # Perform detection
                response = await basic_detection_processor.detect_on_frame(detection_request)
                
                results.append({
                    "index": i,
                    "success": True,
                    "data": asdict(response)
                })
                
            except Exception as e:
                logger.error(f"❌ Error in batch detection item {i}: {e}")
                results.append({
                    "index": i,
                    "success": False,
                    "error": str(e)
                })
        
        return JSONResponse(
            status_code=200,
            content={
                "success": True,
                "results": results,
                "total_processed": len(results),
                "successful": sum(1 for r in results if r['success']),
                "failed": sum(1 for r in results if not r['success'])
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing batch detection: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Batch detection failed: {str(e)}"
        )
