# detection_stats_routes.py - FastAPI routes for DetectionStatisticsService

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, List
import logging

from detection.app.db.session import get_session
from detection.app.service.alternative.statistics_service import detection_statistics_service

logger = logging.getLogger(__name__)

statistic_detection_router = APIRouter(
    prefix="/basic/statistics",
    tags=["Basic Detection Statistics with Database"],
    responses={404: {"description": "Not found"}},
)

# --- Last session for each lot ---
@statistic_detection_router.get("/lots/last-sessions")
async def get_last_sessions_per_lot(db: Session = Depends(get_session)):
    """Get last detection session for each lot with brief metrics"""
    try:
        data = detection_statistics_service.last_session_per_lot(db)
        return {
            "success": True,
            "message": "Last sessions per lot retrieved successfully",
            "data": [d.__dict__ for d in data]
        }
    except Exception as e:
        logger.exception("Error in get_last_sessions_per_lot")
        raise HTTPException(status_code=500, detail=str(e))


# --- Lot summary ---
@statistic_detection_router.get("/lots/{lot_id}/summary")
async def get_lot_summary(lot_id: int, db: Session = Depends(get_session)):
    """Get summary statistics for a specific lot"""
    try:
        summary = detection_statistics_service.lot_summary(lot_id, db)
        if not summary:
            raise HTTPException(status_code=404, detail=f"Lot {lot_id} not found")
        return {
            "success": True,
            "message": f"Summary for lot {lot_id} retrieved successfully",
            "data": summary.__dict__
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Error in get_lot_summary")
        raise HTTPException(status_code=500, detail=str(e))


# --- Sessions to completion ---
@statistic_detection_router.get("/lots/{lot_id}/sessions-to-completion")
async def get_sessions_to_completion(lot_id: int, db: Session = Depends(get_session)):
    """Get the number of sessions required to complete a lot"""
    try:
        count = detection_statistics_service.sessions_to_completion(lot_id, db)
        return {
            "success": True,
            "message": f"Sessions to completion for lot {lot_id} retrieved successfully",
            "data": count
        }
    except Exception as e:
        logger.exception("Error in get_sessions_to_completion")
        raise HTTPException(status_code=500, detail=str(e))


# --- System start statistics ---
@statistic_detection_router.get("/system/start-stats")
async def get_system_start_stats(db: Session = Depends(get_session)):
    """Get stats on lots correctness from their first session"""
    try:
        stats = detection_statistics_service.system_start_stats(db)
        return {
            "success": True,
            "message": "System start statistics retrieved successfully",
            "data": stats.__dict__
        }
    except Exception as e:
        logger.exception("Error in get_system_start_stats")
        raise HTTPException(status_code=500, detail=str(e))


# --- Common failures for problem lots ---
@statistic_detection_router.get("/system/common-failures")
async def get_common_failures(top_n: int = Query(10, ge=1), db: Session = Depends(get_session)):
    """Get most common failure categories for problem lots"""
    try:
        failures = detection_statistics_service.common_failures_for_problem_lots(db, top_n=top_n)
        return {
            "success": True,
            "message": f"Top {top_n} common failures retrieved successfully",
            "data": [f.__dict__ for f in failures]
        }
    except Exception as e:
        logger.exception("Error in get_common_failures")
        raise HTTPException(status_code=500, detail=str(e))


# --- Top mixed-up piece pairs ---
@statistic_detection_router.get("/system/top-mixed-pairs")
async def get_top_mixed_pairs(top_n: int = Query(10, ge=1), db: Session = Depends(get_session)):
    """Get most confused piece pairs"""
    try:
        pairs = detection_statistics_service.top_mixed_pairs(db, top_n=top_n)
        return {
            "success": True,
            "message": f"Top {top_n} mixed-up pairs retrieved successfully",
            "data": [p.__dict__ for p in pairs]
        }
    except Exception as e:
        logger.exception("Error in get_top_mixed_pairs")
        raise HTTPException(status_code=500, detail=str(e))


# --- Clear statistics cache ---
@statistic_detection_router.delete("/cache")
async def clear_statistics_cache():
    """Clear all cached statistics"""
    try:
        detection_statistics_service.clear_cache()
        return {"success": True, "message": "Statistics cache cleared successfully"}
    except Exception as e:
        logger.exception("Error in clear_statistics_cache")
        raise HTTPException(status_code=500, detail=str(e))


# --- Health check ---
@statistic_detection_router.get("/health")
async def statistics_health_check():
    """Health check for statistics service"""
    try:
        cache_size = len(detection_statistics_service._cache)
        return {
            "success": True,
            "message": "Statistics service is healthy",
            "data": {
                "service_status": "healthy",
                "cache_entries": cache_size,
                "cache_timeout": detection_statistics_service.cache_timeout
            }
        }
    except Exception as e:
        logger.exception("Error in statistics_health_check")
        raise HTTPException(status_code=503, detail=str(e))
