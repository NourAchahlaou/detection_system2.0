# detection_stats_routes.py - FastAPI routes for detection statistics

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Optional, List
import logging

from detection.app.db.session import get_session
from detection.app.service.alternative.statistics_service import detection_statistics_service

logger = logging.getLogger(__name__)

# Create the router
statistic_detection_router = APIRouter(
    prefix="/basic/statistics",
    tags=["Basic Detection statistic with Database"],
    responses={404: {"description": "Not found"}},
)


@statistic_detection_router.get("/overview")
async def get_detection_overview(db: Session = Depends(get_session)):
    """Get overall detection system overview and statistics"""
    try:
        logger.info("📊 Getting detection overview...")
        
        overview = detection_statistics_service.get_detection_overview(db)
        
        return {
            "success": True,
            "message": "Detection overview retrieved successfully",
            "data": overview
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting detection overview: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get detection overview: {str(e)}"
        )

@statistic_detection_router.get("/lots/{lot_id}/progress")
async def get_lot_progress(
    lot_id: int,
    db: Session = Depends(get_session)
):
    """Get detailed progress information for a specific detection lot"""
    try:
        logger.info(f"📊 Getting progress for lot {lot_id}...")
        
        progress = detection_statistics_service.get_lot_progress(lot_id, db)
        
        if not progress:
            raise HTTPException(
                status_code=404,
                detail=f"Lot {lot_id} not found"
            )
        
        return {
            "success": True,
            "message": f"Lot {lot_id} progress retrieved successfully",
            "data": progress
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting lot {lot_id} progress: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get lot progress: {str(e)}"
        )

@statistic_detection_router.get("/pieces")
async def get_piece_statistics(
    piece_id: Optional[int] = Query(None, description="Specific piece ID to get statistics for"),
    db: Session = Depends(get_session)
):
    """Get statistics for detected pieces"""
    try:
        logger.info(f"📊 Getting piece statistics{f' for piece {piece_id}' if piece_id else ' for all pieces'}...")
        
        piece_stats = detection_statistics_service.get_piece_statistics(piece_id, db)
        
        return {
            "success": True,
            "message": f"Piece statistics retrieved successfully",
            "data": piece_stats,
            "total_pieces": len(piece_stats)
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting piece statistics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get piece statistics: {str(e)}"
        )

@statistic_detection_router.get("/realtime")
async def get_real_time_stats(
    camera_id: Optional[int] = Query(None, description="Specific camera ID to get stats for"),
    db: Session = Depends(get_session)
):
    """Get real-time detection statistics"""
    try:
        logger.info(f"📊 Getting real-time stats{f' for camera {camera_id}' if camera_id else ''}...")
        
        realtime_stats = detection_statistics_service.get_real_time_stats(camera_id, db)
        
        return {
            "success": True,
            "message": "Real-time statistics retrieved successfully",
            "data": realtime_stats
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting real-time stats: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get real-time statistics: {str(e)}"
        )

@statistic_detection_router.get("/lots/{lot_id}/timeline")
async def get_lot_completion_timeline(
    lot_id: int,
    db: Session = Depends(get_session)
):
    """Get timeline of detections for a specific lot"""
    try:
        logger.info(f"📊 Getting completion timeline for lot {lot_id}...")
        
        timeline = detection_statistics_service.get_lot_completion_timeline(lot_id, db)
        
        return {
            "success": True,
            "message": f"Lot {lot_id} completion timeline retrieved successfully",
            "data": timeline,
            "total_sessions": len(timeline)
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting lot {lot_id} timeline: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get lot completion timeline: {str(e)}"
        )

@statistic_detection_router.get("/analytics")
async def get_performance_analytics(
    days: int = Query(7, ge=1, le=365, description="Number of days to analyze (1-365)"),
    db: Session = Depends(get_session)
):
    """Get performance analytics for the specified number of days"""
    try:
        logger.info(f"📊 Getting performance analytics for last {days} days...")
        
        analytics = detection_statistics_service.get_performance_analytics(db, days)
        
        return {
            "success": True,
            "message": f"Performance analytics for last {days} days retrieved successfully",
            "data": analytics
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting performance analytics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get performance analytics: {str(e)}"
        )

@statistic_detection_router.get("/lots/all/progress")
async def get_all_lots_progress(
    limit: Optional[int] = Query(50, ge=1, le=1000, description="Maximum number of lots to return"),
    offset: Optional[int] = Query(0, ge=0, description="Number of lots to skip"),
    completed_only: Optional[bool] = Query(False, description="Return only completed lots"),
    active_only: Optional[bool] = Query(False, description="Return only active lots"),
    db: Session = Depends(get_session)
):
    """Get progress information for multiple lots"""
    try:
        logger.info(f"📊 Getting progress for lots (limit: {limit}, offset: {offset})...")
        
        # Import the DetectionLot model to query lots
        from detection.app.db.models.detectionLot import DetectionLot
        
        # Build query based on filters
        query = db.query(DetectionLot)
        
        if completed_only and active_only:
            raise HTTPException(
                status_code=400,
                detail="Cannot filter by both completed_only and active_only"
            )
        
        if completed_only:
            query = query.filter(DetectionLot.is_target_match == True)
        elif active_only:
            query = query.filter(DetectionLot.is_target_match == False)
        
        # Get total count before applying limit/offset
        total_count = query.count()
        
        # Apply pagination
        lots = query.offset(offset).limit(limit).all()
        
        # Get progress for each lot
        lots_progress = []
        for lot in lots:
            progress = detection_statistics_service.get_lot_progress(lot.id, db)
            if progress:
                lots_progress.append(progress)
        
        return {
            "success": True,
            "message": f"Retrieved progress for {len(lots_progress)} lots",
            "data": lots_progress,
            "pagination": {
                "total_count": total_count,
                "returned_count": len(lots_progress),
                "limit": limit,
                "offset": offset,
                "has_more": (offset + limit) < total_count
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error getting all lots progress: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get lots progress: {str(e)}"
        )

@statistic_detection_router.get("/summary")
async def get_detection_summary(
    include_realtime: bool = Query(True, description="Include real-time statistics"),
    include_analytics: bool = Query(True, description="Include performance analytics"),
    analytics_days: int = Query(7, ge=1, le=30, description="Days for analytics (1-30)"),
    db: Session = Depends(get_session)
):
    """Get comprehensive detection system summary"""
    try:
        logger.info("📊 Getting comprehensive detection summary...")
        
        # Get basic overview
        overview = detection_statistics_service.get_detection_overview(db)
        
        summary = {
            "overview": overview,
            "timestamp": "2025-01-01T00:00:00Z"  # Replace with actual timestamp
        }
        
        # Include real-time stats if requested
        if include_realtime:
            realtime_stats = detection_statistics_service.get_real_time_stats(None, db)
            summary["realtime"] = realtime_stats
        
        # Include analytics if requested
        if include_analytics:
            analytics = detection_statistics_service.get_performance_analytics(db, analytics_days)
            summary["analytics"] = analytics
        
        return {
            "success": True,
            "message": "Detection system summary retrieved successfully",
            "data": summary
        }
        
    except Exception as e:
        logger.error(f"❌ Error getting detection summary: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get detection summary: {str(e)}"
        )

@statistic_detection_router.delete("/cache")
async def clear_statistics_cache():
    """Clear all cached statistics data"""
    try:
        logger.info("🧹 Clearing statistics cache...")
        
        detection_statistics_service.clear_cache()
        
        return {
            "success": True,
            "message": "Statistics cache cleared successfully"
        }
        
    except Exception as e:
        logger.error(f"❌ Error clearing statistics cache: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear cache: {str(e)}"
        )

@statistic_detection_router.get("/health")
async def statistics_health_check():
    """Health check endpoint for statistics service"""
    try:
        # Simple health check - try to access the service
        cache_size = len(detection_statistics_service.cache)
        
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
        logger.error(f"❌ Statistics service health check failed: {e}")
        raise HTTPException(
            status_code=503,
            detail=f"Statistics service unhealthy: {str(e)}"
        )
