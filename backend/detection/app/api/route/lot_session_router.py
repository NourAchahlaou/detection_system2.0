import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from datetime import datetime, timezone

from detection.app.db.session import get_session
from detection.app.service.lotSessionService.lotSessionDashboardService import lot_dashboard_service

logger = logging.getLogger(__name__)

lot_dashboard_router = APIRouter(
    prefix="/lotSession",
    tags=["Lot Session Dashboard"],
    responses={404: {"description": "Not found"}},
)

@lot_dashboard_router.get("/data")
async def get_dashboard_data(
    group_filter: Optional[str] = Query(None, description="Filter by specific piece group"),
    search: Optional[str] = Query(None, description="Search in lot names or piece labels"),
    status_filter: Optional[str] = Query(None, description="Filter by lot status (completed, failed, running, in_progress, pending)"),
    db: Session = Depends(get_session)
):
    """
    Get comprehensive dashboard data for lot sessions grouped by piece groups
    """
    try:
        logger.info("Fetching dashboard data...")
        
        # Get full dashboard data
        dashboard_data = lot_dashboard_service.get_dashboard_data(db)
        
        if not dashboard_data['success']:
            raise HTTPException(status_code=500, detail="Failed to retrieve dashboard data")
        
        # Apply filters if provided
        grouped_lots = dashboard_data['groupedLots']
        group_stats = dashboard_data['groupStats']
        
        # Filter by group if specified
        if group_filter:
            if group_filter in grouped_lots:
                grouped_lots = {group_filter: grouped_lots[group_filter]}
                group_stats = {group_filter: group_stats[group_filter]}
            else:
                grouped_lots = {}
                group_stats = {}
        
        # Apply search filter if specified
        if search and grouped_lots:
            search_lower = search.lower()
            filtered_grouped_lots = {}
            
            for group_name, lots in grouped_lots.items():
                filtered_lots = [
                    lot for lot in lots
                    if (search_lower in lot['lotName'].lower() or 
                        search_lower in lot['expectedPiece'].lower())
                ]
                
                if filtered_lots:
                    filtered_grouped_lots[group_name] = filtered_lots
            
            grouped_lots = filtered_grouped_lots
            
            # Update group stats to match filtered data
            filtered_group_stats = {}
            for group_name in grouped_lots:
                if group_name in group_stats:
                    filtered_group_stats[group_name] = group_stats[group_name]
            group_stats = filtered_group_stats
        
        # Apply status filter if specified
        if status_filter and grouped_lots:
            status_filtered_lots = {}
            
            for group_name, lots in grouped_lots.items():
                filtered_lots = [lot for lot in lots if lot['status'] == status_filter]
                if filtered_lots:
                    status_filtered_lots[group_name] = filtered_lots
            
            grouped_lots = status_filtered_lots
            
            # Update group stats to match filtered data
            filtered_group_stats = {}
            for group_name in grouped_lots:
                if group_name in group_stats:
                    filtered_group_stats[group_name] = group_stats[group_name]
            group_stats = filtered_group_stats
        
        # Recalculate statistics based on filtered data
        all_filtered_lots = [lot for group_lots in grouped_lots.values() for lot in group_lots]
        
        if all_filtered_lots:
            total_lots = len(all_filtered_lots)
            matched_lots = len([lot for lot in all_filtered_lots if lot['isLotMatched']])
            total_sessions = sum(lot['totalSessions'] for lot in all_filtered_lots)
            successful_sessions = sum(lot['successfulSessions'] for lot in all_filtered_lots)
            
            filtered_statistics = {
                'totalGroups': len(grouped_lots),
                'totalLots': total_lots,
                'matchedLots': matched_lots,
                'lotMatchRate': round((matched_lots / total_lots * 100), 1) if total_lots > 0 else 0,
                'totalSessions': total_sessions,
                'successfulSessions': successful_sessions,
                'sessionSuccessRate': round((successful_sessions / total_sessions * 100), 1) if total_sessions > 0 else 0,
                'avgLotConfidence': round(
                    sum(lot['lotMatchConfidence'] for lot in all_filtered_lots if lot['lotMatchConfidence'] > 0) /
                    len([lot for lot in all_filtered_lots if lot['lotMatchConfidence'] > 0])
                , 1) if any(lot['lotMatchConfidence'] > 0 for lot in all_filtered_lots) else 0,
                'activeGroups': len(grouped_lots)
            }
        else:
            filtered_statistics = {
                'totalGroups': 0,
                'totalLots': 0,
                'matchedLots': 0,
                'lotMatchRate': 0,
                'totalSessions': 0,
                'successfulSessions': 0,
                'sessionSuccessRate': 0,
                'avgLotConfidence': 0,
                'activeGroups': 0
            }
        
        response_data = {
            'success': True,
            'statistics': filtered_statistics,
            'groupedLots': grouped_lots,
            'groupStats': group_stats,
            'filters_applied': {
                'group_filter': group_filter,
                'search': search,
                'status_filter': status_filter
            },
            'timestamp': dashboard_data['timestamp']
        }
        
        logger.info(f"Dashboard data retrieved successfully - {filtered_statistics['totalLots']} lots in {filtered_statistics['totalGroups']} groups")
        
        return JSONResponse(status_code=200, content=response_data)
        
    except Exception as e:
        logger.error(f"Error retrieving dashboard data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve dashboard data: {str(e)}")

@lot_dashboard_router.get("/lots/{lot_id}/details")
async def get_lot_details(
    lot_id: int,
    db: Session = Depends(get_session)
):
    """
    Get detailed information for a specific lot including all sessions and matching statistics
    """
    try:
        if lot_id <= 0:
            raise HTTPException(status_code=400, detail="Invalid lot_id")
        
        logger.info(f"Fetching details for lot {lot_id}")
        
        lot_details = lot_dashboard_service.get_lot_details(lot_id, db)
        
        if not lot_details:
            raise HTTPException(status_code=404, detail=f"Lot {lot_id} not found")
        
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': lot_details,
                'message': f"Details for lot {lot_id} retrieved successfully"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting lot details for {lot_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get lot details: {str(e)}")

@lot_dashboard_router.get("/groups/{group_name}/summary")
async def get_group_summary(
    group_name: str,
    db: Session = Depends(get_session)
):
    """
    Get summary statistics for a specific piece group including lot matching rates
    """
    try:
        logger.info(f"Fetching summary for group {group_name}")
        
        group_summary = lot_dashboard_service.get_group_summary(group_name, db)
        
        if not group_summary.get('success', False):
            raise HTTPException(status_code=404, detail=group_summary.get('message', f'Group {group_name} not found'))
        
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': group_summary,
                'message': f"Summary for group {group_name} retrieved successfully"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting group summary for {group_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get group summary: {str(e)}")

@lot_dashboard_router.get("/groups")
async def list_piece_groups(db: Session = Depends(get_session)):
    """
    Get list of all piece groups with basic statistics
    """
    try:
        logger.info("Fetching all piece groups")
        
        dashboard_data = lot_dashboard_service.get_dashboard_data(db)
        
        if not dashboard_data['success']:
            raise HTTPException(status_code=500, detail="Failed to retrieve groups data")
        
        groups_list = []
        for group_name, group_stats in dashboard_data['groupStats'].items():
            groups_list.append({
                'groupName': group_name,
                'totalLots': group_stats['totalLots'],
                'totalSessions': group_stats['totalSessions'],
                'avgSuccessRate': group_stats['avgSuccessRate'],
                'avgConfidence': group_stats['avgConfidence'],
                'lastActivity': group_stats['lastActivity']
            })
        
        # Sort by last activity (most recent first)
        groups_list.sort(key=lambda x: x['lastActivity'], reverse=True)
        
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': groups_list,
                'total_groups': len(groups_list),
                'message': f"Found {len(groups_list)} piece groups"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error listing piece groups: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list groups: {str(e)}")

@lot_dashboard_router.get("/statistics/overview")
async def get_statistics_overview(db: Session = Depends(get_session)):
    """
    Get high-level statistics for the dashboard overview
    """
    try:
        logger.info("Fetching statistics overview")
        
        dashboard_data = lot_dashboard_service.get_dashboard_data(db)
        
        if not dashboard_data['success']:
            raise HTTPException(status_code=500, detail="Failed to retrieve statistics")
        
        statistics = dashboard_data['statistics']
        
        # Add additional derived statistics
        enhanced_stats = {
            **statistics,
            'performance_status': 'good' if statistics['avgSuccessRate'] > 80 else 'fair' if statistics['avgSuccessRate'] > 60 else 'needs_attention',
            'confidence_status': 'good' if statistics['avgConfidence'] > 80 else 'fair' if statistics['avgConfidence'] > 60 else 'needs_attention',
            'activity_level': 'high' if statistics['activeGroups'] > 3 else 'medium' if statistics['activeGroups'] > 1 else 'low'
        }
        
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'data': enhanced_stats,
                'timestamp': dashboard_data['timestamp'],
                'message': "Statistics overview retrieved successfully"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting statistics overview: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

@lot_dashboard_router.get("/health")
async def dashboard_health_check():
    """
    Health check for the dashboard service
    """
    try:
        return JSONResponse(
            status_code=200,
            content={
                'success': True,
                'service': 'lot_dashboard',
                'status': 'healthy',
                'message': 'Lot dashboard service is operational',
                'timestamp': lot_dashboard_service.__dict__.get('cache', {}) is not None
            }
        )
        
    except Exception as e:
        logger.error(f"Dashboard health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                'success': False,
                'service': 'lot_dashboard',
                'status': 'unhealthy',
                'error': str(e),
                'message': 'Lot dashboard service is not operational'
            }
        )