"""
Activity Tracking Module

This module handles all aspects of user activity tracking and reporting
for the user management application.
"""

# activity.py - Service functions for activity tracking
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc, func
from fastapi import HTTPException

from user_management.app.db.models.user import User
from user_management.app.db.models.activity import Activity
from user_management.app.db.models.actionType import ActionType

class ActivityTrackingError(Exception):
    """Exception raised for errors in the activity tracking process."""
    pass

async def log_activity(
    user_id: int,
    action_type: ActionType,
    details: Optional[str] = None,
    session: Session = None
) -> Activity:
    """
    Log a user activity
    
    Args:
        user_id: The ID of the user performing the action
        action_type: The type of action being performed
        details: Additional details about the action (optional)
        session: Database session
        
    Returns:
        The created activity object
        
    Raises:
        ActivityTrackingError: If user is not found or other errors occur
    """
    # Validate user
    user = session.query(User).filter(User.id == user_id).first()
    if not user:
        raise ActivityTrackingError("User not found")
    
    # Create activity record
    activity = Activity(
        user_id=user_id,
        action_type=action_type,
        timestamp=datetime.utcnow(),
        details=details
    )
    
    try:
        session.add(activity)
        session.commit()
        session.refresh(activity)
        return activity
    except Exception as e:
        session.rollback()
        raise ActivityTrackingError(f"Error logging activity: {str(e)}")

async def get_user_activities(
    user_id: int,
    limit: int = 50,
    action_type: Optional[ActionType] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    session: Session = None
) -> List[Dict[str, Any]]:
    """
    Get activities for a specific user
    
    Args:
        user_id: The ID of the user
        limit: Maximum number of activities to return
        action_type: Filter by action type (optional)
        start_date: Filter activities after this date (optional)
        end_date: Filter activities before this date (optional)
        session: Database session
        
    Returns:
        A list of activity records with formatted data
        
    Raises:
        ActivityTrackingError: If user is not found
    """
    # Validate user
    user = session.query(User).filter(User.id == user_id).first()
    if not user:
        raise ActivityTrackingError("User not found")
    
    # Build query with filters
    query = session.query(Activity).filter(Activity.user_id == user_id)
    
    if action_type:
        query = query.filter(Activity.action_type == action_type)
    
    if start_date:
        query = query.filter(Activity.timestamp >= start_date)
    
    if end_date:
        query = query.filter(Activity.timestamp <= end_date)
    
    # Order by timestamp (newest first)
    activities = query.order_by(desc(Activity.timestamp)).limit(limit).all()
    
    # Format activities for response
    formatted_activities = []
    for activity in activities:
        formatted_activities.append({
            "id": activity.id,
            "action_type": activity.action_type.name,
            "timestamp": activity.timestamp.isoformat(),
            "details": activity.details
        })
    
    return formatted_activities

async def get_all_activities(
    limit: int = 50,
    action_type: Optional[ActionType] = None,
    user_id: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    session: Session = None
) -> List[Dict[str, Any]]:
    """
    Get activities for all users, with optional filtering
    
    Args:
        limit: Maximum number of activities to return
        action_type: Filter by action type (optional)
        user_id: Filter by user ID (optional)
        start_date: Filter activities after this date (optional)
        end_date: Filter activities before this date (optional)
        session: Database session
        
    Returns:
        A list of activity records with formatted data
    """
    # Build query with filters
    query = session.query(Activity)
    
    if action_type:
        query = query.filter(Activity.action_type == action_type)
    
    if user_id:
        query = query.filter(Activity.user_id == user_id)
    
    if start_date:
        query = query.filter(Activity.timestamp >= start_date)
    
    if end_date:
        query = query.filter(Activity.timestamp <= end_date)
    
    # Order by timestamp (newest first)
    activities = query.order_by(desc(Activity.timestamp)).limit(limit).all()
    
    # Format activities for response with user info
    formatted_activities = []
    for activity in activities:
        formatted_activities.append({
            "id": activity.id,
            "user_id": activity.user_id,
            "user_name": activity.user.name,
            "action_type": activity.action_type.name,
            "timestamp": activity.timestamp.isoformat(),
            "details": activity.details
        })
    
    return formatted_activities

async def get_activity_summary(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    session: Session = None
) -> Dict[str, Any]:
    """
    Get a summary of activities across the system
    
    Args:
        start_date: Start date for the summary period (optional, defaults to last 30 days)
        end_date: End date for the summary period (optional, defaults to now)
        session: Database session
        
    Returns:
        A dictionary with activity counts by type and user
    """
    # Set default date range if not provided
    if not end_date:
        end_date = datetime.utcnow()
    
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    # Get all activities in date range
    activities = session.query(Activity).filter(
        Activity.timestamp >= start_date,
        Activity.timestamp <= end_date
    ).all()
    
    # Count by action type
    action_type_counts = {}
    for action_type in ActionType:
        count = sum(1 for activity in activities if activity.action_type == action_type)
        action_type_counts[action_type.name] = count
    
    # Count by user
    user_activity_counts = {}
    for activity in activities:
        user_id = activity.user_id
        if user_id not in user_activity_counts:
            user_activity_counts[user_id] = {
                "user_id": user_id,
                "user_name": activity.user.name,
                "count": 0
            }
        user_activity_counts[user_id]["count"] += 1
    
    # Sort users by activity count (most active first)
    most_active_users = sorted(
        user_activity_counts.values(),
        key=lambda x: x["count"],
        reverse=True
    )[:10]  # Top 10 most active users
    
    # Get activity count by day
    query = session.query(
        func.date(Activity.timestamp).label('date'),
        func.count().label('count')
    ).filter(
        Activity.timestamp >= start_date,
        Activity.timestamp <= end_date
    ).group_by(
        func.date(Activity.timestamp)
    ).order_by(
        func.date(Activity.timestamp)
    )
    
    daily_counts = query.all()
    daily_activity = [
        {"date": str(day.date), "count": day.count}
        for day in daily_counts
    ]
    
    # Build summary
    summary = {
        "total_activities": len(activities),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "by_action_type": action_type_counts,
        "most_active_users": most_active_users,
        "daily_activity": daily_activity
    }
    
    return summary

async def get_user_login_history(
    user_id: int,
    limit: int = 10,
    session: Session = None
) -> List[Dict[str, Any]]:
    """
    Get login history for a specific user
    
    Args:
        user_id: The ID of the user
        limit: Maximum number of login records to return
        session: Database session
        
    Returns:
        A list of login activity records with formatted data
        
    Raises:
        ActivityTrackingError: If user is not found
    """
    # Validate user
    user = session.query(User).filter(User.id == user_id).first()
    if not user:
        raise ActivityTrackingError("User not found")
    
    # Get login activities
    login_activities = session.query(Activity).filter(
        Activity.user_id == user_id,
        Activity.action_type == ActionType.USER_LOGIN
    ).order_by(
        desc(Activity.timestamp)
    ).limit(limit).all()
    
    # Format login history for response
    login_history = []
    for activity in login_activities:
        login_history.append({
            "timestamp": activity.timestamp.isoformat(),
            "details": activity.details
        })
    
    return login_history