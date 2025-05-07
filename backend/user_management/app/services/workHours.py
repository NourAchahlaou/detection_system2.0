"""
Work Hours Module

This module handles tracking and management of user work hours
for the user management application.
"""

# work_hours.py - Service functions for work hours management
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import desc, and_
from fastapi import HTTPException

from user_management.app.db.models.user import User
from user_management.app.db.models.workHours import WorkHours
from user_management.app.db.models.actionType import ActionType
from user_management.app.services.activity import log_activity

class WorkHoursError(Exception):
    """Exception raised for errors in the work hours management process."""
    pass

async def log_user_login(user_id: int, session: Session) -> WorkHours:
    """
    Log a user login and create a work hours entry
    
    Args:
        user_id: The ID of the user logging in
        session: Database session
        
    Returns:
        The created work hours object
        
    Raises:
        WorkHoursError: If user is not found or has an active session
    """
    # Validate user
    user = session.query(User).filter(User.id == user_id).first()
    if not user:
        raise WorkHoursError("User not found")
    
    # Check if user already has an active session
    active_session = session.query(WorkHours).filter(
        WorkHours.user_id == user_id,
        WorkHours.logout_time == None
    ).first()
    
    if active_session:
        raise WorkHoursError("User already has an active work session")
    
    # Create work hours entry
    work_hours = WorkHours(
        user_id=user_id,
        login_time=datetime.utcnow(),
        logout_time=None,
        total_minutes=None
    )
    
    try:
        session.add(work_hours)
        session.commit()
        session.refresh(work_hours)
        
        # Log activity
        await log_activity(
            user_id=user_id,
            action_type=ActionType.USER_LOGIN,
            details="User logged in",
            session=session
        )
        
        return work_hours
    except Exception as e:
        session.rollback()
        raise WorkHoursError(f"Error logging user login: {str(e)}")

async def log_user_logout(user_id: int, session: Session) -> WorkHours:
    """
    Log a user logout and update the active work hours entry
    
    Args:
        user_id: The ID of the user logging out
        session: Database session
        
    Returns:
        The updated work hours object
        
    Raises:
        WorkHoursError: If user is not found or has no active session
    """
    # Validate user
    user = session.query(User).filter(User.id == user_id).first()
    if not user:
        raise WorkHoursError("User not found")
    
    # Find active session
    active_session = session.query(WorkHours).filter(
        WorkHours.user_id == user_id,
        WorkHours.logout_time == None
    ).first()
    
    if not active_session:
        raise WorkHoursError("User has no active work session")
    
    # Update work hours entry
    logout_time = datetime.utcnow()
    duration = logout_time - active_session.login_time
    total_minutes = int(duration.total_seconds() / 60)
    
    active_session.logout_time = logout_time
    active_session.total_minutes = total_minutes
    
    try:
        session.commit()
        session.refresh(active_session)
        
        # Log activity
        await log_activity(
            user_id=user_id,
            action_type=ActionType.USER_LOGOUT,
            details=f"User logged out. Session duration: {total_minutes} minutes",
            session=session
        )
        
        return active_session
    except Exception as e:
        session.rollback()
        raise WorkHoursError(f"Error logging user logout: {str(e)}")

async def get_user_work_sessions(
    user_id: int,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    limit: int = 50,
    session: Session = None
) -> List[Dict[str, Any]]:
    """
    Get work sessions for a specific user
    
    Args:
        user_id: The ID of the user
        start_date: Filter sessions after this date (optional)
        end_date: Filter sessions before this date (optional)
        limit: Maximum number of sessions to return
        session: Database session
        
    Returns:
        A list of work session records with formatted data
        
    Raises:
        WorkHoursError: If user is not found
    """
    # Validate user
    user = session.query(User).filter(User.id == user_id).first()
    if not user:
        raise WorkHoursError("User not found")
    
    # Build query with filters
    query = session.query(WorkHours).filter(WorkHours.user_id == user_id)
    
    if start_date:
        query = query.filter(WorkHours.login_time >= start_date)
    
    if end_date:
        query = query.filter(WorkHours.login_time <= end_date)
    
    # Order by login time (newest first)
    work_sessions = query.order_by(desc(WorkHours.login_time)).limit(limit).all()
    
    # Format work sessions for response
    formatted_sessions = []
    for work_session in work_sessions:
        session_data = {
            "id": work_session.id,
            "login_time": work_session.login_time.isoformat(),
            "logout_time": work_session.logout_time.isoformat() if work_session.logout_time else None,
            "duration_minutes": work_session.total_minutes,
            "duration_hours": round(work_session.total_minutes / 60, 2) if work_session.total_minutes else None,
            "is_active": work_session.logout_time is None
        }
        formatted_sessions.append(session_data)
    
    return formatted_sessions

async def get_current_active_sessions(session: Session) -> List[Dict[str, Any]]:
    """
    Get all currently active work sessions
    
    Args:
        session: Database session
        
    Returns:
        A list of active session records with user info
    """
    # Find all active sessions
    active_sessions = session.query(WorkHours).filter(
        WorkHours.logout_time == None
    ).all()
    
    # Format active sessions for response
    formatted_sessions = []
    for work_session in active_sessions:
        # Calculate current duration
        current_time = datetime.utcnow()
        duration = current_time - work_session.login_time
        current_minutes = int(duration.total_seconds() / 60)
        
        session_data = {
            "id": work_session.id,
            "user_id": work_session.user_id,
            "user_name": work_session.user.name,
            "login_time": work_session.login_time.isoformat(),
            "current_duration_minutes": current_minutes,
            "current_duration_hours": round(current_minutes / 60, 2)
        }
        formatted_sessions.append(session_data)
    
    return formatted_sessions

async def check_user_session_status(user_id: int, session: Session) -> Dict[str, Any]:
    """
    Check if a user has an active work session
    
    Args:
        user_id: The ID of the user to check
        session: Database session
        
    Returns:
        A dictionary with session status information
        
    Raises:
        WorkHoursError: If user is not found
    """
    # Validate user
    user = session.query(User).filter(User.id == user_id).first()
    if not user:
        raise WorkHoursError("User not found")
    
    # Find active session if any
    active_session = session.query(WorkHours).filter(
        WorkHours.user_id == user_id,
        WorkHours.logout_time == None
    ).first()
    
    if active_session:
        # Calculate current duration
        current_time = datetime.utcnow()
        duration = current_time - active_session.login_time
        current_minutes = int(duration.total_seconds() / 60)
        
        return {
            "has_active_session": True,
            "session_id": active_session.id,
            "login_time": active_session.login_time.isoformat(),
            "current_duration_minutes": current_minutes,
            "current_duration_hours": round(current_minutes / 60, 2)
        }
    else:
        # Get last session if any
        last_session = session.query(WorkHours).filter(
            WorkHours.user_id == user_id
        ).order_by(desc(WorkHours.logout_time)).first()
        
        if last_session:
            return {
                "has_active_session": False,
                "last_session_id": last_session.id,
                "last_logout_time": last_session.logout_time.isoformat() if last_session.logout_time else None,
                "last_session_duration_minutes": last_session.total_minutes
            }
        else:
            return {
                "has_active_session": False,
                "last_session_id": None
            }

async def get_weekly_work_summary(
    user_id: int, 
    week_start_date: Optional[datetime] = None,
    session: Session = None
) -> Dict[str, Any]:
    """
    Get a weekly summary of work hours for a user
    
    Args:
        user_id: The ID of the user
        week_start_date: The starting date of the week (optional, defaults to previous Monday)
        session: Database session
        
    Returns:
        A dictionary with weekly work hours data
        
    Raises:
        WorkHoursError: If user is not found
    """
    # Validate user
    user = session.query(User).filter(User.id == user_id).first()
    if not user:
        raise WorkHoursError("User not found")
    
    # Set default week start date if not provided (previous Monday)
    if not week_start_date:
        today = datetime.utcnow().date()
        days_since_monday = today.weekday()
        week_start_date = datetime.combine(today - timedelta(days=days_since_monday), datetime.min.time())
    
    # Calculate week end date (Sunday)
    week_end_date = week_start_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
    
    # Get work hours for the week
    work_hours = session.query(WorkHours).filter(
        WorkHours.user_id == user_id,
        WorkHours.login_time >= week_start_date,
        WorkHours.login_time <= week_end_date,
        WorkHours.total_minutes != None  # Only include completed sessions
    ).all()
    
    # Calculate hours by day of week
    daily_hours = {i: 0 for i in range(7)}  # 0=Monday, 6=Sunday
    
    for wh in work_hours:
        day_of_week = wh.login_time.weekday()
        daily_hours[day_of_week] += (wh.total_minutes or 0) / 60
    
    # Format daily hours
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    formatted_daily_hours = [
        {
            "day": day_names[i],
            "hours": round(hours, 2)
        }
        for i, hours in daily_hours.items()
    ]
    
    # Calculate total and average
    total_hours = sum(daily_hours.values())
    workdays_count = sum(1 for i, hours in daily_hours.items() if i < 5 and hours > 0)  # Count days with hours Mon-Fri
    avg_workday_hours = total_hours / workdays_count if workdays_count > 0 else 0
    
    # Build weekly summary
    weekly_summary = {
        "user_id": user_id,
        "user_name": user.name,
        "week_start_date": week_start_date.date().isoformat(),
        "week_end_date": week_end_date.date().isoformat(),
        "total_hours": round(total_hours, 2),
        "workdays_logged": workdays_count,
        "average_workday_hours": round(avg_workday_hours, 2),
        "daily_hours": formatted_daily_hours
    }
    
    return weekly_summary

async def adjust_work_hours(
    work_hours_id: int,
    login_time: Optional[datetime] = None,
    logout_time: Optional[datetime] = None,
    adjusted_by_id: int = None,
    reason: str = None,
    session: Session = None
) -> WorkHours:
    """
    Adjust a work hours record (for corrections or administrative purposes)
    
    Args:
        work_hours_id: The ID of the work hours record to adjust
        login_time: The adjusted login time (optional)
        logout_time: The adjusted logout time (optional)
        adjusted_by_id: The ID of the user making the adjustment
        reason: The reason for the adjustment
        session: Database session
        
    Returns:
        The updated work hours object
        
    Raises:
        WorkHoursError: If work hours record is not found or other errors occur
    """
    # Find work hours record
    work_hours = session.query(WorkHours).filter(WorkHours.id == work_hours_id).first()
    
    if not work_hours:
        raise WorkHoursError("Work hours record not found")
    
    # Track changes for activity log
    changes = []
    
    # Update fields if provided
    if login_time is not None and login_time != work_hours.login_time:
        old_login = work_hours.login_time
        work_hours.login_time = login_time
        changes.append(f"login time: {old_login.isoformat()} -> {login_time.isoformat()}")
    
    if logout_time is not None and logout_time != work_hours.logout_time:
        old_logout = work_hours.logout_time
        work_hours.logout_time = logout_time
        changes.append(f"logout time: {old_logout.isoformat() if old_logout else 'None'} -> {logout_time.isoformat()}")
    
    # Recalculate total minutes if both login and logout time are set
    if work_hours.login_time and work_hours.logout_time:
        duration = work_hours.logout_time - work_hours.login_time
        total_minutes = int(duration.total_seconds() / 60)
        
    if total_minutes != work_hours.total_minutes:
                old_total = work_hours.total_minutes
                work_hours.total_minutes = total_minutes
                changes.append(f"total minutes: {old_total if old_total else 'None'} -> {total_minutes}")
    
    try:
        session.commit()
        session.refresh(work_hours)
        
        # Log activity
        if adjusted_by_id and changes:
            change_details = f"Work hours record {work_hours_id} adjusted: {', '.join(changes)}. Reason: {reason or 'No reason provided'}"
            await log_activity(
                user_id=adjusted_by_id,
                action_type=ActionType.ADMIN_ADJUST_HOURS,
                details=change_details,
                session=session
            )
        
        return work_hours
    except Exception as e:
        session.rollback()
        raise WorkHoursError(f"Error adjusting work hours: {str(e)}")

async def get_department_work_summary(
    department_id: int,
    start_date: datetime,
    end_date: datetime,
    session: Session
) -> Dict[str, Any]:
    """
    Get a summary of work hours for all users in a department
    
    Args:
        department_id: The ID of the department
        start_date: The start date of the period to summarize
        end_date: The end date of the period to summarize
        session: Database session
        
    Returns:
        A dictionary with department work hours summary
        
    Raises:
        WorkHoursError: If department is not found or other errors occur
    """
    # Get all users in the department
    users = session.query(User).filter(User.department_id == department_id).all()
    
    if not users:
        raise WorkHoursError(f"No users found in department {department_id}")
    
    # Calculate work hours for each user
    user_summaries = []
    department_total_hours = 0
    
    for user in users:
        # Get completed work sessions for the user in date range
        work_sessions = session.query(WorkHours).filter(
            and_(
                WorkHours.user_id == user.id,
                WorkHours.login_time >= start_date,
                WorkHours.login_time <= end_date,
                WorkHours.total_minutes != None
            )
        ).all()
        
        # Calculate total hours
        user_total_minutes = sum(session.total_minutes for session in work_sessions)
        user_total_hours = round(user_total_minutes / 60, 2)
        department_total_hours += user_total_hours
        
        # Create user summary
        user_summary = {
            "user_id": user.id,
            "user_name": user.name,
            "total_hours": user_total_hours,
            "session_count": len(work_sessions)
        }
        user_summaries.append(user_summary)
    
    # Sort users by total hours (descending)
    user_summaries.sort(key=lambda x: x["total_hours"], reverse=True)
    
    # Build department summary
    department_summary = {
        "department_id": department_id,
        "date_range": {
            "start_date": start_date.date().isoformat(),
            "end_date": end_date.date().isoformat()
        },
        "total_department_hours": round(department_total_hours, 2),
        "user_count": len(users),
        "active_user_count": len([u for u in user_summaries if u["total_hours"] > 0]),
        "user_summaries": user_summaries
    }
    
    return department_summary

async def force_logout_inactive_sessions(inactivity_hours: int, admin_id: int, session: Session) -> int:
    """
    Force logout all sessions that have been inactive for longer than specified hours
    
    Args:
        inactivity_hours: Number of hours of inactivity to trigger force logout
        admin_id: ID of admin performing this action
        session: Database session
        
    Returns:
        Number of sessions that were force logged out
        
    Raises:
        WorkHoursError: If admin is not found or other errors occur
    """
    # Validate admin
    admin = session.query(User).filter(User.id == admin_id).first()
    if not admin:
        raise WorkHoursError("Admin user not found")
    
    # Calculate cutoff time
    cutoff_time = datetime.utcnow() - timedelta(hours=inactivity_hours)
    
    # Find active sessions older than cutoff time
    inactive_sessions = session.query(WorkHours).filter(
        WorkHours.login_time < cutoff_time,
        WorkHours.logout_time == None
    ).all()
    
    count = 0
    for work_hours in inactive_sessions:
        # Set logout time to cutoff time
        work_hours.logout_time = cutoff_time
        
        # Calculate duration
        duration = work_hours.logout_time - work_hours.login_time
        work_hours.total_minutes = int(duration.total_seconds() / 60)
        
        count += 1
    
    if count > 0:
        try:
            session.commit()
            
            # Log the activity
            await log_activity(
                user_id=admin_id,
                action_type=ActionType.ADMIN_FORCE_LOGOUT,
                details=f"Forced logout of {count} inactive sessions (inactive > {inactivity_hours} hours)",
                session=session
            )
        except Exception as e:
            session.rollback()
            raise WorkHoursError(f"Error forcing logout: {str(e)}")
    
    return count

async def export_user_work_history(
    user_id: int,
    start_date: datetime,
    end_date: datetime,
    session: Session
) -> List[Dict[str, Any]]:
    """
    Export detailed work history for a user within a date range
    
    Args:
        user_id: The ID of the user
        start_date: Start date for the export
        end_date: End date for the export
        session: Database session
        
    Returns:
        List of work history records for export
        
    Raises:
        WorkHoursError: If user is not found or other errors occur
    """
    # Validate user
    user = session.query(User).filter(User.id == user_id).first()
    if not user:
        raise WorkHoursError("User not found")
    
    # Get work sessions in date range
    work_sessions = session.query(WorkHours).filter(
        WorkHours.user_id == user_id,
        WorkHours.login_time >= start_date,
        WorkHours.login_time <= end_date
    ).order_by(WorkHours.login_time).all()
    
    # Format for export
    export_data = []
    for work_session in work_sessions:
        record = {
            "session_id": work_session.id,
            "user_id": user_id,
            "user_name": user.name,
            "login_date": work_session.login_time.date().isoformat(),
            "login_time": work_session.login_time.strftime("%H:%M:%S"),
            "logout_time": work_session.logout_time.strftime("%H:%M:%S") if work_session.logout_time else "Active",
            "duration_minutes": work_session.total_minutes if work_session.total_minutes else "N/A",
            "duration_hours": round(work_session.total_minutes / 60, 2) if work_session.total_minutes else "N/A"
        }
        export_data.append(record)
    
    return export_data    