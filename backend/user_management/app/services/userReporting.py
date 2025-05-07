"""
User Reporting Module

This module provides reporting and analytics capabilities for the user management application,
including user statistics, workload analysis, and system insights.
"""

# reporting.py - Service functions for user reporting and analytics
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session
from sqlalchemy import func, desc, and_, or_

from user_management.app.db.models.user import User, Shift
from user_management.app.db.models.task import Task
from user_management.app.db.models.activities import Activity
from user_management.app.db.models.roleType import RoleType
from user_management.app.db.models.shiftDay import ShiftDay
from user_management.app.db.models.workHours import WorkHours
from user_management.app.db.models.actionType import ActionType

class ReportingError(Exception):
    """Exception raised for errors in the reporting process."""
    pass

async def get_user_statistics(session: Session) -> Dict[str, Any]:
    """
    Get general user statistics for the system
    
    Args:
        session: Database session
        
    Returns:
        A dictionary with user statistics data
    """
    # Get all users
    users = session.query(User).all()
    
    # Count users by role
    role_counts = {}
    for role in RoleType:
        count = sum(1 for user in users if user.role == role)
        role_counts[role.name] = count
    
    # Count users by active status
    active_count = sum(1 for user in users if user.is_active)
    inactive_count = len(users) - active_count
    
    # Count users by verification status
    verified_count = sum(1 for user in users if user.verified_at is not None)
    unverified_count = len(users) - verified_count
    
    # Get recently joined users (last 30 days)
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    recent_users = [user for user in users if user.created_at and user.created_at >= thirty_days_ago]
    
    # Calculate average profile completion
    total_completion = 0
    for user in users:
        # This is a simplified calculation - in a real system, we would call
        # the get_profile_completion function for each user
        completion = 0
        if user.airbus_id:
            completion += 25
        if user.role:
            completion += 25
        
        # Check if user has shifts
        shifts = session.query(Shift).filter(Shift.user_id == user.id).all()
        if shifts:
            completion += 25
            
            # Check if user has weekday coverage
            weekday_shifts = [
                shift for shift in shifts 
                if shift.day_of_week in [
                    ShiftDay.MONDAY, ShiftDay.TUESDAY, ShiftDay.WEDNESDAY, 
                    ShiftDay.THURSDAY, ShiftDay.FRIDAY
                ]
            ]
            if weekday_shifts:
                completion += 25
                
        total_completion += completion
    
    avg_completion = total_completion / len(users) if users else 0
    
    # Build statistics object
    statistics = {
        "total_users": len(users),
        "active_users": active_count,
        "inactive_users": inactive_count,
        "verified_users": verified_count,
        "unverified_users": unverified_count,
        "by_role": role_counts,
        "new_users_last_30_days": len(recent_users),
        "average_profile_completion": avg_completion
    }
    
    return statistics

async def get_workload_analysis(session: Session) -> Dict[str, Any]:
    """
    Get workload analysis data across users
    
    Args:
        session: Database session
        
    Returns:
        A dictionary with workload analysis data
    """
    # Get all active users
    active_users = session.query(User).filter(User.is_active == True).all()
    
    # Get all tasks
    tasks = session.query(Task).all()
    
    # Get all work hours for the last 30 days
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    work_hours = session.query(WorkHours).filter(WorkHours.login_time >= thirty_days_ago).all()
    
    # Calculate tasks per user
    task_counts = {}
    for user in active_users:
        user_tasks = [task for task in tasks if task.assigned_user_id == user.id]
        pending_tasks = [task for task in user_tasks if task.status == "pending"]
        in_progress_tasks = [task for task in user_tasks if task.status == "in_progress"]
        completed_tasks = [task for task in user_tasks if task.status == "completed"]
        
        task_counts[user.id] = {
            "user_id": user.id,
            "user_name": user.name,
            "total_tasks": len(user_tasks),
            "pending_tasks": len(pending_tasks),
            "in_progress_tasks": len(in_progress_tasks),
            "completed_tasks": len(completed_tasks),
            "completion_rate": (len(completed_tasks) / len(user_tasks)) * 100 if user_tasks else 0
        }
    
    # Calculate work hours per user
    hours_per_user = {}
    for user in active_users:
        user_hours = [wh for wh in work_hours if wh.user_id == user.id]
        total_minutes = sum(wh.total_minutes or 0 for wh in user_hours)
        
        # Calculate daily average over the last 30 days
        hours_per_user[user.id] = {
            "user_id": user.id,
            "user_name": user.name,
            "total_hours": round(total_minutes / 60, 2),
            "daily_average_hours": round((total_minutes / 60) / 30, 2),
            "login_count": len(user_hours)
        }
    
    # Find users with no tasks
    users_without_tasks = [
        {
            "user_id": user.id,
            "user_name": user.name,
            "role": user.role.name if user.role else None
        }
        for user in active_users 
        if user.id not in task_counts or task_counts[user.id]["total_tasks"] == 0
    ]
    
    # Find users with high workload (more than 10 active tasks)
    high_workload_users = [
        {
            "user_id": user.id,
            "user_name": user.name,
            "active_tasks": task_counts[user.id]["pending_tasks"] + task_counts[user.id]["in_progress_tasks"]
        }
        for user in active_users 
        if user.id in task_counts and 
           (task_counts[user.id]["pending_tasks"] + task_counts[user.id]["in_progress_tasks"]) > 10
    ]
    
    # Build workload analysis
    workload_analysis = {
        "task_distribution": list(task_counts.values()),
        "work_hours": list(hours_per_user.values()),
        "users_without_tasks": users_without_tasks,
        "high_workload_users": high_workload_users
    }
    
    return workload_analysis

async def get_user_work_hours(
    user_id: int,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    session: Session = None
) -> Dict[str, Any]:
    """
    Get work hours summary for a specific user
    
    Args:
        user_id: The ID of the user
        start_date: Start date for the analysis (optional, defaults to last 30 days)
        end_date: End date for the analysis (optional, defaults to now)
        session: Database session
        
    Returns:
        A dictionary with work hours data
        
    Raises:
        ReportingError: If user is not found
    """
    # Validate user
    user = session.query(User).filter(User.id == user_id).first()
    if not user:
        raise ReportingError("User not found")
    
    # Set default date range if not provided
    if not end_date:
        end_date = datetime.utcnow()
    
    if not start_date:
        start_date = end_date - timedelta(days=30)
    
    # Get work hours in date range
    work_hours = session.query(WorkHours).filter(
        WorkHours.user_id == user_id,
        WorkHours.login_time >= start_date,
        WorkHours.login_time <= end_date
    ).all()
    
    # Calculate total work hours
    total_minutes = sum(wh.total_minutes or 0 for wh in work_hours)
    total_hours = total_minutes / 60
    
    # Calculate average daily hours
    days_in_range = (end_date - start_date).days
    avg_daily_hours = total_hours / days_in_range if days_in_range > 0 else 0
    
    # Calculate hours by day of week
    hours_by_day = {day.value: 0 for day in ShiftDay}
    for wh in work_hours:
        day_of_week = wh.login_time.weekday()
        hours_by_day[day_of_week] += (wh.total_minutes or 0) / 60
    
    # Format hours by day of week
    formatted_hours_by_day = {
        ShiftDay(day).name: round(hours, 2)
        for day, hours in hours_by_day.items()
    }
    
    # Get daily work hours details for charting
    daily_hours = {}
    for wh in work_hours:
        date_str = wh.login_time.date().isoformat()
        if date_str not in daily_hours:
            daily_hours[date_str] = 0
        daily_hours[date_str] += (wh.total_minutes or 0) / 60
    
    daily_work_hours = [
        {"date": date_str, "hours": round(hours, 2)}
        for date_str, hours in daily_hours.items()
    ]
    
    # Sort by date
    daily_work_hours.sort(key=lambda x: x["date"])
    
    # Build work hours summary
    work_hours_summary = {
        "user_id": user_id,
        "user_name": user.name,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "total_hours": round(total_hours, 2),
        "average_daily_hours": round(avg_daily_hours, 2),
        "login_count": len(work_hours),
        "by_day_of_week": formatted_hours_by_day,
        "daily_work_hours": daily_work_hours
    }
    
    return work_hours_summary

async def get_shift_coverage_analysis(session: Session) -> Dict[str, Any]:
    """
    Analyze shift coverage across all users
    
    Args:
        session: Database session
        
    Returns:
        A dictionary with shift coverage analysis data
    """
    # Get all active users
    active_users = session.query(User).filter(User.is_active == True).all()
    
    # Get all shifts
    all_shifts = session.query(Shift).all()
    
    # Count shifts by day of week
    shifts_by_day = {day.value: 0 for day in ShiftDay}
    for shift in all_shifts:
        shifts_by_day[shift.day_of_week.value] += 1
    
    # Format shifts by day
    formatted_shifts_by_day = {
        ShiftDay(day).name: count
        for day, count in shifts_by_day.items()
    }
    
    # Identify days with low coverage (less than 20% of users)
    min_coverage_threshold = len(active_users) * 0.2
    low_coverage_days = [
        {
            "day": ShiftDay(day).name,
            "coverage": count,
            "coverage_percentage": (count / len(active_users)) * 100 if len(active_users) > 0 else 0
        }
        for day, count in shifts_by_day.items()
        if count < min_coverage_threshold
    ]
    
    # Calculate coverage by hour of day
    hours_coverage = {hour: {day.value: 0 for day in ShiftDay} for hour in range(24)}
    
    for shift in all_shifts:
        start_hour = shift.start_time.hour
        end_hour = shift.end_time.hour
        
        # Handle shifts that span midnight
        if end_hour < start_hour:
            end_hour += 24
        
        for hour in range(start_hour, end_hour):
            hour_mod = hour % 24
            hours_coverage[hour_mod][shift.day_of_week.value] += 1
    
    # Identify hours with low coverage
    low_coverage_hours = []
    for hour, day_counts in hours_coverage.items():
        for day, count in day_counts.items():
            if day < 5:  # Only look at weekdays (Monday - Friday)
                if count < min_coverage_threshold:
                    low_coverage_hours.append({
                        "day": ShiftDay(day).name,
                        "hour": hour,
                        "time": f"{hour:02d}:00",
                        "coverage": count,
                        "coverage_percentage": (count / len(active_users)) * 100 if len(active_users) > 0 else 0
                    })
    
    # Build shift coverage analysis
    coverage_analysis = {
        "total_users": len(active_users),
        "users_with_shifts": len(set(shift.user_id for shift in all_shifts)),
        "coverage_by_day": formatted_shifts_by_day,
        "low_coverage_days": low_coverage_days,
        "low_coverage_hours": low_coverage_hours
    }
    
    return coverage_analysis

async def get_system_health_report(session: Session) -> Dict[str, Any]:
    """
    Generate a system health report with key metrics
    
    Args:
        session: Database session
        
    Returns:
        A dictionary with system health metrics
    """
    # Time periods
    now = datetime.utcnow()
    last_day = now - timedelta(days=1)
    last_week = now - timedelta(days=7)
    last_month = now - timedelta(days=30)
    
    # User metrics
    total_users = session.query(func.count(User.id)).scalar()
    active_users = session.query(func.count(User.id)).filter(User.is_active == True).scalar()
    
    new_users_last_day = session.query(func.count(User.id)).filter(User.created_at >= last_day).scalar()
    new_users_last_week = session.query(func.count(User.id)).filter(User.created_at >= last_week).scalar()
    new_users_last_month = session.query(func.count(User.id)).filter(User.created_at >= last_month).scalar()
    
    # Task metrics
    total_tasks = session.query(func.count(Task.id)).scalar()
    pending_tasks = session.query(func.count(Task.id)).filter(Task.status == "pending").scalar()
    in_progress_tasks = session.query(func.count(Task.id)).filter(Task.status == "in_progress").scalar()
    completed_tasks = session.query(func.count(Task.id)).filter(Task.status == "completed").scalar()
    
    tasks_last_day = session.query(func.count(Task.id)).filter(Task.created_at >= last_day).scalar()
    tasks_last_week = session.query(func.count(Task.id)).filter(Task.created_at >= last_week).scalar()
    tasks_last_month = session.query(func.count(Task.id)).filter(Task.created_at >= last_month).scalar()
    
    # Activity metrics
    activities_last_day = session.query(func.count(Activity.id)).filter(Activity.timestamp >= last_day).scalar()
    activities_last_week = session.query(func.count(Activity.id)).filter(Activity.timestamp >= last_week).scalar()
    activities_last_month = session.query(func.count(Activity.id)).filter(Activity.timestamp >= last_month).scalar()
    
    # Work hours metrics
    total_work_hours = session.query(func.sum(WorkHours.total_minutes)).filter(WorkHours.total_minutes != None).scalar() or 0
    total_work_hours = total_work_hours / 60  # Convert to hours
    
    work_hours_last_day = session.query(func.sum(WorkHours.total_minutes)).filter(
        WorkHours.login_time >= last_day,
        WorkHours.total_minutes != None
    ).scalar() or 0
    work_hours_last_day = work_hours_last_day / 60
    
    work_hours_last_week = session.query(func.sum(WorkHours.total_minutes)).filter(
        WorkHours.login_time >= last_week,
        WorkHours.total_minutes != None
    ).scalar() or 0
    work_hours_last_week = work_hours_last_week / 60
    
    # Build system health report
    health_report = {
        "users": {
            "total": total_users,
            "active": active_users,
            "inactive": total_users - active_users,
            "new_last_day": new_users_last_day,
            "new_last_week": new_users_last_week,
            "new_last_month": new_users_last_month
        },
        "tasks": {
            "total": total_tasks,
            "pending": pending_tasks,
            "in_progress": in_progress_tasks,
            "completed": completed_tasks,
            "completion_rate": (completed_tasks / total_tasks) * 100 if total_tasks > 0 else 0,
            "created_last_day": tasks_last_day,
            "created_last_week": tasks_last_week,
            "created_last_month": tasks_last_month
        },
        "activities": {
            "last_day": activities_last_day,
            "last_week": activities_last_week,
            "last_month": activities_last_month,
            "daily_average": activities_last_month / 30
        },
        "work_hours": {
            "total": round(total_work_hours, 2),
            "last_day": round(work_hours_last_day, 2),
            "last_week": round(work_hours_last_week, 2),
            "average_per_active_user": round(total_work_hours / active_users, 2) if active_users > 0 else 0
        },
        "generated_at": now.isoformat()
    }
    
    return health_report