# user_management/app/services/profile_tab.py
"""
Profile Tab Service Module

This module handles profile tab operations including fetching user information,
editing user details, and managing user profile data specifically for the profile tab.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_

from user_management.app.db.models.user import User
from user_management.app.db.models.shift import Shift
from user_management.app.db.models.roleType import RoleType
from user_management.app.db.models.shiftDay import ShiftDay
from user_management.app.db.schemas.profileSchema import (
    ProfileTabUpdateRequest,
    ProfileTabResponse,
    ShiftUpdateRequest  # New schema for shift updates
)


class ProfileTabError(Exception):
    """Exception raised for errors in profile tab operations."""
    pass


async def get_user_profile_tab_info(user_id: int, session: Session) -> ProfileTabResponse:
    """
    Get complete user profile information for the profile tab
    
    Args:
        user_id: The ID of the user
        session: Database session
        
    Returns:
        ProfileTabResponse with all user information
        
    Raises:
        ProfileTabError: If user is not found
    """
    user = session.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise ProfileTabError("User not found")
    
    # Get user shifts
    shifts = session.query(Shift).filter(Shift.user_id == user_id).all()
    
    # Calculate total weekly hours
    total_weekly_minutes = 0
    for shift in shifts:
        start_minutes = shift.start_time.hour * 60 + shift.start_time.minute
        end_minutes = shift.end_time.hour * 60 + shift.end_time.minute
        
        # Handle overnight shifts
        if end_minutes < start_minutes:
            end_minutes += 24 * 60
            
        shift_minutes = end_minutes - start_minutes
        total_weekly_minutes += shift_minutes
    
    # Convert to hours and minutes
    total_hours = total_weekly_minutes // 60
    remaining_minutes = total_weekly_minutes % 60
    total_hours_display = f"{total_hours}h {remaining_minutes}min"
    
    # Determine access level based on role
    access_level = "Level 1" if user.role == RoleType.DATA_MANAGER else  "Level 1" if user.role == RoleType.OPERATOR else "Level 2"
    access_description = "Standard data manager" if user.role == RoleType.DATA_MANAGER else  "Standard operator" if user.role == RoleType.OPERATOR else "Auditor"
    
    # Get primary shift times (most common pattern or first shift)
    shift_start = "N/A"
    shift_end = "N/A"
    if shifts:
        # Find the most common shift pattern or use first shift
        first_shift = shifts[0]
        shift_start = first_shift.start_time.strftime("%H:%M")
        shift_end = first_shift.end_time.strftime("%H:%M")
    
    return ProfileTabResponse(
        # Personal Information
        first_name=user.name.split()[0] if user.name else "",
        last_name=" ".join(user.name.split()[1:]) if user.name and len(user.name.split()) > 1 else "",
        email=user.email,
        role=user.role.name.title() if user.role else "N/A",
        shift_start=shift_start,
        shift_end=shift_end,
        current_status="Online" if user.is_active else "Offline",
        total_hours_this_week=total_hours_display,
        
        # Access & Credentials
        employee_id=f"T-{user.airbus_id}" if user.airbus_id else "N/A",
        work_area="Quality area",  # This could be made dynamic later
        badge_number=f"AIRBUS-{user.airbus_id}" if user.airbus_id else "N/A",
        access_level=f"{access_level} ({access_description})",
        
        # Additional info for calculations
        user_id=user.id,
        airbus_id=user.airbus_id,
        role_enum=user.role,
        is_active=user.is_active
    )


async def update_user_profile_tab(user_id: int, data: ProfileTabUpdateRequest, session: Session) -> User:
    """
    Update user profile information from the profile tab
    
    Args:
        user_id: The ID of the user to update
        data: The profile data to update
        session: Database session
        
    Returns:
        The updated user object
        
    Raises:
        ProfileTabError: If user is not found or validation fails
    """
    user = session.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise ProfileTabError("User not found")
    
    try:
        # Update personal information
        if data.first_name is not None or data.last_name is not None:
            first_name = data.first_name or (user.name.split()[0] if user.name else "")
            last_name = data.last_name or (" ".join(user.name.split()[1:]) if user.name and len(user.name.split()) > 1 else "")
            user.name = f"{first_name} {last_name}".strip()
        
        if data.email is not None:
            # Check if email is already in use by another user
            existing_user = session.query(User).filter(
                and_(User.email == data.email, User.id != user_id)
            ).first()
            
            if existing_user:
                raise ProfileTabError("Email is already in use")
            
            user.email = data.email
        
        if data.role is not None:
            try:
                user.role = RoleType[data.role.upper()]
            except KeyError:
                raise ProfileTabError(f"Invalid role: {data.role}")
        
        if data.airbus_id is not None:
            # Check if airbus_id is already in use by another user
            existing_user = session.query(User).filter(
                and_(User.airbus_id == data.airbus_id, User.id != user_id)
            ).first()
            
            if existing_user:
                raise ProfileTabError("Airbus ID is already in use")
                
            user.airbus_id = data.airbus_id
        
        user.updated_at = datetime.utcnow()
        
        session.commit()
        session.refresh(user)
        
        return user
        
    except Exception as e:
        session.rollback()
        raise ProfileTabError(f"Error updating profile: {str(e)}")


async def update_user_shifts(user_id: int, shifts_data: List[ShiftUpdateRequest], session: Session) -> bool:
    """
    Update user shifts with support for regular and special shifts
    
    Args:
        user_id: The ID of the user
        shifts_data: List of shift updates
        session: Database session
        
    Returns:
        True if successful
        
    Raises:
        ProfileTabError: If user is not found or validation fails
    """
    user = session.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise ProfileTabError("User not found")
    
    try:
        # Process each shift update
        for shift_data in shifts_data:
            # Validate time format
            try:
                from datetime import time
                start_parts = shift_data.start_time.split(":")
                end_parts = shift_data.end_time.split(":")
                
                start_time = time(int(start_parts[0]), int(start_parts[1]))
                end_time = time(int(end_parts[0]), int(end_parts[1]))
                
            except (ValueError, IndexError) as e:
                raise ProfileTabError(f"Invalid time format for {shift_data.day_of_week}: {str(e)}")
            
            # Check if shift exists for this day
            existing_shift = session.query(Shift).filter(
                and_(
                    Shift.user_id == user_id,
                    Shift.day_of_week == shift_data.day_of_week
                )
            ).first()
            
            if shift_data.action == "update" or shift_data.action == "create":
                if existing_shift:
                    # Update existing shift
                    existing_shift.start_time = start_time
                    existing_shift.end_time = end_time
                    existing_shift.updated_at = datetime.utcnow()
                else:
                    # Create new shift
                    new_shift = Shift(
                        user_id=user_id,
                        start_time=start_time,
                        end_time=end_time,
                        day_of_week=shift_data.day_of_week
                    )
                    session.add(new_shift)
                    
            elif shift_data.action == "delete":
                if existing_shift:
                    session.delete(existing_shift)
        
        session.commit()
        return True
        
    except Exception as e:
        session.rollback()
        raise ProfileTabError(f"Error updating shifts: {str(e)}")


async def bulk_update_regular_shifts(user_id: int, start_time: str, end_time: str, days: List[ShiftDay], session: Session) -> bool:
    """
    Update regular shifts for multiple days with the same time
    
    Args:
        user_id: The ID of the user
        start_time: Start time in HH:MM format
        end_time: End time in HH:MM format
        days: List of days to apply the shift to
        session: Database session
        
    Returns:
        True if successful
        
    Raises:
        ProfileTabError: If user is not found or validation fails
    """
    user = session.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise ProfileTabError("User not found")
    
    try:
        # Validate time format
        from datetime import time
        start_parts = start_time.split(":")
        end_parts = end_time.split(":")
        
        start_time_obj = time(int(start_parts[0]), int(start_parts[1]))
        end_time_obj = time(int(end_parts[0]), int(end_parts[1]))
        
        # Update shifts for specified days
        for day in days:
            existing_shift = session.query(Shift).filter(
                and_(
                    Shift.user_id == user_id,
                    Shift.day_of_week == day
                )
            ).first()
            
            if existing_shift:
                # Update existing shift
                existing_shift.start_time = start_time_obj
                existing_shift.end_time = end_time_obj
                existing_shift.updated_at = datetime.utcnow()
            else:
                # Create new shift
                new_shift = Shift(
                    user_id=user_id,
                    start_time=start_time_obj,
                    end_time=end_time_obj,
                    day_of_week=day
                )
                session.add(new_shift)
        
        session.commit()
        return True
        
    except (ValueError, IndexError) as e:
        session.rollback()
        raise ProfileTabError(f"Invalid time format: {str(e)}")
    except Exception as e:
        session.rollback()
        raise ProfileTabError(f"Error updating regular shifts: {str(e)}")


async def get_user_detailed_shifts(user_id: int, session: Session) -> Dict[str, Any]:
    """
    Get detailed user shifts organized by day
    
    Args:
        user_id: The ID of the user
        session: Database session
        
    Returns:
        Dictionary with detailed shift information
        
    Raises:
        ProfileTabError: If user is not found
    """
    user = session.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise ProfileTabError("User not found")
    
    shifts = session.query(Shift).filter(Shift.user_id == user_id).all()
    
    # Organize shifts by day
    shifts_by_day = {}
    day_names = {
        ShiftDay.MONDAY: "Monday",
        ShiftDay.TUESDAY: "Tuesday", 
        ShiftDay.WEDNESDAY: "Wednesday",
        ShiftDay.THURSDAY: "Thursday",
        ShiftDay.FRIDAY: "Friday",
        ShiftDay.SATURDAY: "Saturday",
        ShiftDay.SUNDAY: "Sunday"
    }
    
    total_weekly_minutes = 0
    
    for shift in shifts:
        day_name = day_names[shift.day_of_week]
        start_time = shift.start_time.strftime("%H:%M")
        end_time = shift.end_time.strftime("%H:%M")
        
        shifts_by_day[day_name] = {
            "id": shift.id,
            "day": shift.day_of_week,
            "start": start_time,
            "end": end_time,
            "created_at": shift.created_at.isoformat() if shift.created_at else None,
            "updated_at": shift.updated_at.isoformat() if shift.updated_at else None
        }
        
        # Calculate minutes for this shift
        start_minutes = shift.start_time.hour * 60 + shift.start_time.minute
        end_minutes = shift.end_time.hour * 60 + shift.end_time.minute
        
        if end_minutes < start_minutes:
            end_minutes += 24 * 60
            
        total_weekly_minutes += end_minutes - start_minutes
    
    total_hours = total_weekly_minutes // 60
    remaining_minutes = total_weekly_minutes % 60
    
    return {
        "shifts_by_day": shifts_by_day,
        "total_weekly_hours": total_hours,
        "total_weekly_minutes": remaining_minutes,
        "total_weekly_display": f"{total_hours}h {remaining_minutes}min",
        "has_weekend_shifts": any(
            day in shifts_by_day for day in ["Saturday", "Sunday"]
        ),
        "weekday_count": sum(
            1 for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
            if day in shifts_by_day
        )
    }


async def delete_user_profile(user_id: int, session: Session) -> bool:
    """
    Delete a user profile (soft delete by setting is_active to False)
    
    Args:
        user_id: The ID of the user to delete
        session: Database session
        
    Returns:
        True if successful
        
    Raises:
        ProfileTabError: If user is not found or deletion fails
    """
    user = session.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise ProfileTabError("User not found")
    
    try:
        # Soft delete - set user as inactive
        user.is_active = False
        user.updated_at = datetime.utcnow()
        
        session.commit()
        
        return True
        
    except Exception as e:
        session.rollback()
        raise ProfileTabError(f"Error deleting user profile: {str(e)}")


# Keep the existing get_user_work_summary for backward compatibility
async def get_user_work_summary(user_id: int, session: Session) -> dict:
    """
    Get user work summary including shifts and total hours
    
    Args:
        user_id: The ID of the user
        session: Database session
        
    Returns:
        Dictionary with work summary information
        
    Raises:
        ProfileTabError: If user is not found
    """
    return await get_user_detailed_shifts(user_id, session)