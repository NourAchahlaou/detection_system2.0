"""
Profile Management Module

This module handles all aspects of profile completion and management
for the user management application.
"""

# profile.py - Service functions for profile management
from datetime import datetime, time
from typing import List, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_

from user_management.app.db.models.user import User, Shift
from user_management.app.db.models.roleType import RoleType
from user_management.app.db.models.shiftDay import ShiftDay
from user_management.app.db.schemas.profile import (
    ProfileUpdateRequest, 
  
    ShiftRequest
)
from user_management.app.responses.profile import ProfileResponse, ProfileCompletionResponse ,ShiftResponse

class ProfileCompletionError(Exception):
    """Exception raised for errors in the profile completion process."""
    pass

async def update_user_profile(user_id: int, data: ProfileUpdateRequest, session: Session) -> User:
    """
    Update a user's profile with the provided data
    
    Args:
        user_id: The ID of the user to update
        data: The profile data to update
        session: Database session
        
    Returns:
        The updated user object
        
    Raises:
        ProfileCompletionError: If user is not found or other errors occur
    """
    user = session.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise ProfileCompletionError("User not found")
    
    # Update user fields if provided
    if data.airbus_id is not None:
        # Check if airbus_id is already in use by another user
        existing_user = session.query(User).filter(
            and_(User.airbus_id == data.airbus_id, User.id != user_id)
        ).first()
        
        if existing_user:
            raise ProfileCompletionError("Airbus ID is already in use")
            
        user.airbus_id = data.airbus_id
    
    if data.role is not None:
        try:
            user.role = RoleType[data.role]
        except KeyError:
            raise ProfileCompletionError(f"Invalid role: {data.role}")
    
    # Update shifts if provided
    if data.main_shifts is not None:
        try:
            # Remove existing shifts
            session.query(Shift).filter(Shift.user_id == user_id).delete()
            
            # Add new shifts
            for shift_data in data.main_shifts:
                # Convert string time to time object
                try:
                    start_time_parts = shift_data.start_time.split(":")
                    end_time_parts = shift_data.end_time.split(":")
                    
                    start_time = time(int(start_time_parts[0]), int(start_time_parts[1]))
                    end_time = time(int(end_time_parts[0]), int(end_time_parts[1]))
                    
                    # No need to validate the day_of_week as it's now an Enum
                    # This will automatically enforce valid values
                    
                    new_shift = Shift(
                        user_id=user_id,
                        start_time=start_time,
                        end_time=end_time,
                        day_of_week=shift_data.day_of_week
                    )
                    session.add(new_shift)
                except (ValueError, IndexError) as e:
                    raise ProfileCompletionError(f"Invalid time format: {str(e)}")
        except Exception as e:
            session.rollback()
            raise ProfileCompletionError(f"Error updating shifts: {str(e)}")
    
    user.updated_at = datetime.now()
    
    try:
        session.commit()
    except Exception as e:
        session.rollback()
        raise ProfileCompletionError(f"Error updating profile: {str(e)}")
    
    return user

async def get_profile_completion(user_id: int, session: Session) -> ProfileCompletionResponse:
    """
    Calculate profile completion percentage and status for a user
    
    Args:
        user_id: The ID of the user
        session: Database session
        
    Returns:
        A ProfileCompletionResponse object with completion percentage and status
        
    Raises:
        ProfileCompletionError: If user is not found
    """
    user = session.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise ProfileCompletionError("User not found")
    
    # Get user shifts
    shifts = session.query(Shift).filter(Shift.user_id == user_id).all()
    
    # Group shifts by day to determine if we have full coverage
    shifts_by_day = {}
    for shift in shifts:
        if shift.day_of_week not in shifts_by_day:
            shifts_by_day[shift.day_of_week] = []
        shifts_by_day[shift.day_of_week].append(shift)
    
    # Define fields required for completion and their status
    completion_status = {
        "personal_info": {
            "airbus_id": bool(user.airbus_id),
            "role": bool(user.role)
        },
        "main_shift": {
            "shifts": len(shifts) > 0,
            "has_weekday_coverage": any(
                day in shifts_by_day for day in [
                    ShiftDay.MONDAY, ShiftDay.TUESDAY, ShiftDay.WEDNESDAY, 
                    ShiftDay.THURSDAY, ShiftDay.FRIDAY
                ]
            )  # At least one weekday
        }
    }
    
    # Calculate completion percentage
    # We count: airbus_id, role, shifts existence, weekday coverage
    total_fields = 4
    completed_fields = sum([
        completion_status["personal_info"]["airbus_id"],
        completion_status["personal_info"]["role"],
        completion_status["main_shift"]["shifts"],
        completion_status["main_shift"]["has_weekday_coverage"]
    ])
    
    completion_percentage = round((completed_fields / total_fields) * 100)
    
    # Generate list of missing fields for user guidance
    missing_fields = []
    if not completion_status["personal_info"]["airbus_id"]:
        missing_fields.append("Airbus ID")
    if not completion_status["personal_info"]["role"]:
        missing_fields.append("Role")
    if not completion_status["main_shift"]["shifts"]:
        missing_fields.append("Work Schedule")
    elif not completion_status["main_shift"]["has_weekday_coverage"]:
        missing_fields.append("Weekday Work Schedule")
    
    return ProfileCompletionResponse(
        completion_percentage=completion_percentage,
        completion_status=completion_status,
        missing_fields=missing_fields
    )

async def get_user_shifts(user_id: int, session: Session) -> List[Dict[str, Any]]:
    """
    Get all shifts for a user, formatted for frontend display
    
    Args:
        user_id: The ID of the user
        session: Database session
        
    Returns:
        A list of shifts with formatted time and day information
        
    Raises:
        ProfileCompletionError: If user is not found
    """
    user = session.query(User).filter(User.id == user_id).first()
    
    if not user:
        raise ProfileCompletionError("User not found")
    
    shifts = session.query(Shift).filter(Shift.user_id == user_id).all()
    
    # Get day names directly from the enum
    day_names = {
        ShiftDay.MONDAY: "Monday",
        ShiftDay.TUESDAY: "Tuesday",
        ShiftDay.WEDNESDAY: "Wednesday",
        ShiftDay.THURSDAY: "Thursday",
        ShiftDay.FRIDAY: "Friday",
        ShiftDay.SATURDAY: "Saturday",
        ShiftDay.SUNDAY: "Sunday"
    }
    
    formatted_shifts = []
    for shift in shifts:
        formatted_shifts.append({
            "id": shift.id,
            "day": shift.day_of_week,
            "day_name": day_names[shift.day_of_week],
            "start_time": shift.start_time.strftime("%H:%M"),
            "end_time": shift.end_time.strftime("%H:%M")
        })
    
    return formatted_shifts