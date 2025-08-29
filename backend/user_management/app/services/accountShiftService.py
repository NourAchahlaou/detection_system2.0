"""
Shift Service Module

This module handles all shift-related operations including getting, creating,
updating, and deleting user shifts.
"""

from datetime import datetime, time
from typing import List, Dict, Any, Optional
from sqlalchemy.orm import Session

from user_management.app.db.models.user import User
from user_management.app.db.models.shift import Shift
from user_management.app.db.models.shiftDay import ShiftDay


class ShiftServiceError(Exception):
    """Exception raised for errors in shift service operations."""
    pass


class ShiftService:
    """Service class for handling shift operations."""
    
    @staticmethod
    async def get_all_user_shifts(user_id: int, session: Session) -> List[Dict[str, Any]]:
        """
        Get all shifts for a specific user.
        
        Args:
            user_id: The ID of the user
            session: Database session
            
        Returns:
            List of shifts with formatted information
            
        Raises:
            ShiftServiceError: If user is not found
        """
        # Verify user exists
        user = session.query(User).filter(User.id == user_id).first()
        if not user:
            raise ShiftServiceError("User not found")
        
        shifts = session.query(Shift).filter(Shift.user_id == user_id).all()
        
        # Map day enum values to readable names
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
                "user_id": shift.user_id,
                "day_of_week": shift.day_of_week.value,
                "day_name": day_names[shift.day_of_week],
                "start_time": shift.start_time.strftime("%H:%M"),
                "end_time": shift.end_time.strftime("%H:%M"),
                "created_at": shift.created_at.isoformat() if shift.created_at else None,
                "updated_at": shift.updated_at.isoformat() if shift.updated_at else None
            })
        
        return formatted_shifts
    
    @staticmethod
    async def get_shift_by_id(shift_id: int, session: Session) -> Dict[str, Any]:
        """
        Get a specific shift by ID.
        
        Args:
            shift_id: The ID of the shift
            session: Database session
            
        Returns:
            Shift information
            
        Raises:
            ShiftServiceError: If shift is not found
        """
        shift = session.query(Shift).filter(Shift.id == shift_id).first()
        
        if not shift:
            raise ShiftServiceError("Shift not found")
        
        day_names = {
            ShiftDay.MONDAY: "Monday",
            ShiftDay.TUESDAY: "Tuesday", 
            ShiftDay.WEDNESDAY: "Wednesday",
            ShiftDay.THURSDAY: "Thursday",
            ShiftDay.FRIDAY: "Friday",
            ShiftDay.SATURDAY: "Saturday",
            ShiftDay.SUNDAY: "Sunday"
        }
        
        return {
            "id": shift.id,
            "user_id": shift.user_id,
            "day_of_week": shift.day_of_week.value,
            "day_name": day_names[shift.day_of_week],
            "start_time": shift.start_time.strftime("%H:%M"),
            "end_time": shift.end_time.strftime("%H:%M"),
            "created_at": shift.created_at.isoformat() if shift.created_at else None,
            "updated_at": shift.updated_at.isoformat() if shift.updated_at else None
        }
    
    @staticmethod
    async def create_shift(
        user_id: int,
        day_of_week: str,
        start_time: str,
        end_time: str,
        session: Session
    ) -> Shift:
        """
        Create a new shift for a user.
        
        Args:
            user_id: The ID of the user
            day_of_week: Day of the week (e.g., 'MONDAY')
            start_time: Start time in HH:MM format
            end_time: End time in HH:MM format
            session: Database session
            
        Returns:
            The created shift object
            
        Raises:
            ShiftServiceError: If validation fails or user not found
        """
        # Verify user exists
        user = session.query(User).filter(User.id == user_id).first()
        if not user:
            raise ShiftServiceError("User not found")
        
        try:
            # Validate and convert day_of_week
            try:
                shift_day = ShiftDay[day_of_week.upper()]
            except KeyError:
                valid_days = [day.value for day in ShiftDay]
                raise ShiftServiceError(f"Invalid day. Valid days are: {', '.join(valid_days)}")
            
            # Parse and validate time formats
            start_time_obj = ShiftService._parse_time(start_time)
            end_time_obj = ShiftService._parse_time(end_time)
            
            # Validate time logic
            if start_time_obj >= end_time_obj:
                raise ShiftServiceError("Start time must be before end time")
            
            # Check for overlapping shifts on the same day
            existing_shifts = session.query(Shift).filter(
                Shift.user_id == user_id,
                Shift.day_of_week == shift_day
            ).all()
            
            for existing_shift in existing_shifts:
                if ShiftService._shifts_overlap(start_time_obj, end_time_obj, 
                                              existing_shift.start_time, existing_shift.end_time):
                    raise ShiftServiceError(f"Shift overlaps with existing shift on {shift_day.value}")
            
            # Create new shift
            new_shift = Shift(
                user_id=user_id,
                day_of_week=shift_day,
                start_time=start_time_obj,
                end_time=end_time_obj,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow()
            )
            
            session.add(new_shift)
            session.commit()
            
            return new_shift
            
        except ShiftServiceError:
            session.rollback()
            raise
        except Exception as e:
            session.rollback()
            raise ShiftServiceError(f"Error creating shift: {str(e)}")
    
    @staticmethod
    async def update_shift(
        shift_id: int,
        session: Session,
        day_of_week: Optional[str] = None,
        start_time: Optional[str] = None,
        end_time: Optional[str] = None
    ) -> Shift:
        """
        Update an existing shift.
        
        Args:
            shift_id: The ID of the shift to update
            session: Database session
            day_of_week: New day of the week (optional)
            start_time: New start time in HH:MM format (optional)
            end_time: New end time in HH:MM format (optional)
            
        Returns:
            The updated shift object
            
        Raises:
            ShiftServiceError: If shift not found or validation fails
        """
        shift = session.query(Shift).filter(Shift.id == shift_id).first()
        
        if not shift:
            raise ShiftServiceError("Shift not found")
        
        try:
            # Store original values for validation
            new_day = shift.day_of_week
            new_start = shift.start_time
            new_end = shift.end_time
            
            # Update day_of_week if provided
            if day_of_week is not None:
                try:
                    new_day = ShiftDay[day_of_week.upper()]
                except KeyError:
                    valid_days = [day.value for day in ShiftDay]
                    raise ShiftServiceError(f"Invalid day. Valid days are: {', '.join(valid_days)}")
            
            # Update start_time if provided
            if start_time is not None:
                new_start = ShiftService._parse_time(start_time)
            
            # Update end_time if provided
            if end_time is not None:
                new_end = ShiftService._parse_time(end_time)
            
            # Validate time logic
            if new_start >= new_end:
                raise ShiftServiceError("Start time must be before end time")
            
            # Check for overlapping shifts (excluding current shift)
            existing_shifts = session.query(Shift).filter(
                Shift.user_id == shift.user_id,
                Shift.day_of_week == new_day,
                Shift.id != shift_id
            ).all()
            
            for existing_shift in existing_shifts:
                if ShiftService._shifts_overlap(new_start, new_end,
                                              existing_shift.start_time, existing_shift.end_time):
                    raise ShiftServiceError(f"Updated shift would overlap with existing shift on {new_day.value}")
            
            # Apply updates
            shift.day_of_week = new_day
            shift.start_time = new_start
            shift.end_time = new_end
            shift.updated_at = datetime.utcnow()
            
            session.commit()
            
            return shift
            
        except ShiftServiceError:
            session.rollback()
            raise
        except Exception as e:
            session.rollback()
            raise ShiftServiceError(f"Error updating shift: {str(e)}")
    
    @staticmethod
    async def delete_shift(shift_id: int, session: Session) -> bool:
        """
        Delete a specific shift.
        
        Args:
            shift_id: The ID of the shift to delete
            session: Database session
            
        Returns:
            True if deletion was successful
            
        Raises:
            ShiftServiceError: If shift not found or deletion fails
        """
        shift = session.query(Shift).filter(Shift.id == shift_id).first()
        
        if not shift:
            raise ShiftServiceError("Shift not found")
        
        try:
            session.delete(shift)
            session.commit()
            return True
            
        except Exception as e:
            session.rollback()
            raise ShiftServiceError(f"Error deleting shift: {str(e)}")
    
    @staticmethod
    async def delete_all_user_shifts(user_id: int, session: Session) -> int:
        """
        Delete all shifts for a specific user.
        
        Args:
            user_id: The ID of the user
            session: Database session
            
        Returns:
            Number of shifts deleted
            
        Raises:
            ShiftServiceError: If user not found or deletion fails
        """
        # Verify user exists
        user = session.query(User).filter(User.id == user_id).first()
        if not user:
            raise ShiftServiceError("User not found")
        
        try:
            # Count shifts before deletion
            shifts_count = session.query(Shift).filter(Shift.user_id == user_id).count()
            
            # Delete all user shifts
            session.query(Shift).filter(Shift.user_id == user_id).delete()
            session.commit()
            
            return shifts_count
            
        except Exception as e:
            session.rollback()
            raise ShiftServiceError(f"Error deleting user shifts: {str(e)}")
    
    @staticmethod
    def _parse_time(time_str: str) -> time:
        """
        Parse time string in HH:MM format to time object.
        
        Args:
            time_str: Time string in HH:MM format
            
        Returns:
            time object
            
        Raises:
            ShiftServiceError: If time format is invalid
        """
        try:
            time_parts = time_str.split(":")
            if len(time_parts) != 2:
                raise ValueError("Invalid format")
            
            hour = int(time_parts[0])
            minute = int(time_parts[1])
            
            if hour < 0 or hour > 23:
                raise ValueError("Hour must be between 0 and 23")
            if minute < 0 or minute > 59:
                raise ValueError("Minute must be between 0 and 59")
            
            return time(hour, minute)
            
        except ValueError as e:
            raise ShiftServiceError(f"Invalid time format '{time_str}'. Expected HH:MM format. {str(e)}")
    
    @staticmethod
    def _shifts_overlap(start1: time, end1: time, start2: time, end2: time) -> bool:
        """
        Check if two time ranges overlap.
        
        Args:
            start1: Start time of first shift
            end1: End time of first shift
            start2: Start time of second shift
            end2: End time of second shift
            
        Returns:
            True if shifts overlap, False otherwise
        """
        return start1 < end2 and start2 < end1