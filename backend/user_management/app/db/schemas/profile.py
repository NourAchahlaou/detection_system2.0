
from pydantic import BaseModel, Field, validator
from typing import List, Optional

# Import the ShiftDay enum directly from the models
from user_management.app.db.models.shiftDay import ShiftDay


class ShiftRequest(BaseModel):
    """Schema for shift request data"""
    start_time: str = Field(..., description="Shift start time in HH:MM format")
    end_time: str = Field(..., description="Shift end time in HH:MM format")
    day_of_week: ShiftDay = Field(..., description="Day of week (ShiftDay enum)")
    
    @validator('start_time', 'end_time')
    def validate_time_format(cls, v):
        """Validate time format is HH:MM"""
        try:
            parts = v.split(':')
            if len(parts) != 2:
                raise ValueError("Time must be in HH:MM format")
            
            hour, minute = int(parts[0]), int(parts[1])
            if not (0 <= hour <= 23 and 0 <= minute <= 59):
                raise ValueError("Invalid time values")
        except (ValueError, IndexError):
            raise ValueError("Time must be in valid HH:MM format")
        return v


class ProfileUpdateRequest(BaseModel):
    """Schema for profile update request"""
    airbus_id: Optional[int] = Field(None, description="User's Airbus ID")
    role: Optional[str] = Field(None, description="User's role in the system")
    main_shifts: Optional[List[ShiftRequest]] = Field(
        None, 
        description="List of user's regular shifts"
    )
    
    @validator('airbus_id')
    def validate_airbus_id(cls, v):
        """Validate airbus_id is a positive integer"""
        if v is not None and v <= 0:
            raise ValueError("Airbus ID must be a positive integer")
        return v

