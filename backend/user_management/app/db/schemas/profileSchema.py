# user_management/app/db/schemas/profileSchema.py
"""
Profile Tab Pydantic Schemas

This module contains the Pydantic models for profile tab requests and responses.
"""

from pydantic import BaseModel, EmailStr, Field, validator
from typing import Optional, List, Literal
from user_management.app.db.models.roleType import RoleType
from user_management.app.db.models.shiftDay import ShiftDay


class ShiftUpdateRequest(BaseModel):
    """Schema for individual shift updates"""
    
    day_of_week: ShiftDay
    start_time: str = Field(..., pattern="^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    end_time: str = Field(..., pattern="^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    action: Literal["create", "update", "delete"] = "update"
    
    @validator('end_time')
    def validate_shift_times(cls, v, values):
        if 'start_time' in values and v:
            start_hour, start_min = map(int, values['start_time'].split(':'))
            end_hour, end_min = map(int, v.split(':'))
            
            start_minutes = start_hour * 60 + start_min
            end_minutes = end_hour * 60 + end_min
            
            # Allow overnight shifts
            if end_minutes <= start_minutes and end_hour <= start_hour:
                # This might be an overnight shift, which is valid
                pass
            
        return v

    class Config:
        use_enum_values = True


class BulkShiftUpdateRequest(BaseModel):
    """Schema for bulk shift updates (regular shifts)"""
    
    start_time: str = Field(..., pattern="^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    end_time: str = Field(..., pattern="^([01]?[0-9]|2[0-3]):[0-5][0-9]$")
    days: List[ShiftDay] = Field(..., min_items=1, max_items=7)
    
    class Config:
        use_enum_values = True


class ProfileTabUpdateRequest(BaseModel):
    """Schema for updating user profile from profile tab"""
    
    # Personal Information
    first_name: Optional[str] = Field(None, min_length=1, max_length=50)
    last_name: Optional[str] = Field(None, min_length=1, max_length=50)
    email: Optional[EmailStr] = None
    role: Optional[str] = Field(None, pattern="^(DATA_MANAGER|AUDITOR|OPERATOR)$")
    
    # Access & Credentials
    airbus_id: Optional[int] = Field(None, gt=0)
    
    @validator('role')
    def validate_role(cls, v):
        if v is not None:
            v = v.upper()
            if v not in [role.name for role in RoleType]:
                raise ValueError(f'Role must be one of: {[role.name for role in RoleType]}')
        return v

    class Config:
        json_encoders = {
            # Custom encoders if needed
        }


class ProfileTabResponse(BaseModel):
    """Schema for profile tab response data"""
    
    # Personal Information
    first_name: str
    last_name: str
    email: str
    role: str
    shift_start: str
    shift_end: str
    current_status: str
    total_hours_this_week: str
    
    # Access & Credentials  
    employee_id: str
    work_area: str
    badge_number: str
    access_level: str
    
    # Additional fields for internal use
    user_id: int
    airbus_id: Optional[int]
    role_enum: Optional[RoleType]
    is_active: bool

    class Config:
        from_attributes = True
        json_encoders = {
            RoleType: lambda v: v.name if v else None
        }


class WorkSummaryResponse(BaseModel):
    """Schema for work summary response"""
    
    shifts_by_day: dict
    total_weekly_hours: int
    total_weekly_minutes: int
    total_weekly_display: str

    class Config:
        from_attributes = True


class DetailedWorkSummaryResponse(BaseModel):
    """Schema for detailed work summary response"""
    
    shifts_by_day: dict
    total_weekly_hours: int
    total_weekly_minutes: int
    total_weekly_display: str
    has_weekend_shifts: bool
    weekday_count: int

    class Config:
        from_attributes = True


class ProfileTabDeleteResponse(BaseModel):
    """Schema for profile deletion response"""
    
    success: bool
    message: str

    class Config:
        from_attributes = True