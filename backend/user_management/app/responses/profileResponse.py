# user_management/app/responses/profile_tab.py
"""
Profile Tab Response Models

This module contains response models for profile tab operations.
"""

from pydantic import BaseModel
from typing import Optional
from user_management.app.db.models.roleType import RoleType
class DetailedWorkSummaryResponse (BaseModel):
    """Response model for detailed work summary"""
    
    shifts_by_day: dict
    total_weekly_hours: int
    total_weekly_minutes: int
    total_weekly_display: str
    average_daily_hours: str
    max_daily_hours: str
    min_daily_hours: str

    class Config:
        from_attributes = True

class ProfileTabInfoResponse(BaseModel):
    """Response model for profile tab information"""
    
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

    class Config:
        from_attributes = True


class ProfileTabUpdateResponse(BaseModel):
    """Response model for profile tab updates"""
    
    success: bool
    message: str
    updated_user: Optional[ProfileTabInfoResponse] = None

    class Config:
        from_attributes = True


class ProfileTabDeleteResponse(BaseModel):
    """Response model for profile deletion"""
    
    success: bool
    message: str

    class Config:
        from_attributes = True


class WorkSummaryResponse(BaseModel):
    """Response model for work summary"""
    
    shifts_by_day: dict
    total_weekly_hours: int
    total_weekly_minutes: int
    total_weekly_display: str

    class Config:
        from_attributes = True