from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

# Import the ShiftDay enum directly from the models
from user_management.app.db.models.shiftDay import ShiftDay

class ShiftResponse(BaseModel):
    """Schema for shift response data"""
    id: int
    day: ShiftDay
    day_name: str
    start_time: str  # HH:MM format
    end_time: str  # HH:MM format


class ProfileCompletionResponse(BaseModel):
    """Schema for profile completion response"""
    completion_percentage: int = Field(..., ge=0, le=100, description="Profile completion percentage")
    completion_status: Dict[str, Any] = Field(..., description="Detailed completion status")
    missing_fields: List[str] = Field(..., description="List of fields that need to be completed")


class ProfileResponse(BaseModel):
    """Schema for profile response"""
    airbus_id: Optional[int] = None
    role: Optional[str] = None
    shifts: List[ShiftResponse] = []
    completion: ProfileCompletionResponse