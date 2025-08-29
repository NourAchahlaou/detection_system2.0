"""
Shift Router Module

This module defines API routes for shift-related operations.
"""

from fastapi import APIRouter, Depends, status, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List, Optional

from user_management.app.db.session import get_session
from user_management.app.core.security import get_current_user
from user_management.app.services.accountShiftService import ShiftService, ShiftServiceError
from pydantic import BaseModel, Field, validator
import re


# Request/Response Models
class CreateShiftRequest(BaseModel):
    day_of_week: str = Field(..., description="Day of the week (e.g., 'MONDAY')")
    start_time: str = Field(..., description="Start time in HH:MM format")
    end_time: str = Field(..., description="End time in HH:MM format")
    
    @validator('day_of_week')
    def validate_day_of_week(cls, v):
        valid_days = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']
        if v.upper() not in valid_days:
            raise ValueError(f'Invalid day. Must be one of: {", ".join(valid_days)}')
        return v.upper()
    
    @validator('start_time', 'end_time')
    def validate_time_format(cls, v):
        if not re.match(r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$', v):
            raise ValueError('Time must be in HH:MM format (e.g., "09:30", "14:00")')
        return v

    class Config:
        schema_extra = {
            "example": {
                "day_of_week": "MONDAY",
                "start_time": "09:00",
                "end_time": "17:00"
            }
        }


class UpdateShiftRequest(BaseModel):
    day_of_week: Optional[str] = Field(None, description="Day of the week")
    start_time: Optional[str] = Field(None, description="Start time in HH:MM format")
    end_time: Optional[str] = Field(None, description="End time in HH:MM format")
    
    @validator('day_of_week')
    def validate_day_of_week(cls, v):
        if v is not None:
            valid_days = ['MONDAY', 'TUESDAY', 'WEDNESDAY', 'THURSDAY', 'FRIDAY', 'SATURDAY', 'SUNDAY']
            if v.upper() not in valid_days:
                raise ValueError(f'Invalid day. Must be one of: {", ".join(valid_days)}')
            return v.upper()
        return v
    
    @validator('start_time', 'end_time')
    def validate_time_format(cls, v):
        if v is not None and not re.match(r'^([0-1]?[0-9]|2[0-3]):[0-5][0-9]$', v):
            raise ValueError('Time must be in HH:MM format (e.g., "09:30", "14:00")')
        return v

    class Config:
        schema_extra = {
            "example": {
                "day_of_week": "TUESDAY",
                "start_time": "10:00",
                "end_time": "18:00"
            }
        }


class ShiftResponse(BaseModel):
    id: int
    user_id: int
    day_of_week: str
    day_name: str
    start_time: str
    end_time: str
    created_at: Optional[str]
    updated_at: Optional[str]


# Protected shift endpoints
shift_router = APIRouter(
    prefix="/shifts",
    tags=["Shifts"],
    responses={404: {"description": "Not found"}},
)


@shift_router.get("/", status_code=status.HTTP_200_OK, response_model=List[ShiftResponse])
async def get_user_shifts(
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Get all shifts for the current user."""
    try:
        shifts_data = await ShiftService.get_all_user_shifts(current_user.id, session)
        return [ShiftResponse(**shift) for shift in shifts_data]
    except ShiftServiceError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@shift_router.get("/{shift_id}", status_code=status.HTTP_200_OK, response_model=ShiftResponse)
async def get_shift(
    shift_id: int,
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Get a specific shift by ID."""
    try:
        shift_data = await ShiftService.get_shift_by_id(shift_id, session)
        
        # Ensure user can only access their own shifts
        if shift_data["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        return ShiftResponse(**shift_data)
    except ShiftServiceError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@shift_router.post("/", status_code=status.HTTP_201_CREATED, response_model=ShiftResponse)
async def create_shift(
    data: CreateShiftRequest,
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Create a new shift for the current user."""
    try:
        created_shift = await ShiftService.create_shift(
            user_id=current_user.id,
            day_of_week=data.day_of_week,
            start_time=data.start_time,
            end_time=data.end_time,
            session=session
        )
        
        # Get the formatted shift data to return
        shift_data = await ShiftService.get_shift_by_id(created_shift.id, session)
        return ShiftResponse(**shift_data)
        
    except ShiftServiceError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        elif "overlap" in str(e).lower():
            raise HTTPException(status_code=409, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@shift_router.put("/{shift_id}", status_code=status.HTTP_200_OK, response_model=ShiftResponse)
async def update_shift(
    shift_id: int,
    data: UpdateShiftRequest,
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Update an existing shift."""
    try:
        # First, verify the shift belongs to the current user
        shift_data = await ShiftService.get_shift_by_id(shift_id, session)
        if shift_data["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        updated_shift = await ShiftService.update_shift(
            shift_id=shift_id,
            session=session,
            day_of_week=data.day_of_week,
            start_time=data.start_time,
            end_time=data.end_time
        )
        
        # Get the updated shift data to return
        updated_shift_data = await ShiftService.get_shift_by_id(shift_id, session)
        return ShiftResponse(**updated_shift_data)
        
    except ShiftServiceError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        elif "overlap" in str(e).lower():
            raise HTTPException(status_code=409, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@shift_router.delete("/{shift_id}", status_code=status.HTTP_200_OK)
async def delete_shift(
    shift_id: int,
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Delete a specific shift."""
    try:
        # First, verify the shift belongs to the current user
        shift_data = await ShiftService.get_shift_by_id(shift_id, session)
        if shift_data["user_id"] != current_user.id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        success = await ShiftService.delete_shift(shift_id, session)
        if success:
            return JSONResponse({"message": "Shift deleted successfully."})
        else:
            raise HTTPException(status_code=500, detail="Failed to delete shift")
            
    except ShiftServiceError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@shift_router.delete("/", status_code=status.HTTP_200_OK)
async def delete_all_user_shifts(
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Delete all shifts for the current user."""
    try:
        deleted_count = await ShiftService.delete_all_user_shifts(current_user.id, session)
        return JSONResponse({
            "message": f"Successfully deleted {deleted_count} shift(s).",
            "deleted_count": deleted_count
        })
        
    except ShiftServiceError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


# Admin endpoint to get any user's shifts (optional)
@shift_router.get("/user/{user_id}", status_code=status.HTTP_200_OK, response_model=List[ShiftResponse])
async def get_user_shifts_by_id(
    user_id: int,
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Get all shifts for a specific user (admin only or self)."""
    # Allow users to view their own shifts or admin users to view any user's shifts
    if current_user.id != user_id and (not current_user.role or current_user.role.value != 'ADMIN'):
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        shifts_data = await ShiftService.get_all_user_shifts(user_id, session)
        return [ShiftResponse(**shift) for shift in shifts_data]
    except ShiftServiceError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")