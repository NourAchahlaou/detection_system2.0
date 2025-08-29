"""
Profile Router Module

This module defines API routes for profile-related operations.
"""

from fastapi import APIRouter, Depends, status, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import Optional

from user_management.app.db.session import get_session
from user_management.app.core.security import get_current_user
from user_management.app.services.accountservice import ProfileService, ProfileServiceError
from pydantic import BaseModel, Field


# Request/Response Models
class UpdateProfileRequest(BaseModel):
    name: Optional[str] = Field(None, description="User's name")
    email: Optional[str] = Field(None, description="User's email address")
    password: Optional[str] = Field(None, min_length=6, description="New password")
    airbus_id: Optional[int] = Field(None, gt=0, description="Airbus ID")
    role: Optional[str] = Field(None, description="User role")

    class Config:
        schema_extra = {
            "example": {
                "name": "John Doe",
                "email": "john.doe@example.com",
                "airbus_id": 12345,
                "role": "USER"
            }
        }


class ProfileResponse(BaseModel):
    id: int
    airbus_id: Optional[int]
    name: str
    email: str
    role: Optional[str]
    is_active: bool
    verified_at: Optional[str]
    created_at: Optional[str]
    updated_at: Optional[str]
    shifts_count: int


class BasicProfileResponse(BaseModel):
    id: int
    name: str
    email: str
    airbus_id: Optional[int]
    role: Optional[str]
    is_active: bool


# Protected profile endpoints
profile_router = APIRouter(
    prefix="/profile",
    tags=["Profile"],
    responses={404: {"description": "Not found"}},
)


@profile_router.get("/", status_code=status.HTTP_200_OK, response_model=ProfileResponse)
async def get_user_profile(
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Get complete user profile information."""
    try:
        profile_data = await ProfileService.get_user_profile(current_user.id, session)
        return ProfileResponse(**profile_data)
    except ProfileServiceError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@profile_router.get("/basic", status_code=status.HTTP_200_OK, response_model=BasicProfileResponse)
async def get_user_basic_info(
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Get basic user information (without sensitive data)."""
    try:
        basic_data = await ProfileService.get_user_basic_info(current_user.id, session)
        return BasicProfileResponse(**basic_data)
    except ProfileServiceError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@profile_router.put("/", status_code=status.HTTP_200_OK, response_model=ProfileResponse)
async def update_user_profile(
    data: UpdateProfileRequest,
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Update user profile with provided data."""
    try:
        updated_user = await ProfileService.update_user_profile(
            user_id=current_user.id,
            session=session,
            name=data.name,
            email=data.email,
            password=data.password,
            airbus_id=data.airbus_id,
            role=data.role
        )
        
        # Get updated profile data to return
        profile_data = await ProfileService.get_user_profile(current_user.id, session)
        return ProfileResponse(**profile_data)
        
    except ProfileServiceError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        elif "already in use" in str(e).lower():
            raise HTTPException(status_code=409, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


@profile_router.delete("/", status_code=status.HTTP_200_OK)
async def delete_user_account(
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Delete user account and all associated data."""
    try:
        success = await ProfileService.delete_user_account(current_user.id, session)
        if success:
            return JSONResponse({"message": "Account deleted successfully."})
        else:
            raise HTTPException(status_code=500, detail="Failed to delete account")
            
    except ProfileServiceError as e:
        if "not found" in str(e).lower():
            raise HTTPException(status_code=404, detail=str(e))
        else:
            raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")


# Admin endpoint to get any user's basic profile (optional)
@profile_router.get("/{user_id}", status_code=status.HTTP_200_OK, response_model=BasicProfileResponse)
async def get_user_basic_info_by_id(
    user_id: int,
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """Get basic user information by user ID (admin or self only)."""
    # Allow users to view their own profile or admin users to view any profile
    if current_user.id != user_id and (not current_user.role or current_user.role.value != 'ADMIN'):
        raise HTTPException(status_code=403, detail="Access denied")
    
    try:
        basic_data = await ProfileService.get_user_basic_info(user_id, session)
        return BasicProfileResponse(**basic_data)
    except ProfileServiceError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")