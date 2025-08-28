# user_management/app/api/routes/profile_tab.py
"""
Profile Tab Router

This module contains the API endpoints for profile tab operations.
"""

from fastapi import APIRouter, Depends, status, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from typing import List

from user_management.app.db.session import get_session
from user_management.app.core.security import get_current_user
from user_management.app.db.schemas.profileSchema import (
    ProfileTabUpdateRequest,
    ProfileTabResponse,
    ShiftUpdateRequest,
    BulkShiftUpdateRequest
)
from user_management.app.responses.profileResponse import (
    ProfileTabInfoResponse,
    ProfileTabUpdateResponse,
    ProfileTabDeleteResponse,
    WorkSummaryResponse,
    DetailedWorkSummaryResponse
)
from user_management.app.services.profile_service import profileService as profile_tab


# Router for profile tab operations (requires authentication)
profile_tab_router = APIRouter(
    prefix="/profile-tab",
    tags=["Profile Tab"],
    responses={404: {"description": "Not found"}},
    dependencies=[Depends(get_current_user)]
)


@profile_tab_router.get("/info", status_code=status.HTTP_200_OK, response_model=ProfileTabInfoResponse)
async def get_profile_tab_info(
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Get complete user profile information for the profile tab display
    """
    try:
        profile_data = await profile_tab.get_user_profile_tab_info(current_user.id, session)
        
        return ProfileTabInfoResponse(
            first_name=profile_data.first_name,
            last_name=profile_data.last_name,
            email=profile_data.email,
            role=profile_data.role,
            shift_start=profile_data.shift_start,
            shift_end=profile_data.shift_end,
            current_status=profile_data.current_status,
            total_hours_this_week=profile_data.total_hours_this_week,
            employee_id=profile_data.employee_id,
            work_area=profile_data.work_area,
            badge_number=profile_data.badge_number,
            access_level=profile_data.access_level
        )
        
    except profile_tab.ProfileTabError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get profile information: {str(e)}"
        )


@profile_tab_router.put("/update", status_code=status.HTTP_200_OK, response_model=ProfileTabUpdateResponse)
async def update_profile_tab_info(
    data: ProfileTabUpdateRequest,
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Update user profile information from the profile tab
    """
    try:
        updated_user = await profile_tab.update_user_profile_tab(current_user.id, data, session)
        
        # Get the updated profile info to return
        updated_profile_data = await profile_tab.get_user_profile_tab_info(current_user.id, session)
        
        updated_info = ProfileTabInfoResponse(
            first_name=updated_profile_data.first_name,
            last_name=updated_profile_data.last_name,
            email=updated_profile_data.email,
            role=updated_profile_data.role,
            shift_start=updated_profile_data.shift_start,
            shift_end=updated_profile_data.shift_end,
            current_status=updated_profile_data.current_status,
            total_hours_this_week=updated_profile_data.total_hours_this_week,
            employee_id=updated_profile_data.employee_id,
            work_area=updated_profile_data.work_area,
            badge_number=updated_profile_data.badge_number,
            access_level=updated_profile_data.access_level
        )
        
        return ProfileTabUpdateResponse(
            success=True,
            message="Profile updated successfully",
            updated_user=updated_info
        )
        
    except profile_tab.ProfileTabError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update profile: {str(e)}"
        )


@profile_tab_router.put("/shifts/update", status_code=status.HTTP_200_OK)
async def update_user_shifts(
    shifts_data: List[ShiftUpdateRequest],
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Update user shifts with support for individual shift modifications
    
    Example request body:
    [
        {
            "day_of_week": "MONDAY",
            "start_time": "09:00",
            "end_time": "17:00",
            "action": "update"
        },
        {
            "day_of_week": "SATURDAY",
            "start_time": "09:00", 
            "end_time": "13:00",
            "action": "create"
        },
        {
            "day_of_week": "SUNDAY",
            "start_time": "09:00",
            "end_time": "17:00", 
            "action": "delete"
        }
    ]
    """
    try:
        success = await profile_tab.update_user_shifts(current_user.id, shifts_data, session)
        
        if success:
            # Get updated shift summary
            work_summary = await profile_tab.get_user_detailed_shifts(current_user.id, session)
            
            return JSONResponse({
                "success": True,
                "message": f"Successfully updated {len(shifts_data)} shift(s)",
                "work_summary": work_summary
            })
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update shifts"
            )
            
    except profile_tab.ProfileTabError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update shifts: {str(e)}"
        )


@profile_tab_router.put("/shifts/bulk-update", status_code=status.HTTP_200_OK)
async def bulk_update_regular_shifts(
    bulk_data: BulkShiftUpdateRequest,
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Update regular shifts for multiple days with the same time
    
    Example request body:
    {
        "start_time": "09:00",
        "end_time": "17:00",
        "days": ["MONDAY", "TUESDAY", "WEDNESDAY", "THURSDAY", "FRIDAY"]
    }
    """
    try:
        success = await profile_tab.bulk_update_regular_shifts(
            current_user.id,
            bulk_data.start_time,
            bulk_data.end_time,
            bulk_data.days,
            session
        )
        
        if success:
            # Get updated shift summary
            work_summary = await profile_tab.get_user_detailed_shifts(current_user.id, session)
            
            return JSONResponse({
                "success": True,
                "message": f"Successfully updated regular shifts for {len(bulk_data.days)} day(s)",
                "work_summary": work_summary
            })
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to update regular shifts"
            )
            
    except profile_tab.ProfileTabError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update regular shifts: {str(e)}"
        )


@profile_tab_router.get("/shifts/detailed", status_code=status.HTTP_200_OK, response_model=DetailedWorkSummaryResponse)
async def get_detailed_shifts(
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Get detailed user shifts organized by day with additional metadata
    """
    try:
        detailed_summary = await profile_tab.get_user_detailed_shifts(current_user.id, session)
        
        return DetailedWorkSummaryResponse(     
            shifts_by_day=detailed_summary["shifts_by_day"],
            total_weekly_hours=detailed_summary["total_weekly_hours"],  
            total_weekly_minutes=detailed_summary["total_weekly_minutes"],
            total_weekly_display=detailed_summary["total_weekly_display"],
            has_weekend_shifts=detailed_summary["has_weekend_shifts"],
            weekday_count=detailed_summary["weekday_count"]
        )
    except profile_tab.ProfileTabError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get detailed shifts: {str(e)}"
        )

   


@profile_tab_router.delete("/delete", status_code=status.HTTP_200_OK, response_model=ProfileTabDeleteResponse)
async def delete_user_profile(
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Delete user profile (soft delete - sets user as inactive)
    """
    try:
        success = await profile_tab.delete_user_profile(current_user.id, session)
        
        if success:
            return ProfileTabDeleteResponse(
                success=True,
                message="User profile deleted successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to delete user profile"
            )
            
    except profile_tab.ProfileTabError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete profile: {str(e)}"
        )


@profile_tab_router.get("/work-summary", status_code=status.HTTP_200_OK, response_model=WorkSummaryResponse)
async def get_work_summary(
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Get user work summary including shifts and total hours
    """
    try:
        work_summary = await profile_tab.get_user_work_summary(current_user.id, session)
        
        return WorkSummaryResponse(
            shifts_by_day=work_summary["shifts_by_day"],
            total_weekly_hours=work_summary["total_weekly_hours"],
            total_weekly_minutes=work_summary["total_weekly_minutes"],
            total_weekly_display=work_summary["total_weekly_display"]
        )
        
    except profile_tab.ProfileTabError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get work summary: {str(e)}"
        )


# Additional endpoint for specific field updates (optional)
@profile_tab_router.patch("/update-field", status_code=status.HTTP_200_OK)
async def update_profile_field(
    field_name: str,
    field_value: str,
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Update a specific profile field
    """
    try:
        # Create a partial update request
        update_data = {}
        
        # Map field names to schema fields
        field_mapping = {
            'first_name': 'first_name',
            'last_name': 'last_name',
            'email': 'email',
            'role': 'role',
            'shift_start': 'shift_start',
            'shift_end': 'shift_end',
            'airbus_id': 'airbus_id'
        }
        
        if field_name not in field_mapping:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid field name: {field_name}"
            )
        
        # Convert airbus_id to integer if needed
        if field_name == 'airbus_id':
            try:
                field_value = int(field_value)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="airbus_id must be a valid integer"
                )
        
        update_data[field_mapping[field_name]] = field_value
        
        # Create update request
        update_request = ProfileTabUpdateRequest(**update_data)
        
        # Update the profile
        await profile_tab.update_user_profile_tab(current_user.id, update_request, session)
        
        return JSONResponse({
            "success": True,
            "message": f"Field '{field_name}' updated successfully"
        })
        
    except profile_tab.ProfileTabError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update field: {str(e)}"
        )