from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from user_management.app.db.session import get_session
from user_management.app.core.security import get_current_user
from user_management.app.db.schemas.profile import ProfileUpdateRequest
from user_management.app.services import profile
from user_management.app.responses.profile import ProfileResponse, ProfileCompletionResponse ,ShiftResponse

# Create a router that requires authentication
profile_router = APIRouter(
    prefix="/profile",
    tags=["Profile"],
    responses={404: {"description": "Not found"}},
    dependencies=[Depends(get_current_user)]
)
guest_router = APIRouter(
    prefix="/auth",
    tags=["Auth"],
    responses={404: {"description": "Not found"}},
    dependencies=[Depends(get_current_user)]
)

@guest_router.get("/completion", status_code=status.HTTP_200_OK, response_model=ProfileCompletionResponse)
async def get_profile_completion(
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Get the profile completion percentage and status
    """
    return await profile.get_profile_completion(current_user.id, session)

@profile_router.put("/update", status_code=status.HTTP_200_OK)
async def update_profile(
    data: ProfileUpdateRequest,
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Update user profile information
    """
    await profile.update_user_profile(current_user.id, data, session)
    return {"message": "Profile updated successfully"}

@profile_router.get("/profile", status_code=status.HTTP_200_OK, response_model=ProfileResponse)
async def get_user_profile(
    current_user = Depends(get_current_user),
    session: Session = Depends(get_session)
):
    """
    Get the user's complete profile information including personal details,
    shifts, and profile completion status
    """
    # Get the user profile completion status
    completion_data = await profile.get_profile_completion(current_user.id, session)
    
    # Get the user's shifts
    shifts_data = await profile.get_user_shifts(current_user.id, session)
    
    # Format the shifts data according to the ShiftResponse model
    formatted_shifts = [
        ShiftResponse(
            id=shift["id"],
            day=shift["day"],
            day_name=shift["day_name"],
            start_time=shift["start_time"],
            end_time=shift["end_time"]
        ) for shift in shifts_data
    ]
    
    # Return the complete profile response
    return ProfileResponse(
        airbus_id=current_user.airbus_id,
        role=current_user.role.name if current_user.role else None,
        shifts=formatted_shifts,
        completion=completion_data
    )