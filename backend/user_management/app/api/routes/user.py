# user_management/app/api/routes/user.py
from fastapi import APIRouter, BackgroundTasks, Depends, status, Header
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from user_management.app.core.customAuth import EmailPasswordRequestForm
from user_management.app.db.session import get_session
from user_management.app.responses.user import UserResponse, LoginResponse
from user_management.app.db.schemas.user import LoginRequest, RegisterUserRequest, ResetRequest, VerifyUserRequest, EmailRequest
from user_management.app.services import user
from user_management.app.core.security import get_current_user, oauth2_scheme

# Public user endpoints (non-auth)
user_router = APIRouter(
    prefix="/users",
    tags=["Users"],
    responses={404: {"description": "Not found"}},
)

# Protected user endpoints
auth_router = APIRouter(
    prefix="/users",
    tags=["Users"],
    responses={404: {"description": "Not found"}},
)

# Auth endpoints for login, refresh, etc.
guest_router = APIRouter(
    prefix="/auth",
    tags=["Auth"],
    responses={404: {"description": "Not found"}},
)

@user_router.post("/register", status_code=status.HTTP_201_CREATED, response_model=UserResponse)
async def register_user(data: RegisterUserRequest, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    return await user.create_user_account(data, session, background_tasks)

@user_router.post("/verify", status_code=status.HTTP_200_OK)
async def verify_user_account(data: VerifyUserRequest, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    await user.activate_user_account(data, session, background_tasks)
    return JSONResponse({"message": "Account is activated successfully."})

@user_router.post("/resend-verification", status_code=status.HTTP_201_CREATED, response_model=UserResponse)
async def resend_verification_code(data: EmailRequest, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    return await user.resend_verification_code(data, session, background_tasks)

@guest_router.post("/login", status_code=status.HTTP_200_OK, response_model=LoginResponse)
async def user_login(
    form_data: EmailPasswordRequestForm = Depends(),
    session: Session = Depends(get_session)
):
    login_data = LoginRequest(
        email=form_data.email,
        password=form_data.password
    )
    return await user.get_login_token(login_data, session)

@guest_router.post("/refresh", status_code=status.HTTP_200_OK, response_model=LoginResponse)
async def refresh_token(refresh_token: str = Header(None), session: Session = Depends(get_session)):
    return await user.get_refresh_token(refresh_token, session)

@guest_router.post("/forgot-password", status_code=status.HTTP_200_OK)
async def forgot_password(data: EmailRequest, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    await user.email_forgot_password_link(data, background_tasks, session)
    return JSONResponse({"message": "A email with password reset link has been sent to you."})

@guest_router.put("/reset-password", status_code=status.HTTP_200_OK)
async def reset_password(data: ResetRequest, session: Session = Depends(get_session)):
    await user.reset_user_password(data, session)
    return JSONResponse({"message": "Your password has been updated."})

# Protected endpoints
@auth_router.get("/me", status_code=status.HTTP_200_OK, response_model=UserResponse)
async def fetch_user(current_user = Depends(get_current_user)):
    return current_user

@auth_router.get("/{pk}", status_code=status.HTTP_200_OK, response_model=UserResponse)
async def get_user_info(pk, session: Session = Depends(get_session), current_user = Depends(get_current_user)):
    return await user.fetch_user_detail(pk, session)
