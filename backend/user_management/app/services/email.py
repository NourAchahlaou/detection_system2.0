import datetime
import random
import string
from fastapi import BackgroundTasks, HTTPException
from user_management.app.core.settings import get_settings
from user_management.app.db.models.user import User
from user_management.app.core.email import send_email
from user_management.app.utils.email_context import USER_VERIFY_ACCOUNT, FORGOT_PASSWORD

settings = get_settings()


# Generate a random activation code
def generate_activation_code(length=6):
    characters = string.ascii_uppercase + string.digits
    return ''.join(random.choice(characters) for _ in range(length))

# Fixed send_account_verification_email function
async def send_account_verification_email(user, background_tasks, session=None):
    # Generate a code instead of a token
    activation_code = generate_activation_code()
    
    # Store the code in the database with expiration time
    user.activation_code = activation_code
    user.activation_code_expires_at = datetime.datetime.utcnow() + datetime.timedelta(hours=24)
    
    # Update the database - only if session is provided
    if session:
        session.add(user)
        session.commit()
    
    # Prepare email context with correct settings
    context = {
        "app_name": settings.APP_NAME,
        "name": user.name,
        "activation_code": activation_code,
        # Using FRONTEND_HOST instead of SITE_DOMAIN
        "activate_url": f"{settings.FRONTEND_HOST}/account/verify?email={user.email}&code={activation_code}"
    }
    
    # Send email
    await send_email(
        recipients=[user.email],
        subject=f"Verify your account on {settings.APP_NAME}",
        context=context,
        template_name="user/account-verification.html",
        background_tasks=background_tasks
    )

# Fixed activate_user_account_with_code function 
async def activate_user_account_with_code(email: str, code: str, session, background_tasks):
    user = session.query(User).filter(User.email == email).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid email address.")
    
    # Verify the activation code
    if not user.activation_code or user.activation_code != code:
        raise HTTPException(status_code=400, detail="Invalid activation code.")
    
    # Check if code is expired
    if not user.activation_code_expires_at or user.activation_code_expires_at < datetime.datetime.utcnow():
        raise HTTPException(status_code=400, detail="Activation code has expired.")
    
    # Activate the account
    user.is_active = True
    user.verified_at = datetime.datetime.utcnow()
    user.activation_code = None  # Clear the code
    user.activation_code_expires_at = None  # Clear the expiration
    user.updated_at = datetime.datetime.utcnow()
    
    session.add(user)
    session.commit()
    session.refresh(user)
    
    # Send confirmation email
    await send_account_activation_confirmation_email(user, background_tasks)
    
    return user

async def send_account_activation_confirmation_email(user: User, background_tasks: BackgroundTasks):
    data = {
        'app_name': settings.APP_NAME,
        "name": user.name,
        'login_url': f'{settings.FRONTEND_HOST}'
    }
    subject = f"Welcome - {settings.APP_NAME}"
    await send_email(
        recipients=[user.email],
        subject=subject,
        template_name="user/account-verification-confirmation.html",
        context=data,
        background_tasks=background_tasks
    )
    
async def send_password_reset_email(user: User, background_tasks: BackgroundTasks):
    from user_management.app.core.security import hash_password
    string_context = user.get_context_string(context=FORGOT_PASSWORD)
    token = hash_password(string_context)
    reset_url = f"{settings.FRONTEND_HOST}/reset-password?token={token}&email={user.email}"
    data = {
        'app_name': settings.APP_NAME,
        "name": user.name,
        'activate_url': reset_url,
    }
    subject = f"Reset Password - {settings.APP_NAME}"
    await send_email(
        recipients=[user.email],
        subject=subject,
        template_name="user/password-reset.html",
        context=data,
        background_tasks=background_tasks
    )