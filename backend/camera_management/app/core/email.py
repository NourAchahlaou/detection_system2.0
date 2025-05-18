import os
from pathlib import Path
from fastapi_mail import FastMail, MessageSchema, MessageType, ConnectionConfig
from fastapi.background import BackgroundTasks
from camera_management.app.core.settings import get_settings

settings = get_settings()

conf = ConnectionConfig(
    # Use full email address as username 
    MAIL_USERNAME=os.environ.get("MAIL_USERNAME", "achahlaou.nour@gmail.com"),
    MAIL_PASSWORD=os.environ.get("MAIL_PASSWORD", "trcm xmrg imjm ilrr"),  # Your app password
    MAIL_PORT=os.environ.get("MAIL_PORT", 587),
    MAIL_SERVER=os.environ.get("MAIL_SERVER", "smtp.gmail.com"),
    MAIL_STARTTLS=os.environ.get("MAIL_STARTTLS", True),
    MAIL_SSL_TLS=os.environ.get("MAIL_SSL_TLS", False),
    MAIL_DEBUG=True,
    MAIL_FROM=os.environ.get("MAIL_FROM", 'achahlaou.nour@gmail.com'),
    MAIL_FROM_NAME=os.environ.get("MAIL_FROM_NAME", settings.APP_NAME),
    TEMPLATE_FOLDER=Path(__file__).parent.parent / "templates",
    USE_CREDENTIALS=os.environ.get("USE_CREDENTIALS", True),
    VALIDATE_CERTS=os.environ.get("VALIDATE_CERTS", False)
)

fm = FastMail(conf)

async def send_email(recipients: list, subject: str, context: dict, template_name: str,
                     background_tasks: BackgroundTasks):
    message = MessageSchema(
        subject=subject,
        recipients=recipients,
        template_body=context,
        subtype=MessageType.html
    )
    
    background_tasks.add_task(fm.send_message, message, template_name=template_name)