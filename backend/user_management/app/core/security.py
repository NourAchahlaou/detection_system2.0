import logging
from fastapi import Depends, HTTPException
import jwt
from passlib.context import CryptContext
import base64
from sqlalchemy.orm import joinedload, Session
from datetime import datetime, timedelta

from user_management.app.core.customAuth import OAuth2EmailBearer
from user_management.app.db.session import get_session
from user_management.app.core.settings import get_settings
from user_management.app.db.models.user import UserToken, User


SPECIAL_CHARACTERS = ['@', '#', '$', '%', '=', ':', '?', '.', '/', '|', '~', '>']

settings = get_settings()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
# Changed this to reference auth/login instead of users/login
oauth2_scheme = OAuth2EmailBearer(tokenUrl="/auth/login")


def hash_password(password):
    return pwd_context.hash(password)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def is_password_strong_enough(password: str) -> bool:
    if len(password) < 8:
        return False

    if not any(char.isupper() for char in password):
        return False

    if not any(char.islower() for char in password):
        return False

    if not any(char.isdigit() for char in password):
        return False

    if not any(char in SPECIAL_CHARACTERS for char in password):
        return False

    return True


def str_encode(string: str) -> str:
    return base64.b85encode(string.encode('ascii')).decode('ascii')


def str_decode(string: str) -> str:
    return base64.b85decode(string.encode('ascii')).decode('ascii')


def get_token_payload(token: str, secret: str, algo: str):
    try:
        payload = jwt.decode(token, secret, algorithms=algo)
    except Exception as jwt_exec:
        logging.debug(f"JWT Error: {str(jwt_exec)}")
        payload = None
    return payload


def generate_token(payload: dict, secret: str, algo: str, expiry: timedelta):
    expire = datetime.utcnow() + expiry
    payload.update({"exp": expire})
    return jwt.encode(payload, secret, algorithm=algo)


async def get_token_user(token: str, db):
    try:
        payload = get_token_payload(token, settings.JWT_SECRET, settings.JWT_ALGORITHM)
        if payload:
            user_token_id = str_decode(payload.get('r'))
            user_id = str_decode(payload.get('sub'))
            access_key = payload.get('a')
            user_token = db.query(UserToken).options(joinedload(UserToken.user)).filter(UserToken.access_key == access_key,
                                                    UserToken.id == user_token_id,
                                                    UserToken.user_id == user_id,
                                                    UserToken.expires_at > datetime.utcnow()
                                                    ).first()
            if user_token:
                return user_token.user
    except Exception as e:
        logging.error(f"Error getting token user: {str(e)}")
    return None


async def load_user(email: str, db):
    try:
        user = db.query(User).filter(User.email == email).first()
    except Exception as user_exec:
        logging.info(f"User Not Found, Email: {email}")
        user = None
    return user


async def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_session)):
    user = await get_token_user(token=token, db=db)
    if user:
        return user
    raise HTTPException(status_code=401, detail="Not authorized.")