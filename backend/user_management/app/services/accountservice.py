"""
Profile Service Module

This module handles all profile-related operations including getting user info,
updating profile details, and deleting accounts.
"""

from datetime import datetime
from typing import Optional, Dict, Any
from sqlalchemy.orm import Session
from sqlalchemy import and_
from passlib.context import CryptContext

from user_management.app.db.models.user import User, UserToken
from user_management.app.db.models.roleType import RoleType
from user_management.app.db.models.shift import Shift

# Create password context for hashing
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class ProfileServiceError(Exception):
    """Exception raised for errors in profile service operations."""
    pass


class ProfileService:
    """Service class for handling profile operations."""
    
    @staticmethod
    async def get_user_profile(user_id: int, session: Session) -> Dict[str, Any]:
        """
        Get complete user profile information.
        
        Args:
            user_id: The ID of the user
            session: Database session
            
        Returns:
            Dictionary containing user profile information
            
        Raises:
            ProfileServiceError: If user is not found
        """
        user = session.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise ProfileServiceError("User not found")
        
        # Get user shifts count for additional info
        shifts_count = session.query(Shift).filter(Shift.user_id == user_id).count()
        
        return {
            "id": user.id,
            "airbus_id": user.airbus_id,
            "name": user.name,
            "email": user.email,
            "role": user.role.value if user.role else None,
            "is_active": user.is_active,
            "verified_at": user.verified_at.isoformat() if user.verified_at else None,
            "created_at": user.created_at.isoformat() if user.created_at else None,
            "updated_at": user.updated_at.isoformat() if user.updated_at else None,
            "shifts_count": shifts_count
        }
    
    @staticmethod
    async def update_user_profile(
        user_id: int, 
        session: Session,
        name: Optional[str] = None,
        email: Optional[str] = None,
        password: Optional[str] = None,
        airbus_id: Optional[int] = None,
        role: Optional[str] = None
    ) -> User:
        """
        Update user profile with provided data.
        
        Args:
            user_id: The ID of the user to update
            session: Database session
            name: New name (optional)
            email: New email (optional)
            password: New password (optional)
            airbus_id: New airbus ID (optional)
            role: New role (optional)
            
        Returns:
            The updated user object
            
        Raises:
            ProfileServiceError: If user is not found or validation fails
        """
        user = session.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise ProfileServiceError("User not found")
        
        try:
            # Update name if provided
            if name is not None:
                if len(name.strip()) == 0:
                    raise ProfileServiceError("Name cannot be empty")
                user.name = name.strip()
            
            # Update email if provided
            if email is not None:
                email = email.strip().lower()
                if len(email) == 0:
                    raise ProfileServiceError("Email cannot be empty")
                
                # Check if email is already in use by another user
                existing_user = session.query(User).filter(
                    and_(User.email == email, User.id != user_id)
                ).first()
                
                if existing_user:
                    raise ProfileServiceError("Email is already in use")
                
                user.email = email
            
            # Update password if provided
            if password is not None:
                if len(password) < 6:
                    raise ProfileServiceError("Password must be at least 6 characters long")
                # Use passlib instead of werkzeug
                user.password = pwd_context.hash(password)
            
            # Update airbus_id if provided
            if airbus_id is not None:
                if airbus_id <= 0:
                    raise ProfileServiceError("Airbus ID must be a positive number")
                
                # Check if airbus_id is already in use by another user
                existing_user = session.query(User).filter(
                    and_(User.airbus_id == airbus_id, User.id != user_id)
                ).first()
                
                if existing_user:
                    raise ProfileServiceError("Airbus ID is already in use")
                
                user.airbus_id = airbus_id
            
            # Update role if provided
            if role is not None:
                try:
                    user.role = RoleType[role.upper()]
                except KeyError:
                    valid_roles = [r.value for r in RoleType]
                    raise ProfileServiceError(f"Invalid role. Valid roles are: {', '.join(valid_roles)}")
            
            # Update timestamp
            user.updated_at = datetime.now()
            
            # Commit changes
            session.commit()
            
        except ProfileServiceError:
            session.rollback()
            raise
        except Exception as e:
            session.rollback()
            raise ProfileServiceError(f"Error updating profile: {str(e)}")
        
        return user
    
    @staticmethod
    async def delete_user_account(user_id: int, session: Session) -> bool:
        """
        Delete a user account and all associated data.
        
        Args:
            user_id: The ID of the user to delete
            session: Database session
            
        Returns:
            True if deletion was successful
            
        Raises:
            ProfileServiceError: If user is not found or deletion fails
        """
        user = session.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise ProfileServiceError("User not found")
        
        try:
            # Delete user tokens
            session.query(UserToken).filter(UserToken.user_id == user_id).delete()
            
            # Delete user shifts
            session.query(Shift).filter(Shift.user_id == user_id).delete()
            
            # Delete the user
            session.delete(user)
            
            # Commit all deletions
            session.commit()
            
            return True
            
        except Exception as e:
            session.rollback()
            raise ProfileServiceError(f"Error deleting user account: {str(e)}")
    
    @staticmethod
    async def get_user_basic_info(user_id: int, session: Session) -> Dict[str, Any]:
        """
        Get basic user information (without sensitive data).
        
        Args:
            user_id: The ID of the user
            session: Database session
            
        Returns:
            Dictionary containing basic user information
            
        Raises:
            ProfileServiceError: If user is not found
        """
        user = session.query(User).filter(User.id == user_id).first()
        
        if not user:
            raise ProfileServiceError("User not found")
        
        return {
            "id": user.id,
            "name": user.name,
            "email": user.email,
            "airbus_id": user.airbus_id,
            "role": user.role.value if user.role else None,
            "is_active": user.is_active
        }

    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verify a password against its hash.
        
        Args:
            plain_password: The plain text password
            hashed_password: The hashed password to verify against
            
        Returns:
            True if password matches, False otherwise
        """
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def hash_password(password: str) -> str:
        """
        Hash a password.
        
        Args:
            password: The plain text password to hash
            
        Returns:
            The hashed password
        """
        return pwd_context.hash(password)