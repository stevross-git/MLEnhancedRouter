"""
Authentication System for AI Model Router
Supports JWT tokens, API keys, and user management
"""

import jwt
import hashlib
import os
import time
import secrets
from datetime import datetime, timedelta
from typing import Dict, Optional, Any
from dataclasses import dataclass
from enum import Enum

class UserRole(Enum):
    ADMIN = "admin"
    USER = "user"
    READONLY = "readonly"

@dataclass
class User:
    """User representation"""
    id: str
    username: str
    email: str
    role: UserRole
    api_key: str
    created_at: datetime
    last_login: Optional[datetime] = None
    is_active: bool = True
    permissions: Dict[str, bool] = None
    
    def __post_init__(self):
        if self.permissions is None:
            self.permissions = self._get_default_permissions()
    
    def _get_default_permissions(self) -> Dict[str, bool]:
        """Get default permissions based on role"""
        if self.role == UserRole.ADMIN:
            return {
                "create_models": True,
                "delete_models": True,
                "manage_agents": True,
                "view_analytics": True,
                "manage_users": True,
                "system_config": True
            }
        elif self.role == UserRole.USER:
            return {
                "create_models": True,
                "delete_models": False,
                "manage_agents": True,
                "view_analytics": True,
                "manage_users": False,
                "system_config": False
            }
        else:  # READONLY
            return {
                "create_models": False,
                "delete_models": False,
                "manage_agents": False,
                "view_analytics": True,
                "manage_users": False,
                "system_config": False
            }

class AuthManager:
    """Authentication and authorization manager"""
    
    def __init__(self, secret_key: str = None):
        self.secret_key = secret_key or os.getenv("JWT_SECRET", "dev-secret-change-in-production")
        self.users: Dict[str, User] = {}
        self.api_keys: Dict[str, str] = {}  # api_key -> user_id
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self._initialize_default_users()
    
    def _initialize_default_users(self):
        """Initialize default admin user"""
        admin_api_key = self._generate_api_key()
        admin_user = User(
            id="admin",
            username="admin",
            email="admin@localhost",
            role=UserRole.ADMIN,
            api_key=admin_api_key,
            created_at=datetime.utcnow()
        )
        
        self.users["admin"] = admin_user
        self.api_keys[admin_api_key] = "admin"
    
    def _generate_api_key(self) -> str:
        """Generate a secure API key"""
        return f"mr_{secrets.token_urlsafe(32)}"
    
    def _hash_password(self, password: str) -> str:
        """Hash a password with salt"""
        salt = os.urandom(32)
        pwdhash = hashlib.pbkdf2_hmac('sha256', password.encode('utf-8'), salt, 100000)
        return salt + pwdhash
    
    def _verify_password(self, stored_password: bytes, provided_password: str) -> bool:
        """Verify a password against its hash"""
        salt = stored_password[:32]
        stored_hash = stored_password[32:]
        pwdhash = hashlib.pbkdf2_hmac('sha256', provided_password.encode('utf-8'), salt, 100000)
        return pwdhash == stored_hash
    
    def create_user(self, username: str, email: str, password: str, role: UserRole = UserRole.USER) -> User:
        """Create a new user"""
        user_id = secrets.token_urlsafe(16)
        api_key = self._generate_api_key()
        
        user = User(
            id=user_id,
            username=username,
            email=email,
            role=role,
            api_key=api_key,
            created_at=datetime.utcnow()
        )
        
        self.users[user_id] = user
        self.api_keys[api_key] = user_id
        
        return user
    
    def authenticate_user(self, username: str, password: str) -> Optional[User]:
        """Authenticate user with username/password"""
        for user in self.users.values():
            if user.username == username and user.is_active:
                # In a real implementation, you'd verify password here
                user.last_login = datetime.utcnow()
                return user
        return None
    
    def authenticate_api_key(self, api_key: str) -> Optional[User]:
        """Authenticate user with API key"""
        user_id = self.api_keys.get(api_key)
        if user_id:
            user = self.users.get(user_id)
            if user and user.is_active:
                return user
        return None
    
    def generate_jwt_token(self, user_id: str, expires_in: int = 3600) -> str:
        """Generate JWT token for user"""
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(seconds=expires_in),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, self.secret_key, algorithm="HS256")
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except jwt.ExpiredSignatureError:
            return None
        except jwt.InvalidTokenError:
            return None
    
    def get_user_from_token(self, token: str) -> Optional[User]:
        """Get user from JWT token"""
        payload = self.verify_jwt_token(token)
        if payload:
            user_id = payload.get("user_id")
            return self.users.get(user_id)
        return None
    
    def has_permission(self, user: User, permission: str) -> bool:
        """Check if user has specific permission"""
        return user.permissions.get(permission, False)
    
    def require_permission(self, user: User, permission: str) -> bool:
        """Require specific permission, raise exception if not granted"""
        if not self.has_permission(user, permission):
            raise PermissionError(f"User {user.username} does not have permission: {permission}")
        return True
    
    def get_all_users(self) -> list:
        """Get all users (admin only)"""
        return list(self.users.values())
    
    def deactivate_user(self, user_id: str) -> bool:
        """Deactivate user"""
        user = self.users.get(user_id)
        if user:
            user.is_active = False
            return True
        return False
    
    def regenerate_api_key(self, user_id: str) -> Optional[str]:
        """Regenerate API key for user"""
        user = self.users.get(user_id)
        if user:
            # Remove old API key
            if user.api_key in self.api_keys:
                del self.api_keys[user.api_key]
            
            # Generate new API key
            new_api_key = self._generate_api_key()
            user.api_key = new_api_key
            self.api_keys[new_api_key] = user_id
            
            return new_api_key
        return None
    
    def create_session(self, user_id: str, session_data: Dict[str, Any] = None) -> str:
        """Create a user session"""
        session_id = secrets.token_urlsafe(32)
        self.sessions[session_id] = {
            "user_id": user_id,
            "created_at": datetime.utcnow(),
            "data": session_data or {}
        }
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session data"""
        return self.sessions.get(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """Delete session"""
        if session_id in self.sessions:
            del self.sessions[session_id]
            return True
        return False
    
    def cleanup_expired_sessions(self, max_age: int = 86400):
        """Clean up expired sessions (older than max_age seconds)"""
        cutoff_time = datetime.utcnow() - timedelta(seconds=max_age)
        expired_sessions = [
            session_id for session_id, session in self.sessions.items()
            if session["created_at"] < cutoff_time
        ]
        
        for session_id in expired_sessions:
            del self.sessions[session_id]
        
        return len(expired_sessions)

def require_auth(auth_manager: AuthManager):
    """Decorator for requiring authentication"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # This would be implemented in Flask routes
            # For now, it's a placeholder
            return func(*args, **kwargs)
        return wrapper
    return decorator

def require_permission(auth_manager: AuthManager, permission: str):
    """Decorator for requiring specific permission"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            # This would be implemented in Flask routes
            # For now, it's a placeholder
            return func(*args, **kwargs)
        return wrapper
    return decorator