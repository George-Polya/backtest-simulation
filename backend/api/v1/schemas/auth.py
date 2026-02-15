"""
Authentication schemas for request and response models.
"""

from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class LoginRequest(BaseModel):
    """Login request schema."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password")


class TokenPayload(BaseModel):
    """Token payload schema."""

    sub: str = Field(..., description="User ID")
    exp: int = Field(..., description="Expiration timestamp")
    type: str = Field(..., description="Token type (access or refresh)")


class LoginResponse(BaseModel):
    """Login response schema."""

    access_token: str = Field(..., description="JWT access token")
    refresh_token: str = Field(..., description="JWT refresh token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration in seconds")


class RefreshRequest(BaseModel):
    """Refresh token request schema."""

    refresh_token: str = Field(..., description="Refresh token")


class RefreshResponse(BaseModel):
    """Refresh token response schema."""

    access_token: str = Field(..., description="New access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Access token expiration in seconds")


class UserResponse(BaseModel):
    """User response schema."""

    id: str = Field(..., description="User ID (Supabase UUID)")
    email: str = Field(..., description="User email")
    is_active: bool = Field(..., description="Whether user is active")
    is_superuser: bool = Field(..., description="Whether user is superuser")
    created_at: datetime = Field(..., description="User creation timestamp")

    model_config = {"from_attributes": True}


class RegisterRequest(BaseModel):
    """User registration request schema."""

    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, description="User password (min 8 characters)")


class RegisterResponse(BaseModel):
    """User registration response schema."""

    user: UserResponse = Field(..., description="Created user")
    message: str = Field(..., description="Registration message")


class LogoutResponse(BaseModel):
    """Logout response schema."""

    message: str = Field(..., description="Logout message")


class MessageResponse(BaseModel):
    """Generic message response schema."""

    message: str = Field(..., description="Response message")
