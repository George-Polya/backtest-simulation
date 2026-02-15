"""
Authentication dependencies backed by Supabase Auth API.
"""

from datetime import datetime, timezone
from typing import Any

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from supabase_auth.errors import AuthError

from backend.api.v1.schemas.auth import UserResponse
from backend.services.auth_service import AuthService, get_auth_service

security = HTTPBearer(auto_error=False)


def _map_supabase_user(user: Any) -> UserResponse:
    """Map a Supabase user object to API user response model."""
    email = getattr(user, "email", None)
    if not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User email missing in auth payload",
            headers={"WWW-Authenticate": "Bearer"},
        )

    created_at = getattr(user, "created_at", None) or datetime.now(timezone.utc)
    role = (getattr(user, "role", "") or "").lower()

    return UserResponse(
        id=str(getattr(user, "id")),
        email=email,
        is_active=True,
        is_superuser=role in {"service_role", "supabase_admin"},
        created_at=created_at,
    )


async def get_current_user(
    credentials: HTTPAuthorizationCredentials | None = Depends(security),
    auth_service: AuthService = Depends(get_auth_service),
) -> UserResponse:
    """
    Validate bearer token with Supabase and return current user.
    """
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        client = await auth_service.create_client()
        user_response = await client.auth.get_user(credentials.credentials)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        ) from e

    if user_response is None or user_response.user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return _map_supabase_user(user_response.user)


async def get_current_active_user(
    current_user: UserResponse = Depends(get_current_user),
) -> UserResponse:
    """Ensure current user is active."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user",
        )
    return current_user


async def get_current_superuser(
    current_user: UserResponse = Depends(get_current_active_user),
) -> UserResponse:
    """Ensure current user is a superuser."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient permissions",
        )
    return current_user
