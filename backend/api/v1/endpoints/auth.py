"""
Authentication endpoints backed by Supabase Auth API.
"""

from fastapi import APIRouter, Depends, HTTPException, status
from supabase_auth.errors import AuthApiError, AuthError

from backend.api.v1.dependencies.auth import get_current_active_user, get_current_user
from backend.api.v1.schemas.auth import (
    LoginRequest,
    LoginResponse,
    LogoutResponse,
    RefreshRequest,
    RefreshResponse,
    RegisterRequest,
    RegisterResponse,
    UserResponse,
)
from backend.services.auth_service import AuthService, get_auth_service

router = APIRouter()


def _normalize_auth_error(
    error: AuthApiError,
    default_status: int,
    default_message: str,
) -> HTTPException:
    """
    Convert Supabase auth API errors into stable FastAPI errors.
    """
    status_code = error.status if getattr(error, "status", None) else default_status
    detail = str(error) or default_message
    if status_code < 400 or status_code >= 600:
        status_code = default_status
    return HTTPException(status_code=status_code, detail=detail)


@router.post("/register", response_model=RegisterResponse, status_code=status.HTTP_201_CREATED)
async def register(
    request: RegisterRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> RegisterResponse:
    """Register a new user via Supabase Auth."""
    try:
        client = await auth_service.create_client()
        auth_response = await client.auth.sign_up(
            {"email": request.email, "password": request.password}
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
    except AuthApiError as e:
        raise _normalize_auth_error(
            e,
            default_status=status.HTTP_400_BAD_REQUEST,
            default_message="User registration failed",
        ) from e
    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e) or "User registration failed",
        ) from e

    user = auth_response.user
    if user is None or user.email is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="User registration failed",
        )

    is_superuser = (user.role or "").lower() in {"service_role", "supabase_admin"}
    message = "User registered successfully"
    if auth_response.session is None:
        message = "User registered. Check your email to confirm your account."

    return RegisterResponse(
        user=UserResponse(
            id=str(user.id),
            email=user.email,
            is_active=True,
            is_superuser=is_superuser,
            created_at=user.created_at,
        ),
        message=message,
    )


@router.post("/login", response_model=LoginResponse)
async def login(
    request: LoginRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> LoginResponse:
    """Authenticate user via Supabase Auth and return session tokens."""
    try:
        client = await auth_service.create_client()
        auth_response = await client.auth.sign_in_with_password(
            {"email": request.email, "password": request.password}
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
    except AuthApiError as e:
        raise _normalize_auth_error(
            e,
            default_status=status.HTTP_401_UNAUTHORIZED,
            default_message="Invalid credentials",
        ) from e
    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e) or "Invalid credentials",
        ) from e

    session = auth_response.session
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Login failed: no active session returned",
        )

    return LoginResponse(
        access_token=session.access_token,
        refresh_token=session.refresh_token,
        token_type=session.token_type or "bearer",
        expires_in=session.expires_in,
    )


@router.post("/refresh", response_model=RefreshResponse)
async def refresh_token(
    request: RefreshRequest,
    auth_service: AuthService = Depends(get_auth_service),
) -> RefreshResponse:
    """Refresh access token via Supabase Auth."""
    try:
        client = await auth_service.create_client()
        auth_response = await client.auth.refresh_session(request.refresh_token)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e),
        ) from e
    except AuthApiError as e:
        raise _normalize_auth_error(
            e,
            default_status=status.HTTP_401_UNAUTHORIZED,
            default_message="Invalid or expired refresh token",
        ) from e
    except AuthError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e) or "Invalid or expired refresh token",
        ) from e

    session = auth_response.session
    if session is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )

    return RefreshResponse(
        access_token=session.access_token,
        token_type=session.token_type or "bearer",
        expires_in=session.expires_in,
    )


@router.post("/logout", response_model=LogoutResponse)
async def logout(
    current_user: UserResponse = Depends(get_current_user),
) -> LogoutResponse:
    """
    Logout current user.

    Supabase logout is effectively handled on the client by discarding tokens.
    """
    _ = current_user
    return LogoutResponse(message="Successfully logged out")


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: UserResponse = Depends(get_current_active_user),
) -> UserResponse:
    """Get current authenticated user information."""
    return current_user
