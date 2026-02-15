"""
Authentication service backed by Supabase Auth API.
"""

from supabase import AsyncClient, create_async_client

from backend.core.config import get_settings


class AuthService:
    """Service for creating Supabase Auth clients."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def _resolve_supabase_key(self) -> str:
        """
        Resolve the API key used for Supabase Auth calls.

        Prefers anon key for user auth flows and falls back to service key.
        """
        if self.settings.supabase_anon_key:
            return self.settings.supabase_anon_key
        if self.settings.supabase_service_key:
            return self.settings.supabase_service_key
        raise ValueError(
            "Supabase key is not configured. Set SUPABASE_ANON_KEY "
            "or SUPABASE_SERVICE_KEY in environment variables."
        )

    async def create_client(self) -> AsyncClient:
        """
        Create a Supabase async client for Auth API calls.
        """
        if not self.settings.supabase_url:
            raise ValueError(
                "Supabase URL is not configured. Set SUPABASE_URL in environment variables."
            )

        return await create_async_client(
            self.settings.supabase_url,
            self._resolve_supabase_key(),
        )


def get_auth_service() -> AuthService:
    """Return auth service instance."""
    return AuthService()
