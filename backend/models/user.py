"""
User DTO for authentication.

This model is intentionally ORM-free because authentication is delegated
to Supabase Auth API.
"""

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class User(BaseModel):
    """Application-level user response model."""

    id: str = Field(..., description="Supabase user UUID")
    email: str = Field(..., description="User email address")
    is_active: bool = Field(default=True)
    is_superuser: bool = Field(default=False)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = Field(default=None)

    def __repr__(self) -> str:
        return f"<User(id={self.id}, email={self.email})>"
