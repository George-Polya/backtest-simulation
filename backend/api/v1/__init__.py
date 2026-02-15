"""
API v1 package.

Exports the main API router that aggregates all v1 endpoints.
"""

from fastapi import APIRouter

from backend.api.v1.endpoints import auth, backtest

# Create the main v1 router
api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["Authentication"])
api_router.include_router(backtest.router)

__all__ = ["api_router"]
