"""
FastAPI Application Entry Point.

Natural Language Backtesting Service with AI-Generated Code.
"""

import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from backend.core.config import Settings
from backend.core.container import get_container, get_settings_dep

# Configure application logging so that INFO-level messages
# (agent runs, tool calls, LLM calls, etc.) are printed to the console.
# Uvicorn only configures its own loggers; without this, all application
# logger.info() calls are silently dropped by the root logger (WARNING level).


class _ShortNameFormatter(logging.Formatter):
    """Show only the last two segments of the logger name."""

    def format(self, record: logging.LogRecord) -> str:
        parts = record.name.split(".")
        record.name = ".".join(parts[-2:]) if len(parts) > 2 else record.name
        return super().format(record)


_handler = logging.StreamHandler()
_handler.setFormatter(
    _ShortNameFormatter(
        fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
)
logging.root.setLevel(logging.INFO)
logging.root.addHandler(_handler)

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Health check response model."""

    status: str
    app_name: str
    app_version: str
    timestamp: str
    debug: bool
    llm_provider: str
    data_provider: str
    execution_provider: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifespan context manager.

    Handles startup and shutdown events:
    - Startup: Initialize container, pre-load settings, create HTTP client
    - Shutdown: Close HTTP client and release resources
    """
    # Startup
    container = get_container()
    await container.startup()

    yield

    # Shutdown
    await container.shutdown()


def create_app() -> FastAPI:
    """
    Create and configure the FastAPI application.

    Returns:
        Configured FastAPI application instance.
    """
    settings = get_container().settings

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=(
            "AI-powered natural language backtesting service. "
            "Describe your investment strategy in natural language, "
            "and AI will generate Python code to backtest it."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Configure CORS - always allow all origins for open-source local usage
    allow_origins = ["*"]

    app.add_middleware(
        CORSMiddleware,
        allow_origins=allow_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    register_routes(app)

    return app


def register_routes(app: FastAPI) -> None:
    """
    Register all application routes.

    Args:
        app: FastAPI application instance.
    """
    # Import and register API v1 router
    from backend.api.v1 import api_router

    app.include_router(
        api_router,
        prefix="/api/v1",
        tags=["API v1"],
    )

    @app.get(
        "/health",
        response_model=HealthResponse,
        tags=["Health"],
        summary="Health Check",
        description="Check if the service is running and return configuration metadata.",
    )
    async def health_check(
        settings: Settings = Depends(get_settings_dep),
    ) -> HealthResponse:
        """
        Health check endpoint for readiness probes.

        Returns service status and configuration metadata.
        """
        return HealthResponse(
            status="healthy",
            app_name=settings.app_name,
            app_version=settings.app_version,
            timestamp=datetime.now(timezone.utc).isoformat(),
            debug=settings.debug,
            llm_provider=settings.llm.provider.value,
            data_provider=settings.data.provider.value,
            execution_provider=settings.execution.provider.value,
        )

    @app.get(
        "/",
        tags=["Root"],
        summary="Root",
        description="Redirect to API docs.",
        response_class=RedirectResponse,
    )
    async def root() -> RedirectResponse:
        """
        Root endpoint.

        Redirects to the API documentation.
        """
        return RedirectResponse(url="/docs", status_code=302)


# Create the application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
