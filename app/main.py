"""FastAPI application entry point for HH Goa 2026 Voice RAG."""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import router as api_router
from app.config import get_settings
from app.observability.logging import setup_observability_logging
from app.observability.middleware import ObservabilityMiddleware
from app.observability.security import sanitize_error_detail
from app.observability.security_middleware import RateLimitingMiddleware, SecurityHeadersMiddleware
from app.observability.tracing import get_current_request_id

# Configure structured logging
settings = get_settings()
logger = setup_observability_logging("DEBUG" if settings.DEBUG else "INFO")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lifespan context manager for handling startup and shutdown events."""
    logger.info(
        "Starting %s [Environment: %s | Debug: %s | Version: %s]",
        settings.APP_NAME,
        settings.ENVIRONMENT,
        settings.DEBUG,
        settings.VERSION,
    )
    yield
    logger.info("Shutting down %s", settings.APP_NAME)


def create_application() -> FastAPI:
    """Factory function to create and configure the FastAPI application instance."""
    app = FastAPI(
        title=settings.APP_NAME,
        version=settings.VERSION,
        description="Production-ready backend foundation for HH Goa 2026 Voice-Enabled RAG System.",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # 1. Security Headers Middleware (Outermost response headers)
    app.add_middleware(SecurityHeadersMiddleware)

    # 2. Rate Limiting Middleware
    app.add_middleware(RateLimitingMiddleware)

    # 3. Observability & Request Correlation Middleware
    app.add_middleware(ObservabilityMiddleware)

    # 4. CORS Configuration
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.CORS_ORIGINS,
        allow_credentials=True if settings.CORS_ORIGINS != ["*"] else False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Safe Production Exception Handlers
    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        req_id = get_current_request_id() or ""
        logger.warning(
            "Validation error on %s %s: %s",
            request.method,
            request.url.path,
            exc.errors(),
            extra={"request_id": req_id, "event": "validation_error"},
        )
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={
                "error": "Validation Error",
                "detail": sanitize_error_detail(exc.errors()),
                "path": request.url.path,
                "request_id": req_id,
            },
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(
        request: Request, exc: HTTPException
    ) -> JSONResponse:
        req_id = get_current_request_id() or ""
        logger.warning(
            "HTTP exception on %s %s: status=%s detail=%s",
            request.method,
            request.url.path,
            exc.status_code,
            exc.detail,
            extra={"request_id": req_id, "status_code": exc.status_code, "event": "http_exception"},
        )
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": "HTTP Exception",
                "detail": sanitize_error_detail(exc.detail),
                "status_code": exc.status_code,
                "path": request.url.path,
                "request_id": req_id,
            },
            headers=exc.headers,
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        req_id = get_current_request_id() or ""
        logger.error(
            "Unhandled exception on %s %s: %s",
            request.method,
            request.url.path,
            str(exc),
            exc_info=True,
            extra={"request_id": req_id, "event": "internal_error"},
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal Server Error",
                "detail": "An unexpected error occurred. Please contact support with request_id.",
                "path": request.url.path,
                "request_id": req_id,
            },
        )

    # Include API router (both at root and /api prefix for versatility)
    app.include_router(api_router)
    app.include_router(api_router, prefix="/api")

    # Mount Static Frontend UI at /app
    import os
    from fastapi.responses import RedirectResponse
    from starlette.staticfiles import StaticFiles

    @app.get("/", include_in_schema=False)
    async def root_redirect():
        return RedirectResponse(url="/app/")

    frontend_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
    if not os.path.exists(frontend_dir):
        frontend_dir = "frontend"
    if os.path.exists(frontend_dir):
        app.mount("/app", StaticFiles(directory=frontend_dir, html=True), name="frontend")

    return app


app = create_application()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
