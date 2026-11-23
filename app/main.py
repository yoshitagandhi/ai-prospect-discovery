"""FastAPI application: POST /prospect-discovery, GET /health and safe error handling.

Error responses never include stack traces, internal paths, API keys or the rejected input.
Run locally with: uvicorn app.main:app --host 127.0.0.1 --port 4000
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware

from app.config import get_settings
from app.schemas import ErrorResponse, ProspectRequest, ProspectResponse
from app.services import prospect_service

MAX_BODY_BYTES = 10_000
MAX_ERROR_REASONS = 3
METHODS_WITH_BODY = {"POST", "PUT", "PATCH"}
HTTP_ERROR_MESSAGES = {404: "Not found", 405: "Method not allowed"}

logger = logging.getLogger("prospect_discovery")


def error_response(
    status_code: int, message: str, headers: dict[str, str] | None = None
) -> JSONResponse:
    """Build a JSON error body in the ErrorResponse shape."""
    body = ErrorResponse(error=message[:300]).model_dump()
    return JSONResponse(status_code=status_code, content=body, headers=headers)


async def limit_body_size(request: Request, call_next):
    """Reject large bodies before they are read, and bodies of unknown size.

    Without a Content-Length header the size cannot be checked up front, so such
    requests get 411 instead of being read into memory.
    """
    if request.method in METHODS_WITH_BODY:
        content_length = request.headers.get("content-length")
        if content_length is None:
            return error_response(411, "Content-Length header is required")
        if not content_length.isdigit() or int(content_length) > MAX_BODY_BYTES:
            return error_response(413, "Request body too large")
    return await call_next(request)


async def handle_validation_error(
    request: Request, exc: RequestValidationError
) -> JSONResponse:
    """422 with short reasons. Unlike FastAPI's default, the rejected input is never echoed.

    FastAPI error locations start with the source (body, query, path), which is dropped;
    the rest is the field name, e.g. ("body", "query") becomes "query".
    """
    reasons = []
    for error in exc.errors()[:MAX_ERROR_REASONS]:
        field_path = error.get("loc", ())[1:]
        location = ".".join(part for part in field_path if isinstance(part, str))
        message = str(error.get("msg", "invalid value"))
        reasons.append(f"{location} - {message}" if location else message)
    return error_response(422, "Invalid request: " + "; ".join(reasons))


async def handle_http_error(
    request: Request, exc: StarletteHTTPException
) -> JSONResponse:
    """404, 405 and similar errors in the ErrorResponse shape."""
    message = HTTP_ERROR_MESSAGES.get(exc.status_code, "Request failed")
    return error_response(exc.status_code, message, headers=exc.headers)


async def handle_unexpected_error(request: Request, exc: Exception) -> JSONResponse:
    """Log the full error on the server only; the client gets a generic message."""
    logger.exception(
        "Unexpected error while handling %s %s", request.method, request.url.path
    )
    return error_response(500, "Internal server error")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load settings at start-up, so a missing required key stops the app immediately."""
    settings = get_settings()
    logger.setLevel(settings.log_level)
    yield


app = FastAPI(
    title="AI Prospect Discovery API",
    description=(
        "Finds real, public prospects for a natural-language query using permitted "
        "sources only."
    ),
    version="0.1.0",
    lifespan=lifespan,
    middleware=[Middleware(BaseHTTPMiddleware, dispatch=limit_body_size)],
    exception_handlers={
        RequestValidationError: handle_validation_error,
        StarletteHTTPException: handle_http_error,
        Exception: handle_unexpected_error,
    },
)


@app.post(
    "/prospect-discovery",
    response_model=ProspectResponse,
    response_model_exclude_none=True,
    responses={
        411: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        422: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def prospect_discovery(
    body: ProspectRequest, include_evidence: bool = False
) -> ProspectResponse:
    """Discover public prospects for a natural-language query."""
    prospects, warnings = await prospect_service.find_prospects(
        body.query, include_evidence
    )
    return ProspectResponse(
        success=True,
        query=body.query,
        total_prospects=len(prospects),
        prospects=prospects,
        warnings=warnings if include_evidence and warnings else None,
    )


@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness check for the hosting platform. Calls no external API."""
    return {"status": "ok"}
