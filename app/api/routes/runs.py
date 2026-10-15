from __future__ import annotations

import logging
import sqlite3
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Annotated, Any, TypeVar
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Query,
    Request,
    Response,
    status,
)
from fastapi.responses import FileResponse
from pydantic import (
    AwareDatetime,
    BaseModel,
    ConfigDict,
    Field,
)

from app.graph.workflow import ReviewDecision
from app.schemas.run import RunCreateRequest
from app.services.run_service import (
    RunConflictError,
    RunService,
)


logger = logging.getLogger(__name__)

ResultT = TypeVar("ResultT")


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    detail: ErrorDetail


class RunSummary(BaseModel):
    model_config = ConfigDict(extra="ignore")

    run_id: UUID
    status: str
    updated_at: AwareDatetime


class RunInterrupt(BaseModel):
    id: str
    value: dict[str, Any]


class RunResponse(RunSummary):
    error: str | None = None
    state: dict[str, Any]

    interrupts: list[RunInterrupt] = Field(
        default_factory=list
    )


router = APIRouter(
    prefix="/runs",
    tags=["Runs"],
    responses={
        400: {
            "model": ErrorResponse,
            "description": "Run input was rejected.",
        },
        404: {
            "model": ErrorResponse,
            "description": "Run or dataset was not found.",
        },
        409: {
            "model": ErrorResponse,
            "description": "Action conflicts with run state.",
        },
        500: {
            "model": ErrorResponse,
            "description": "Unexpected server error.",
        },
        503: {
            "model": ErrorResponse,
            "description": "Run service is unavailable.",
        },
    },
)


def _http_error(
    status_code: int,
    code: str,
    message: str,
) -> HTTPException:
    return HTTPException(
        status_code=status_code,
        detail={
            "code": code,
            "message": message,
        },
        headers={
            "Cache-Control": "no-store",
        },
    )


async def get_run_service(
    request: Request,
    response: Response,
) -> RunService:
    """The application lifespan owns this already-opened service."""
    response.headers["Cache-Control"] = "no-store"

    service = getattr(
        request.app.state,
        "run_service",
        None,
    )

    if service is None:
        raise _http_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "run_service_unavailable",
            "Run service is not ready. Try again shortly.",
        )

    return service


ServiceDependency = Annotated[
    RunService,
    Depends(get_run_service),
]


async def _invoke(
    operation: Callable[..., Awaitable[ResultT]],
    *args: Any,
    **kwargs: Any,
) -> ResultT:
    """Translate service errors without exposing internal exception details."""
    try:
        return await operation(
            *args,
            **kwargs,
        )

    except HTTPException:
        raise

    except RunConflictError as exc:
        raise _http_error(
            status.HTTP_409_CONFLICT,
            "run_conflict",
            str(exc),
        ) from None

    except (KeyError, FileNotFoundError):
        raise _http_error(
            status.HTTP_404_NOT_FOUND,
            "resource_not_found",
            "The requested run or dataset was not found.",
        ) from None

    except ValueError as exc:
        logger.warning(
            "Run input rejected; operation=%s; error_type=%s",
            operation.__name__,
            type(exc).__name__,
        )

        raise _http_error(
            status.HTTP_400_BAD_REQUEST,
            "invalid_run_input",
            "Check the selected dataset, provider, model, "
            "and review feedback.",
        ) from None

    except (RuntimeError, OSError, sqlite3.Error):
        logger.exception(
            "Run service operation unavailable: %s",
            operation.__name__,
        )

        raise _http_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "run_service_unavailable",
            "The run service could not complete this request. "
            "Try again shortly.",
        ) from None

    except Exception:
        logger.exception(
            "Unexpected run API error: %s",
            operation.__name__,
        )

        raise _http_error(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            "internal_error",
            "An unexpected server error occurred.",
        ) from None


def _execution_response(
    result: dict[str, Any],
    request: Request,
    response: Response,
) -> dict[str, Any]:
    """A repeated request can return an existing, already-settled run."""
    response.status_code = (
        status.HTTP_202_ACCEPTED
        if result["status"] in {"queued", "running"}
        else status.HTTP_200_OK
    )

    response.headers["Location"] = str(
        request.url_for(
            "autoaudit_get_run",
            run_id=str(result["run_id"]),
        )
    )

    return result


@router.post(
    "",
    name="autoaudit_create_run",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        200: {
            "model": RunResponse,
        },
    },
    summary="Start a run or retrieve an identical existing request",
)
async def create_run(
    payload: RunCreateRequest,
    request: Request,
    response: Response,
    service: ServiceDependency,
) -> dict[str, Any]:
    result = await _invoke(
        service.start,
        payload,
    )

    return _execution_response(
        result,
        request,
        response,
    )


@router.get(
    "",
    name="autoaudit_list_runs",
    response_model=list[RunSummary],
    summary="List recent runs",
)
async def list_runs(
    service: ServiceDependency,
    limit: Annotated[
        int,
        Query(ge=1, le=100),
    ] = 50,
) -> list[dict[str, Any]]:
    return await _invoke(
        service.list_runs,
        limit=limit,
    )


@router.get(
    "/{run_id}",
    name="autoaudit_get_run",
    response_model=RunResponse,
    summary="Get run progress, pending questions, and available results",
)
async def get_run(
    run_id: UUID,
    service: ServiceDependency,
) -> dict[str, Any]:
    return await _invoke(
        service.get,
        run_id,
    )


@router.post(
    "/{run_id}/reviews",
    name="autoaudit_review_run",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        200: {
            "model": RunResponse,
        },
    },
    summary="Approve, request changes, reject, or answer Planner questions",
)
async def review_run(
    run_id: UUID,
    payload: ReviewDecision,
    request: Request,
    response: Response,
    service: ServiceDependency,
) -> dict[str, Any]:
    result = await _invoke(
        service.review,
        run_id,
        payload,
    )

    return _execution_response(
        result,
        request,
        response,
    )


@router.post(
    "/{run_id}/retry",
    name="autoaudit_retry_run",
    response_model=RunResponse,
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        200: {
            "model": RunResponse,
        },
    },
    summary="Continue a failed or interrupted run from its checkpoint",
)
async def retry_run(
    run_id: UUID,
    request: Request,
    response: Response,
    service: ServiceDependency,
) -> dict[str, Any]:
    result = await _invoke(
        service.retry,
        run_id,
    )

    return _execution_response(
        result,
        request,
        response,
    )


@router.post(
    "/{run_id}/handoff",
    name="autoaudit_create_handoff",
    summary="Create and download the pre-audit handoff ZIP",
    response_class=FileResponse,
    responses={
        200: {
            "description": (
                "Handoff ZIP for a trained run. The leakage audit "
                "has not been performed by this endpoint."
            ),
            "content": {
                "application/zip": {},
            },
        },
        400: {
            "model": ErrorResponse,
            "description": "Handoff could not be created from the current run.",
        },
        404: {
            "model": ErrorResponse,
            "description": "The run or its artifact was not found.",
        },
        409: {
            "model": ErrorResponse,
            "description": "The run has not completed training.",
        },
        503: {
            "model": ErrorResponse,
            "description": "The handoff archive is unavailable.",
        },
    },
)
async def create_handoff(
    run_id: UUID,
    service: ServiceDependency,
) -> FileResponse:
    """Create a ZIP from verified training artifacts and stream it to the client."""
    result = await _invoke(
        service.create_handoff,
        run_id,
    )

    archive_value = result.get("handoff_zip")
    if not isinstance(archive_value, str) or not archive_value.strip():
        logger.error(
            "Handoff service returned no archive path; run_id=%s",
            run_id,
        )
        raise _http_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "handoff_unavailable",
            "The handoff archive is not available. Try again shortly.",
        )

    archive_path = Path(archive_value)
    if archive_path.suffix.lower() != ".zip" or not archive_path.is_file():
        logger.error(
            "Handoff archive is missing or invalid; run_id=%s",
            run_id,
        )
        raise _http_error(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "handoff_unavailable",
            "The handoff archive is not available. Try again shortly.",
        )

    return FileResponse(
        path=archive_path,
        media_type="application/zip",
        filename=f"autoaudit-handoff-{run_id}.zip",
        headers={
            "Cache-Control": "no-store",
            "X-AutoAudit-Status": "ready_for_audit",
            "X-AutoAudit-Audit-Status": "not_performed",
        },
    )
