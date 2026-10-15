
"""AutoAudit FastAPI application entry point.

Serves the existing browser workspace, dataset endpoints, the
first-three-agent LangGraph workflow, and the new goal-only
project blueprint router.

The final standalone-project ZIP belongs to the later
Packaging Agent, after all other required agents finish.
"""

from __future__ import annotations

import asyncio
import logging

from contextlib import asynccontextmanager
from pathlib import Path
from uuid import UUID

from fastapi import (
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
)

from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from pydantic import ValidationError

from app.api.routes.project_planning import (
    router as project_planning_router,
)

from app.core.config import get_settings

from app.graph.workflow import ReviewDecision

from app.schemas.run import RunCreateRequest

from app.services.dataset_service import (
    DatasetIntegrityError,
    DatasetService,
)

from app.services.run_service import (
    RunConflictError,
    RunService,
)

from app.services.project_workspace_service import (
    WorkspaceIntegrityError,
)


logger = logging.getLogger(__name__)

settings = get_settings()

MAX_UPLOAD_BYTES = min(
    settings.max_upload_mb * 1024 * 1024,
    25 * 1024 * 1024,
)


# ============================================================
# APPLICATION LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services and close them on application shutdown."""

    settings.create_runtime_directories()

    dataset_service = DatasetService(
        root_dir=settings.uploads_dir
    )

    run_service = RunService(
        dataset_loader=dataset_service.load_bytes,
        root_dir=settings.runtime_dir,
        training_timeout_seconds=(
            settings.training_timeout_seconds
        ),
        max_active_runs=settings.max_concurrent_runs,
        max_pending_runs=max(
            8,
            settings.max_concurrent_runs,
        ),
        max_revisions=settings.max_plan_revisions,
        enable_project_blueprint=False,
    )

    async with run_service:

        app.state.dataset_service = dataset_service
        app.state.run_service = run_service

        try:
            yield

        finally:
            app.state.run_service = None
            app.state.dataset_service = None


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=settings.app_name,
    description=(
        "Multi-agent machine learning project "
        "planning and construction API."
    ),
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url=None,
)


# ============================================================
# NEW: GOAL-ONLY PROJECT PLANNING ROUTER
# ============================================================

app.include_router(
    project_planning_router
)


# ============================================================
# SERVICE ACCESS HELPERS
# ============================================================

def _datasets(request: Request) -> DatasetService:

    service = getattr(
        request.app.state,
        "dataset_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Dataset service unavailable.",
        )

    return service


def _runs(request: Request) -> RunService:

    service = getattr(
        request.app.state,
        "run_service",
        None,
    )

    if service is None:
        raise HTTPException(
            status_code=503,
            detail="Run service unavailable.",
        )

    return service


# ============================================================
# ERROR HANDLING
# ============================================================

def _data_error(exc: Exception) -> HTTPException:

    if isinstance(exc, (FileNotFoundError, KeyError)):
        return HTTPException(
            status_code=404,
            detail="Dataset not found.",
        )

    if isinstance(
        exc,
        (
            DatasetIntegrityError,
            ValueError,
            ValidationError,
        ),
    ):
        return HTTPException(
            status_code=422,
            detail="Dataset is invalid or corrupted.",
        )

    logger.exception(
        "Unexpected dataset storage error"
    )

    return HTTPException(
        status_code=500,
        detail="Dataset service failed.",
    )


def _run_error(exc: Exception) -> HTTPException:

    if isinstance(exc, KeyError):
        return HTTPException(
            status_code=404,
            detail="Run not found.",
        )

    if isinstance(
        exc,
        (
            RunConflictError,
            WorkspaceIntegrityError,
        ),
    ):
        return HTTPException(
            status_code=409,
            detail=str(exc),
        )

    if isinstance(
        exc,
        (
            ValueError,
            ValidationError,
        ),
    ):
        return HTTPException(
            status_code=422,
            detail=str(exc)[:600],
        )

    logger.exception(
        "Unexpected run-service failure"
    )

    return HTTPException(
        status_code=500,
        detail=(
            "Run service failed. "
            "Check server logs."
        ),
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get(
    "/api/health",
    tags=["Health"],
)
async def health() -> dict:

    return {
        "status": "ok",
        "service": "AutoAudit",
    }


@app.get(
    "/api/ready",
    tags=["Health"],
)
async def ready(
    request: Request,
) -> dict:

    _datasets(request)
    _runs(request)

    return {
        "status": "ready",
        "service": "AutoAudit",
    }


# ============================================================
# DATASET UPLOAD
# ============================================================

@app.post(
    "/api/datasets",
    status_code=201,
    tags=["Datasets"],
)
async def upload_dataset(
    request: Request,
    file: UploadFile = File(...),
) -> dict:

    if (
        not file.filename
        or not file.filename.lower().endswith(".csv")
    ):
        raise HTTPException(
            status_code=422,
            detail="Upload a .csv file.",
        )

    content = bytearray()

    try:
        while True:

            chunk = await file.read(
                1024 * 1024
            )

            if not chunk:
                break

            if (
                len(content) + len(chunk)
                > MAX_UPLOAD_BYTES
            ):
                raise HTTPException(
                    status_code=413,
                    detail="CSV exceeds upload limit.",
                )

            content.extend(chunk)

    finally:
        await file.close()

    if not content:
        raise HTTPException(
            status_code=422,
            detail="CSV file is empty.",
        )

    try:
        return await asyncio.to_thread(
            _datasets(request).save_bytes,
            filename=file.filename,
            content=bytes(content),
        )

    except Exception as exc:
        raise _data_error(exc) from None


# ============================================================
# LIST DATASETS
# ============================================================

@app.get(
    "/api/datasets",
    tags=["Datasets"],
)
async def list_datasets(
    request: Request,
    limit: int = Query(
        default=50,
        ge=1,
        le=100,
    ),
) -> list[dict]:

    try:
        return await asyncio.to_thread(
            _datasets(request).list_datasets,
            limit=limit,
        )

    except Exception as exc:
        raise _data_error(exc) from None


# ============================================================
# DATASET METADATA
# ============================================================

@app.get(
    "/api/datasets/{dataset_id}",
    tags=["Datasets"],
)
async def get_dataset(
    request: Request,
    dataset_id: UUID,
) -> dict:

    try:
        return await asyncio.to_thread(
            _datasets(request).get_metadata,
            dataset_id,
        )

    except Exception as exc:
        raise _data_error(exc) from None


# ============================================================
# DATASET PROFILE
# ============================================================

@app.get(
    "/api/datasets/{dataset_id}/profile",
    tags=["Datasets"],
)
async def get_dataset_profile(
    request: Request,
    dataset_id: UUID,
) -> dict:

    try:
        return await asyncio.to_thread(
            _datasets(request).get_profile,
            dataset_id,
        )

    except Exception as exc:
        raise _data_error(exc) from None


# ============================================================
# CREATE DATASET-BACKED ML RUN
# ============================================================

@app.post(
    "/api/runs",
    status_code=202,
    tags=["Runs"],
)
async def create_run(
    request: Request,
    payload: RunCreateRequest,
) -> dict:

    try:
        return await _runs(request).start(
            payload
        )

    except Exception as exc:
        raise _run_error(exc) from None


# ============================================================
# LIST ALL RUNS
# ============================================================

@app.get(
    "/api/runs",
    tags=["Runs"],
)
async def list_runs(
    request: Request,
    limit: int = Query(
        default=50,
        ge=1,
        le=100,
    ),
) -> list[dict]:

    try:
        return await _runs(request).list_runs(
            limit=limit
        )

    except Exception as exc:
        raise _run_error(exc) from None


# ============================================================
# GET ONE RUN
# ============================================================

@app.get(
    "/api/runs/{run_id}",
    tags=["Runs"],
)
async def get_run(
    request: Request,
    run_id: UUID,
) -> dict:

    try:
        return await _runs(request).get(
            run_id
        )

    except Exception as exc:
        raise _run_error(exc) from None


# ============================================================
# HUMAN-IN-THE-LOOP REVIEW
# ============================================================

@app.post(
    "/api/runs/{run_id}/reviews",
    tags=["Runs"],
)
async def review_run(
    request: Request,
    run_id: UUID,
    decision: ReviewDecision,
) -> dict:

    try:
        return await _runs(request).review(
            run_id,
            decision,
        )

    except Exception as exc:
        raise _run_error(exc) from None


# ============================================================
# RETRY FAILED OR INTERRUPTED RUN
# ============================================================

@app.post(
    "/api/runs/{run_id}/retry",
    tags=["Runs"],
)
async def retry_run(
    request: Request,
    run_id: UUID,
) -> dict:

    try:
        return await _runs(request).retry(
            run_id
        )

    except Exception as exc:
        raise _run_error(exc) from None


# ============================================================
# NEW: GENERATED PROJECT PROGRESS
# ============================================================

@app.get(
    "/api/runs/{run_id}/project",
    tags=["Runs"],
)
async def get_project_progress(
    request: Request,
    run_id: UUID,
) -> dict:
    """
    Show the Planner's project structure and the actual
    generation status of each file.

    This endpoint does not create or download a ZIP.
    """

    try:
        return await _runs(
            request
        ).get_project_progress(
            run_id
        )

    except Exception as exc:
        raise _run_error(exc) from None


# ============================================================
# EXISTING AUTOAUDIT FRONTEND
# ============================================================

frontend_dir = Path(
    settings.frontend_dir
)

if frontend_dir.is_dir():

    app.mount(
        "/static",
        StaticFiles(
            directory=frontend_dir
        ),
        name="static",
    )


@app.get(
    "/",
    include_in_schema=False,
)
async def index() -> FileResponse:

    html_path = (
        frontend_dir / "index.html"
    )

    if not html_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="Frontend not installed.",
        )

    return FileResponse(
        html_path,
        media_type="text/html",
    )
