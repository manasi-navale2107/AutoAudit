from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes.datasets import router as datasets_router
from app.api.routes.runs import router as runs_router
from app.core.config import get_settings
from app.core.logging_config import configure_logging
from app.services.dataset_service import DatasetService
from app.services.run_service import RunService


logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNTIME_DIR = PROJECT_ROOT / "runtime"
DEFAULT_FRONTEND_DIR = PROJECT_ROOT / "frontend"


def create_app(
    *,
    runtime_dir: str | Path = DEFAULT_RUNTIME_DIR,
    frontend_dir: str | Path = DEFAULT_FRONTEND_DIR,
) -> FastAPI:
    """Build the API. Services open only when the server starts."""
    runtime_root = Path(runtime_dir).expanduser().resolve()
    frontend_root = Path(frontend_dir).expanduser().resolve()

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        configure_logging()

        application.state.ready = False
        application.state.dataset_service = None
        application.state.run_service = None

        try:
            # Validate settings without making a paid LLM request.
            # RunService validates the selected provider when a run starts.
            get_settings()

            datasets = DatasetService(
                root_dir=runtime_root / "uploads",
            )

            async with RunService(
                dataset_loader=datasets.load_bytes,
                root_dir=runtime_root,
            ) as runs:
                application.state.dataset_service = datasets
                application.state.run_service = runs
                application.state.ready = True

                logger.info("AutoAudit application started.")

                try:
                    yield
                finally:
                    application.state.ready = False
                    logger.info("AutoAudit application stopping.")

                    # RunService.__aexit__ stops its tasks and training
                    # process, then closes databases and releases its lock.
        finally:
            application.state.ready = False
            application.state.run_service = None
            application.state.dataset_service = None

    application = FastAPI(
        title="AutoAudit API",
        version="0.1.0",
        description=(
            "Dataset upload, planning, human-reviewed preprocessing, "
            "and model training."
        ),
        lifespan=lifespan,
        debug=False,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )

    application.state.ready = False
    application.state.dataset_service = None
    application.state.run_service = None

    application.include_router(
        datasets_router,
        prefix="/api",
    )

    application.include_router(
        runs_router,
        prefix="/api",
    )

    @application.get(
        "/api/health",
        tags=["Health"],
        summary="Check whether the API process responds",
    )
    async def health(response: Response) -> dict[str, str]:
        response.headers["Cache-Control"] = "no-store"

        return {
            "status": "ok",
            "service": "AutoAudit",
        }

    @application.get(
        "/api/ready",
        tags=["Health"],
        summary="Check whether application services have started",
        responses={
            503: {
                "description": "Application services are not ready.",
            }
        },
    )
    async def readiness(
        request: Request,
        response: Response,
    ) -> dict[str, str]:
        state = request.app.state

        ready = (
            bool(getattr(state, "ready", False))
            and getattr(state, "dataset_service", None) is not None
            and getattr(state, "run_service", None) is not None
        )

        response.headers["Cache-Control"] = "no-store"
        response.status_code = 200 if ready else 503

        return {
            "status": "ready" if ready else "not_ready",
            "service": "AutoAudit",
        }

    @application.exception_handler(Exception)
    async def unexpected_error(
        request: Request,
        exc: Exception,
    ) -> JSONResponse:
        logger.error(
            "Unhandled AutoAudit server error.",
            exc_info=(type(exc), exc, exc.__traceback__),
        )

        return JSONResponse(
            status_code=500,
            content={
                "detail": {
                    "code": "internal_error",
                    "message": "An unexpected server error occurred.",
                }
            },
            headers={
                "Cache-Control": "no-store",
            },
        )

    # Serve only public frontend assets at /static/css, /static/js, etc.
    # Restart the server if the frontend directory is created later.
    if frontend_root.is_dir():
        application.mount(
            "/static",
            StaticFiles(
                directory=str(frontend_root),
                follow_symlink=False,
            ),
            name="static",
        )

    @application.get(
        "/",
        include_in_schema=False,
    )
    def home(request: Request) -> Response:
        index_path = frontend_root / "index.html"

        if (
            not index_path.is_symlink()
            and index_path.is_file()
            and index_path.stat().st_size > 0
        ):
            return FileResponse(
                index_path,
                media_type="text/html",
                headers={
                    "Cache-Control": "no-cache",
                },
            )

        # Empty scaffold files should open the API docs
        # instead of displaying a blank UI.
        return RedirectResponse(
            url=str(request.url_for("swagger_ui_html")),
            status_code=307,
            headers={
                "Cache-Control": "no-store",
            },
        )

    return application


app = create_app()