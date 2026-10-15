
"""Goal-only project planning HTTP endpoints for AutoAudit.

This router does NOT call RunService.start(), train models, produce project
source code, or return a ZIP. It creates and persists the initial Planner
blueprint, with the uploaded dataset explicitly optional.

The router must be registered in app/main.py.

No secrets, raw dataset rows, or internal filesystem paths are returned
to the client.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import sqlite3

from contextlib import contextmanager
from pathlib import Path
from uuid import UUID

from fastapi import APIRouter, HTTPException

from app.agents.planner_agent import PlannerAgent
from app.core.config import get_settings
from app.llm.provider_factory import create_llm_provider
from app.schemas.dataset import DatasetProfile
from app.schemas.project import ProjectBlueprint

from app.schemas.run import (
    ProjectPlanningRequest,
    ProjectPlanningResponse,
    ProjectPlanningStatus,
    ProviderName,
    RunError,
    RunErrorCode,
    RunLLMConfiguration,
)

from app.services.dataset_service import (
    DatasetService,
    DatasetIntegrityError,
)

from app.services.project_workspace_service import (
    ProjectWorkspaceService,
)


# ============================================================
# ROUTER
# ============================================================

router = APIRouter(
    prefix="/api/projects",
    tags=["Project Planning"],
)


# ============================================================
# SQLITE PLANNING STORAGE
# ============================================================

def _database_file() -> Path:
    settings = get_settings()

    directory = (
        Path(settings.runtime_dir)
        / "database"
    )

    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return directory / "project_planning.sqlite3"


@contextmanager
def _connect():
    connection = sqlite3.connect(
        str(_database_file()),
        timeout=15,
    )

    connection.row_factory = sqlite3.Row

    connection.execute(
        "PRAGMA busy_timeout=15000"
    )

    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS project_plans (
            run_id TEXT PRIMARY KEY,
            request_hash TEXT NOT NULL,
            request_json TEXT NOT NULL,
            status TEXT NOT NULL,
            response_json TEXT,
            created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
        )
        """
    )

    try:
        yield connection
        connection.commit()

    except BaseException:
        connection.rollback()
        raise

    finally:
        connection.close()


# ============================================================
# REQUEST FINGERPRINT
# ============================================================

def _request_hash(
    request: ProjectPlanningRequest,
) -> str:

    value = request.model_dump(
        mode="json"
    )

    normalized = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        normalized.encode("utf-8")
    ).hexdigest()


# ============================================================
# CLAIM PLANNING REQUEST
# ============================================================

def _claim(
    request: ProjectPlanningRequest,
) -> tuple[str, dict | None]:

    """
    Reserve the request ID before calling the LLM.

    Identical requests reuse existing results.
    Reusing an ID with different input is rejected.
    """

    run_id = str(request.request_id)

    digest = _request_hash(request)

    serialized_request = (
        request.model_dump_json()
    )

    with _connect() as db:

        db.execute("BEGIN IMMEDIATE")

        row = db.execute(
            """
            SELECT request_hash, status, response_json
            FROM project_plans
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()

        if row is None:

            db.execute(
                """
                INSERT INTO project_plans (
                    run_id,
                    request_hash,
                    request_json,
                    status
                )
                VALUES (?, ?, ?, 'planning')
                """,
                (
                    run_id,
                    digest,
                    serialized_request,
                ),
            )

            return "generate", None

        if row["request_hash"] != digest:
            return "conflict", None

        if row["status"] == "planning":
            return "running", None

        if row["status"] in {
            "planned",
            "awaiting_dataset",
        }:

            if not row["response_json"]:
                raise RuntimeError(
                    "Completed plan response is missing."
                )

            return (
                "cached",
                json.loads(row["response_json"]),
            )

        # A failed request may be retried
        # using the same request ID and input.

        db.execute(
            """
            UPDATE project_plans
            SET status = 'planning',
                response_json = NULL,
                updated_at = CURRENT_TIMESTAMP
            WHERE run_id = ?
            """,
            (run_id,),
        )

        return "generate", None


# ============================================================
# SAVE COMPLETED RESPONSE
# ============================================================

def _finish(
    response: ProjectPlanningResponse,
) -> None:

    data = response.model_dump(
        mode="json"
    )

    with _connect() as db:

        result = db.execute(
            """
            UPDATE project_plans
            SET status = ?,
                response_json = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE run_id = ?
              AND status = 'planning'
            """,
            (
                response.status.value,
                json.dumps(
                    data,
                    ensure_ascii=False,
                    allow_nan=False,
                ),
                str(response.run_id),
            ),
        )

        if result.rowcount != 1:
            raise RuntimeError(
                "Planning record changed during execution."
            )


# ============================================================
# READ SAVED RESPONSE
# ============================================================

def _get_saved(
    run_id: UUID,
) -> dict | None:

    with _connect() as db:

        row = db.execute(
            """
            SELECT
                status,
                request_json,
                response_json
            FROM project_plans
            WHERE run_id = ?
            """,
            (str(run_id),),
        ).fetchone()

    if row is None:
        return None

    return {
        "status": row["status"],

        "request": json.loads(
            row["request_json"]
        ),

        "response": (
            json.loads(row["response_json"])
            if row["response_json"]
            else None
        ),
    }


# ============================================================
# PLANNING RESPONSE HELPERS
# ============================================================

def _pending(
    request: ProjectPlanningRequest,
) -> ProjectPlanningResponse:

    settings = get_settings()

    return ProjectPlanningResponse(
        run_id=request.request_id,
        objective=request.objective,
        dataset_id=request.dataset_id,
        status=ProjectPlanningStatus.PLANNING,
        language=(
            request.language
            or settings.default_language
        ),
        blueprint=None,
        clarification_questions=[],
        error=None,
        final_zip_ready=False,
    )


def _failure(
    request: ProjectPlanningRequest,
    code: RunErrorCode,
) -> ProjectPlanningResponse:

    messages = {
        RunErrorCode.DATASET_UNAVAILABLE:
            "The selected dataset is unavailable or invalid.",

        RunErrorCode.PROVIDER_UNAVAILABLE:
            "The configured LLM provider is unavailable.",

        RunErrorCode.PROVIDER_RATE_LIMITED:
            "The LLM provider is rate limited.",

        RunErrorCode.MODEL_OUTPUT_INVALID:
            "The Planner did not return a valid blueprint.",

        RunErrorCode.INVALID_PLAN:
            "The project plan did not pass validation.",

        RunErrorCode.STORAGE_FAILED:
            "The project plan could not be stored.",

        RunErrorCode.INTERNAL_ERROR:
            "Project planning failed. Check server logs.",
    }

    return ProjectPlanningResponse(
        run_id=request.request_id,
        objective=request.objective,
        dataset_id=request.dataset_id,
        status=ProjectPlanningStatus.FAILED,
        language=(
            request.language
            or get_settings().default_language
        ),
        blueprint=None,
        clarification_questions=[],
        error=RunError(
            code=code,
            message=messages.get(
                code,
                messages[RunErrorCode.INTERNAL_ERROR],
            ),
            retryable=code not in {
                RunErrorCode.INVALID_PLAN,
                RunErrorCode.DATASET_UNAVAILABLE,
            },
        ),
        final_zip_ready=False,
    )


# ============================================================
# LLM PROVIDER CONFIGURATION
# ============================================================

def _provider_configuration(
    request: ProjectPlanningRequest,
) -> RunLLMConfiguration:

    settings = get_settings()

    provider = ProviderName(
        request.provider
        or settings.llm_provider
    )

    configured_model = settings.require_provider(
        provider.value
    )

    if (
        request.model is not None
        and request.model != configured_model
    ):
        raise ValueError(
            "The requested model is not the configured provider model."
        )

    is_ollama = (
        provider == ProviderName.OLLAMA
    )

    return RunLLMConfiguration(
        provider=provider,
        model=configured_model,
        temperature=settings.llm_temperature,
        max_output_tokens=settings.llm_max_output_tokens,
        timeout_seconds=(
            settings.ollama_timeout_seconds
            if is_ollama
            else settings.llm_timeout_seconds
        ),
        max_retries=settings.llm_max_retries,
        ollama_num_ctx=(
            settings.ollama_num_ctx
            if is_ollama
            else None
        ),
    )


# ============================================================
# LLM CONNECTION CLEANUP
# ============================================================

async def _close_provider(
    provider: object,
) -> None:

    closer = getattr(
        provider,
        "aclose",
        None,
    )

    if closer is None:
        closer = getattr(
            provider,
            "close",
            None,
        )

    if callable(closer):

        result = closer()

        if inspect.isawaitable(result):
            await result


# ============================================================
# VERIFY PROJECT WORKSPACE
# ============================================================

def _verify_saved_project(
    response: ProjectPlanningResponse,
) -> None:

    if response.blueprint is None:
        return

    workspace = ProjectWorkspaceService(
        root_dir=get_settings().runs_dir
    )

    manifest = workspace.get_manifest(
        run_id=response.run_id
    )

    persisted = ProjectBlueprint.model_validate(
        manifest["blueprint"]
    )

    if persisted != response.blueprint:
        raise RuntimeError(
            "Persisted workspace blueprint differs from saved plan."
        )


# ============================================================
# POST — CREATE PROJECT PLAN
# ============================================================

@router.post(
    "/plan",
    response_model=ProjectPlanningResponse,
)
async def plan_project(
    request: ProjectPlanningRequest,
) -> ProjectPlanningResponse:

    """
    Generate an end-to-end project blueprint.

    Dataset is optional at planning time.
    No training or ZIP generation happens here.
    """

    try:
        mode, cached = await asyncio.to_thread(
            _claim,
            request,
        )

    except (sqlite3.Error, OSError):
        raise HTTPException(
            status_code=503,
            detail="Planning storage unavailable.",
        ) from None

    if mode == "conflict":
        raise HTTPException(
            status_code=409,
            detail="Request ID belongs to different input.",
        )

    if mode == "running":
        return _pending(request)

    if mode == "cached":

        response = ProjectPlanningResponse.model_validate(
            cached
        )

        try:
            await asyncio.to_thread(
                _verify_saved_project,
                response,
            )

        except Exception:
            raise HTTPException(
                status_code=409,
                detail="Stored project plan requires inspection.",
            ) from None

        return response

    provider = None

    try:

        # ------------------------------------------
        # 1. Resolve the LLM provider
        # ------------------------------------------

        config = _provider_configuration(
            request
        )

        # ------------------------------------------
        # 2. Load optional dataset profile
        # ------------------------------------------

        profile = None

        if request.dataset_id is not None:

            dataset = DatasetService()

            raw_profile = await asyncio.to_thread(
                dataset.get_profile,
                request.dataset_id,
            )

            profile = DatasetProfile.model_validate(
                raw_profile
            )

        # ------------------------------------------
        # 3. Call Planner Agent
        # ------------------------------------------

        provider = create_llm_provider(
            configuration=config
        )

        planner = PlannerAgent(
            provider=provider
        )

        blueprint = await planner.plan_project(
            goal=request.objective,
            profile=profile,
            requested_target=request.target_column,
            user_answers=(),
            language=(
                request.language
                or get_settings().default_language
            ),
        )

        # ------------------------------------------
        # 4. Initialize project workspace
        # ------------------------------------------

        workspace = ProjectWorkspaceService(
            root_dir=get_settings().runs_dir
        )

        await asyncio.to_thread(
            workspace.initialize,
            run_id=request.request_id,
            blueprint=blueprint,
        )

        # ------------------------------------------
        # 5. Build response
        # ------------------------------------------

        status = (
            ProjectPlanningStatus.PLANNED
            if profile is not None
            else ProjectPlanningStatus.AWAITING_DATASET
        )

        response = ProjectPlanningResponse(
            run_id=request.request_id,
            objective=request.objective,
            dataset_id=request.dataset_id,
            status=status,
            language=(
                request.language
                or get_settings().default_language
            ),
            blueprint=blueprint,
            clarification_questions=[],
            error=None,
            final_zip_ready=False,
        )

        # ------------------------------------------
        # 6. Persist result
        # ------------------------------------------

        await asyncio.to_thread(
            _finish,
            response,
        )

        return response

    except asyncio.CancelledError:

        await asyncio.to_thread(
            _finish,
            _failure(
                request,
                RunErrorCode.INTERNAL_ERROR,
            ),
        )

        raise

    except Exception as exc:

        code_string = getattr(
            exc,
            "code",
            None,
        )

        if code_string in {
            "provider_rate_limited",
            "provider_unavailable",
            "model_output_invalid",
        }:

            code = RunErrorCode(
                code_string
            )

        elif (
            isinstance(
                exc,
                (
                    DatasetIntegrityError,
                    FileNotFoundError,
                    KeyError,
                ),
            )
            and request.dataset_id is not None
        ):

            code = RunErrorCode.DATASET_UNAVAILABLE

        elif isinstance(
            exc,
            (ValueError, TypeError),
        ):

            code = RunErrorCode.INVALID_PLAN

        else:
            code = RunErrorCode.INTERNAL_ERROR

        response = _failure(
            request,
            code,
        )

        await asyncio.to_thread(
            _finish,
            response,
        )

        return response

    finally:

        if provider is not None:

            try:
                await _close_provider(
                    provider
                )

            except Exception:
                # Cleanup must not overwrite a stored result.
                pass


# ============================================================
# GET — FETCH EXISTING PROJECT PLAN
# ============================================================

@router.get(
    "/plan/{run_id}",
    response_model=ProjectPlanningResponse,
)
async def get_project_plan(
    run_id: UUID,
) -> ProjectPlanningResponse:

    """
    Fetch persistent Planner output.

    Returns blueprint information only.
    Final ZIP creation is not part of this endpoint.
    """

    try:
        saved = await asyncio.to_thread(
            _get_saved,
            run_id,
        )

    except (OSError, sqlite3.Error):

        raise HTTPException(
            status_code=503,
            detail="Planning storage unavailable.",
        ) from None

    if saved is None:

        raise HTTPException(
            status_code=404,
            detail="Project plan not found.",
        )

    if saved["response"] is None:

        request = ProjectPlanningRequest.model_validate(
            saved["request"]
        )

        return _pending(
            request
        )

    response = ProjectPlanningResponse.model_validate(
        saved["response"]
    )

    if response.blueprint is not None:

        try:
            await asyncio.to_thread(
                _verify_saved_project,
                response,
            )

        except Exception:

            raise HTTPException(
                status_code=409,
                detail="Stored project plan requires inspection.",
            ) from None

    return response
