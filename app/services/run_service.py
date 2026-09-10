from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import multiprocessing
import os
import tempfile
from collections.abc import Callable
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path
from uuid import UUID

import aiosqlite
from filelock import FileLock
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from app.agents.data_engineering_agent import DataEngineeringAgent
from app.agents.model_training_agent import (
    ModelTrainingAgent,
    TrainingDecision,
    execution_plan_sha256,
)
from app.agents.planner_agent import PlannerAgent
from app.core.config import get_settings
from app.graph.workflow import (
    ReviewDecision,
    build_workflow,
    create_initial_state,
)
from app.llm.provider_factory import create_llm_provider
from app.ml.model_factory import TrainingRuntimeSettings
from app.ml.training import TrainingExecutor
from app.schemas.cleaning import ExecutionPlan
from app.schemas.run import RunCreateRequest, RunLLMConfiguration
from app.services.artifact_service import ArtifactService
from app.services.cleaning_review_store import CleaningReviewStore


logger = logging.getLogger(__name__)

DEFAULT_ROOT = (
    Path(__file__).resolve().parents[2]
    / "runtime"
)


class RunConflictError(ValueError):
    """An action conflicts with the current run."""


def _json(value) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        allow_nan=False,
    )


def _write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")

    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(_json(value))
        stream.flush()
        os.fsync(stream.fileno())

    os.replace(temporary, path)


def _training_worker(job_dir: str) -> None:
    """Top-level function required by Windows multiprocessing spawn."""
    directory = Path(job_dir)

    job = json.loads(
        (directory / "job.json").read_text("utf-8")
    )

    reviews = CleaningReviewStore(
        database_path=job["review_path"]
    )

    arguments = job["arguments"]

    plan = ExecutionPlan.model_validate(
        arguments["execution_plan"]
    )

    result = TrainingExecutor(
        review_store=reviews
    ).fit(
        run_id=UUID(arguments["run_id"]),
        revision=arguments["revision"],
        dataset_id=UUID(arguments["dataset_id"]),
        dataset_bytes=(
            directory / "dataset.csv"
        ).read_bytes(),
        execution_plan=plan,
        decision=TrainingDecision.model_validate(
            arguments["decision"]
        ),
        runtime=TrainingRuntimeSettings.model_validate(
            arguments["runtime"]
        ),
    )

    artifacts = ArtifactService(
        review_store=reviews,
        root_dir=job["artifact_root"],
    )

    manifest = artifacts.save(
        result=result,
        execution_plan=plan,
    )

    _write_json(
        Path(job["receipt_path"]),
        {
            "job_sha256": job["job_sha256"],
            "report": result.report,
            "manifest": manifest,
        },
    )


class _RunProvider:
    """Inspecting saved state does not open an LLM connection."""

    def __init__(self, configuration: dict) -> None:
        self.configuration = RunLLMConfiguration.model_validate(
            configuration
        )

        self.provider = None

    async def generate_structured(self, **kwargs):
        if self.provider is None:
            self.provider = create_llm_provider(
                configuration=self.configuration
            )

        return await self.provider.generate_structured(
            **kwargs
        )

    async def close(self) -> None:
        if self.provider is None:
            return

        closer = getattr(
            self.provider,
            "aclose",
            None,
        )

        if closer is None:
            closer = getattr(
                self.provider,
                "close",
                None,
            )

        if callable(closer):
            result = closer()

            if inspect.isawaitable(result):
                await result


class RunService:
    """
    Single-server coordinator with persistent checkpoints.

    Open one instance using:
        async with RunService(dataset_loader=...)

    dataset_loader(UUID) returns the original stored CSV bytes.

    Public methods:
        start
        get
        list_runs
        review
        retry
        close
    """

    def __init__(
        self,
        *,
        dataset_loader: Callable[[UUID], bytes],
        root_dir: str | Path = DEFAULT_ROOT,
        runtime: TrainingRuntimeSettings | None = None,
        training_timeout_seconds: float = 600,
        max_active_runs: int = 2,
        max_pending_runs: int = 8,
        max_revisions: int = 5,
    ) -> None:
        if not callable(dataset_loader):
            raise TypeError(
                "dataset_loader must be callable."
            )

        if not 10 <= training_timeout_seconds <= 3600:
            raise ValueError(
                "Training timeout must be 10..3600 seconds."
            )

        if any(
            type(value) is not int
            for value in (
                max_active_runs,
                max_pending_runs,
                max_revisions,
            )
        ):
            raise TypeError(
                "Run and revision limits must be integers."
            )

        if not 1 <= max_active_runs <= max_pending_runs <= 32:
            raise ValueError(
                "Require 1 <= active <= pending <= 32."
            )

        if not 1 <= max_revisions <= 10:
            raise ValueError(
                "max_revisions must be 1..10."
            )

        self.root = Path(root_dir).resolve()

        self.database_dir = self.root / "database"

        self.review_path = (
            self.database_dir
            / "cleaning_reviews.sqlite3"
        )

        self.receipts = (
            self.root / "training_receipts"
        )

        self.dataset_loader = dataset_loader

        self.runtime = TrainingRuntimeSettings.model_validate(
            (
                runtime or TrainingRuntimeSettings()
            ).model_dump()
        )

        self.training_timeout = training_timeout_seconds
        self.max_pending_runs = max_pending_runs
        self.max_revisions = max_revisions

        self._actions = asyncio.Lock()

        self._capacity = asyncio.Semaphore(
            max_active_runs
        )

        self._training_slot = asyncio.Semaphore(1)

        self._tasks: dict[str, asyncio.Task] = {}

        self._stack = AsyncExitStack()

        self._opened = False
        self._closed = False

    async def __aenter__(self):
        if self._opened or self._closed:
            raise RuntimeError(
                "Use a new RunService instance."
            )

        self.database_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.receipts.mkdir(
            parents=True,
            exist_ok=True,
        )

        lock = FileLock(
            str(
                self.database_dir
                / "run_service.lock"
            )
        )

        lock.acquire(timeout=0)

        self._stack.callback(lock.release)

        try:
            self.reviews = CleaningReviewStore(
                database_path=self.review_path
            )

            self.artifacts = ArtifactService(
                review_store=self.reviews,
                root_dir=self.root / "runs",
            )

            self._db = await self._stack.enter_async_context(
                aiosqlite.connect(
                    self.database_dir / "runs.sqlite3"
                )
            )

            await self._db.execute(
                "PRAGMA journal_mode=WAL"
            )

            await self._db.execute(
                "CREATE TABLE IF NOT EXISTS runs "
                "(run_id TEXT PRIMARY KEY, payload TEXT NOT NULL)"
            )

            self._saver = await self._stack.enter_async_context(
                AsyncSqliteSaver.from_conn_string(
                    str(
                        self.database_dir
                        / "workflow.sqlite3"
                    )
                )
            )

            await self._saver.setup()

            async with self._db.execute(
                "SELECT payload FROM runs"
            ) as cursor:
                rows = await cursor.fetchall()

            for (payload,) in rows:
                data = json.loads(payload)

                if data["status"] in {"queued", "running"}:
                    data["status"] = "interrupted"
                    await self._save(data)

            await self._db.commit()

            self._opened = True

            return self

        except BaseException:
            await self._stack.aclose()
            self._closed = True
            raise

    async def __aexit__(self, *_):
        await self.close()

    def _check_open(self) -> None:
        if not self._opened or self._closed:
            raise RuntimeError(
                "RunService must be opened with async with."
            )

    @staticmethod
    def _config(run_id: str) -> dict:
        return {
            "configurable": {
                "thread_id": run_id,
            },
            "recursion_limit": 100,
        }

    async def _load(self, run_id: str) -> dict:
        async with self._db.execute(
            "SELECT payload FROM runs WHERE run_id=?",
            (run_id,),
        ) as cursor:
            row = await cursor.fetchone()

        if row is None:
            raise KeyError("Run not found.")

        return json.loads(row[0])

    async def _save(self, data: dict) -> None:
        data["updated_at"] = datetime.now(
            timezone.utc
        ).isoformat()

        await self._db.execute(
            "INSERT INTO runs VALUES (?, ?) "
            "ON CONFLICT(run_id) "
            "DO UPDATE SET payload=excluded.payload",
            (
                data["run_id"],
                _json(data),
            ),
        )

        await self._db.commit()

    def _graph(self, data: dict):
        provider = _RunProvider(
            data["llm"]
        )

        graph = build_workflow(
            planner=PlannerAgent(
                provider=provider
            ),
            engineer=DataEngineeringAgent(
                provider=provider
            ),
            training_agent=ModelTrainingAgent(
                provider=provider,
                review_store=self.reviews,
            ),
            review_store=self.reviews,
            dataset_loader=self.dataset_loader,
            training_runner=self._train,
            checkpointer=self._saver,
            max_revisions=data["max_revisions"],
        )

        return graph, provider

    @staticmethod
    def _interrupts(snapshot) -> list:
        return [
            item
            for task in snapshot.tasks
            for item in task.interrupts
        ]

    def _check_capacity(self) -> None:
        if len(self._tasks) >= self.max_pending_runs:
            raise RunConflictError(
                "Run queue is full. Try again shortly."
            )

    def _schedule(self, data: dict) -> None:
        run_id = data["run_id"]

        self._tasks[run_id] = asyncio.create_task(
            self._drive(data),
            name=f"run:{run_id}",
        )

    async def start(
        self,
        request: RunCreateRequest,
    ) -> dict:
        request = RunCreateRequest.model_validate_json(
            request.model_dump_json()
        )

        run_id = str(request.request_id)

        source = request.model_dump(
            mode="json"
        )

        async with self._actions:
            self._check_open()

            try:
                existing = await self._load(run_id)
            except KeyError:
                existing = None

            if existing is not None:
                if existing["original_request"] != source:
                    raise RunConflictError(
                        "Request ID was reused with different content."
                    )

            else:
                self._check_capacity()

                settings = get_settings()

                name = (
                    request.provider
                    or settings.llm_provider
                )

                model = settings.require_provider(name)

                if request.model not in (None, model):
                    raise ValueError(
                        "Choose the configured model for this provider."
                    )

                resolved = RunCreateRequest.model_validate(
                    {
                        **source,
                        "provider": name,
                        "model": model,
                        "language": (
                            request.language
                            or settings.default_language
                        ),
                    }
                )

                llm = RunLLMConfiguration(
                    provider=name,
                    model=model,
                    temperature=0.0,
                    max_output_tokens=4096,
                    timeout_seconds=(
                        300.0 if name == "ollama" else 120.0
                    ),
                    max_retries=2,
                    ollama_num_ctx=(
                        16384 if name == "ollama" else None
                    ),
                )

                raw = await asyncio.to_thread(
                    self.dataset_loader,
                    request.dataset_id,
                )

                initial = await asyncio.to_thread(
                    create_initial_state,
                    run_id=UUID(run_id),
                    request=resolved,
                    dataset_bytes=raw,
                    runtime=self.runtime,
                )

                data = {
                    "run_id": run_id,
                    "original_request": source,
                    "initial_state": initial,
                    "llm": llm.model_dump(mode="json"),
                    "max_revisions": self.max_revisions,
                    "status": "queued",
                    "error": None,
                    "pending_review": None,
                    "last_review": None,
                }

                await self._save(data)

                self._schedule(data)

        return await self.get(run_id)

    async def get(
        self,
        run_id: UUID | str,
    ) -> dict:
        run_id = str(UUID(str(run_id)))

        async with self._actions:
            self._check_open()

            data = await self._load(run_id)

            graph, _ = self._graph(data)

            snapshot = await graph.aget_state(
                self._config(run_id)
            )

            return {
                "run_id": run_id,
                "status": data["status"],
                "error": data["error"],
                "updated_at": data["updated_at"],
                "state": (
                    snapshot.values
                    or data["initial_state"]
                ),
                "interrupts": [
                    {
                        "id": item.id,
                        "value": item.value,
                    }
                    for item in self._interrupts(snapshot)
                ],
            }

    async def list_runs(
        self,
        limit: int = 50,
    ) -> list[dict]:
        if not 1 <= limit <= 100:
            raise ValueError(
                "limit must be 1..100."
            )

        async with self._actions:
            self._check_open()

            async with self._db.execute(
                "SELECT payload FROM runs "
                "ORDER BY json_extract(payload, '$.updated_at') "
                "DESC LIMIT ?",
                (limit,),
            ) as cursor:
                rows = await cursor.fetchall()

        result = []

        for (payload,) in rows:
            data = json.loads(payload)

            result.append(
                {
                    key: data[key]
                    for key in (
                        "run_id",
                        "status",
                        "updated_at",
                    )
                }
            )

        return result

    async def review(
        self,
        run_id: UUID | str,
        decision: ReviewDecision | dict,
    ) -> dict:
        run_id = str(UUID(str(run_id)))

        command = ReviewDecision.model_validate(
            decision
        ).model_dump(mode="json")

        async with self._actions:
            self._check_open()

            data = await self._load(run_id)

            if (
                data["last_review"]
                and data["last_review"]["decision"] == command
            ):
                # Identical repeated submissions are idempotent.
                pass

            else:
                if run_id in self._tasks:
                    raise RunConflictError(
                        "This run is already executing."
                    )

                graph, _ = self._graph(data)

                snapshot = await graph.aget_state(
                    self._config(run_id)
                )

                interrupts = self._interrupts(
                    snapshot
                )

                if len(interrupts) != 1:
                    raise RunConflictError(
                        "This run is not waiting for a decision."
                    )

                current = interrupts[0]
                payload = current.value

                allowed = (
                    {"answer", "reject"}
                    if payload["kind"] == "planner_clarification"
                    else {"approve", "changes", "reject"}
                )

                if (
                    command["revision"] != payload["revision"]
                    or command["action"] not in allowed
                ):
                    raise RunConflictError(
                        "Respond to the current question or revision."
                    )

                if (
                    command["action"] == "approve"
                    and payload["proposal"].get(
                        "unresolved_questions"
                    )
                ):
                    raise ValueError(
                        "Resolve proposal questions before approval."
                    )

                if command["action"] == "answer":
                    state = snapshot.values

                    answered = json.dumps(
                        {
                            "questions": payload["questions"],
                            "answer": command["feedback"],
                        },
                        ensure_ascii=False,
                    )

                    request = dict(
                        state["request"]
                    )

                    request["objective"] += (
                        "\nUser clarifications:\n"
                        + "\n".join(
                            state["answers"] + [answered]
                        )
                    )

                    RunCreateRequest.model_validate(
                        request
                    )

                self._check_capacity()

                data["pending_review"] = {
                    "id": current.id,
                    "decision": command,
                }

                data["last_review"] = data["pending_review"]

                data.update(
                    status="queued",
                    error=None,
                )

                await self._save(data)

                self._schedule(data)

        return await self.get(run_id)

    async def retry(
        self,
        run_id: UUID | str,
    ) -> dict:
        run_id = str(UUID(str(run_id)))

        async with self._actions:
            self._check_open()

            data = await self._load(run_id)

            if (
                run_id in self._tasks
                or data["status"] not in {
                    "failed",
                    "interrupted",
                }
            ):
                raise RunConflictError(
                    "Only failed or interrupted runs can retry."
                )

            self._check_capacity()

            data.update(
                status="queued",
                error=None,
            )

            await self._save(data)

            self._schedule(data)

        return await self.get(run_id)

    async def _drive(self, data: dict) -> None:
        run_id = data["run_id"]

        provider = None

        status = "interrupted"
        error = None
        clear_review = False

        try:
            async with self._capacity:
                async with self._actions:
                    data["status"] = "running"

                    await self._save(data)

                graph, provider = self._graph(data)

                config = self._config(run_id)

                snapshot = await graph.aget_state(
                    config
                )

                interrupts = self._interrupts(
                    snapshot
                )

                pending = data["pending_review"]

                invoke = True

                if not snapshot.values:
                    value = data["initial_state"]

                elif pending and any(
                    item.id == pending["id"]
                    for item in interrupts
                ):
                    value = Command(
                        resume={
                            pending["id"]: pending["decision"]
                        }
                    )

                elif interrupts or not snapshot.next:
                    invoke = False

                else:
                    # Continue unfinished work from its checkpoint.
                    value = None

                if invoke:
                    await graph.ainvoke(
                        value,
                        config=config,
                    )

                snapshot = await graph.aget_state(
                    config
                )

                status = snapshot.values.get(
                    "status",
                    "needs_review",
                )

                clear_review = True

        except asyncio.CancelledError:
            status = "interrupted"

        except Exception as exc:
            status = "failed"

            error = (
                f"Run failed ({type(exc).__name__}). "
                "Check server logs."
            )

            logger.exception(
                "Run %s failed",
                run_id,
            )

        finally:
            if provider is not None:
                try:
                    await provider.close()

                except Exception as exc:
                    logger.warning(
                        "Provider cleanup failed: %s",
                        type(exc).__name__,
                    )

            async with self._actions:
                data.update(
                    status=status,
                    error=error,
                )

                if clear_review:
                    data["pending_review"] = None

                try:
                    await self._save(data)

                finally:
                    self._tasks.pop(
                        run_id,
                        None,
                    )

    def _read_receipt(
        self,
        path: Path,
        job_hash: str,
    ) -> dict:
        output = json.loads(
            path.read_text("utf-8")
        )

        if output["job_sha256"] != job_hash:
            raise ValueError(
                "Training receipt does not match this job."
            )

        manifest = output["manifest"]

        verified = self.artifacts.get_manifest(
            run_id=UUID(manifest["run_id"]),
            artifact_id=UUID(manifest["artifact_id"]),
        )

        if verified != manifest:
            raise ValueError(
                "Training manifest changed."
            )

        report_path = (
            self.artifacts.root
            / manifest["run_id"]
            / "artifacts"
            / manifest["artifact_id"]
            / "results.json"
        )

        if (
            json.loads(report_path.read_text("utf-8"))
            != output["report"]
        ):
            raise ValueError(
                "Training report differs from the saved artifact."
            )

        return {
            "report": output["report"],
            "manifest": manifest,
        }

    async def _train(self, **kwargs) -> dict:
        async with self._training_slot:
            raw = kwargs["dataset_bytes"]

            self.reviews.require_approved(
                run_id=kwargs["run_id"],
                revision=kwargs["revision"],
                dataset_sha256=hashlib.sha256(
                    raw
                ).hexdigest(),
                plan_sha256=execution_plan_sha256(
                    kwargs["execution_plan"]
                ),
            )

            arguments = {
                "run_id": str(kwargs["run_id"]),
                "revision": kwargs["revision"],
                "dataset_id": str(kwargs["dataset_id"]),
                "dataset_sha256": hashlib.sha256(
                    raw
                ).hexdigest(),
                **{
                    name: kwargs[name].model_dump(
                        mode="json"
                    )
                    for name in (
                        "execution_plan",
                        "decision",
                        "runtime",
                    )
                },
            }

            job_hash = hashlib.sha256(
                _json(arguments).encode("utf-8")
            ).hexdigest()

            receipt = (
                self.receipts
                / f"{job_hash}.json"
            )

            if receipt.exists():
                return await asyncio.to_thread(
                    self._read_receipt,
                    receipt,
                    job_hash,
                )

            with tempfile.TemporaryDirectory(
                prefix="training-",
                dir=self.root,
            ) as temporary:
                directory = Path(temporary)

                await asyncio.to_thread(
                    (directory / "dataset.csv").write_bytes,
                    raw,
                )

                await asyncio.to_thread(
                    _write_json,
                    directory / "job.json",
                    {
                        "arguments": arguments,
                        "job_sha256": job_hash,
                        "receipt_path": str(receipt),
                        "review_path": str(self.review_path),
                        "artifact_root": str(self.artifacts.root),
                    },
                )

                process = multiprocessing.get_context(
                    "spawn"
                ).Process(
                    target=_training_worker,
                    args=(str(directory),),
                    daemon=True,
                )

                process.start()

                try:
                    async with asyncio.timeout(
                        self.training_timeout
                    ):
                        while process.is_alive():
                            await asyncio.sleep(0.2)

                    if process.exitcode != 0:
                        raise RuntimeError(
                            "Training worker failed; "
                            "inspect its server traceback."
                        )

                    return await asyncio.to_thread(
                        self._read_receipt,
                        receipt,
                        job_hash,
                    )

                finally:
                    if process.is_alive():
                        process.terminate()

                        await asyncio.to_thread(
                            process.join,
                            2,
                        )

                    if process.is_alive():
                        process.kill()

                        await asyncio.to_thread(
                            process.join,
                            2,
                        )

                    if process.is_alive():
                        raise RuntimeError(
                            "Training worker could not be stopped."
                        )

                    process.join(timeout=0)
                    process.close()

    async def close(self) -> None:
        async with self._actions:
            if not self._opened or self._closed:
                return

            self._closed = True

            tasks = list(
                self._tasks.values()
            )

        for task in tasks:
            task.cancel()

        await asyncio.gather(
            *tasks,
            return_exceptions=True,
        )

        await self._stack.aclose()

        self._opened = False