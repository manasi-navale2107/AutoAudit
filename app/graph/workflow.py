from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable
from typing import Literal, TypedDict
from uuid import UUID

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agents.data_engineering_agent import DataEngineeringAgent
from app.agents.model_training_agent import (
    ModelTrainingAgent,
    execution_plan_sha256,
)
from app.agents.planner_agent import PlannerAgent
from app.ml.model_factory import TrainingRuntimeSettings
from app.ml.profiling import profile_dataset
from app.ml.splitting import split_dataset
from app.ml.training import load_training_frame
from app.schemas.cleaning import CleaningProposal, ExecutionPlan
from app.schemas.run import RunCreateRequest
from app.services.cleaning_review_store import CleaningReviewStore


class WorkflowState(TypedDict, total=False):
    run_id: str
    request: dict
    dataset_sha256: str
    runtime: dict
    status: str

    planning_round: int
    answers: list[str]
    planner_decision: dict
    execution_plan: dict | None

    revision: int
    proposal: dict | None
    feedback: str | None

    training_decision: dict | None
    training_report: dict
    artifact: dict


class ReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    revision: int = Field(ge=1, strict=True)

    action: Literal[
        "approve",
        "changes",
        "reject",
        "answer",
    ]

    feedback: str | None = Field(
        default=None,
        max_length=4000,
    )

    @model_validator(mode="after")
    def check_feedback(self):
        if self.feedback is not None:
            self.feedback = self.feedback.strip()

            if not self.feedback:
                raise ValueError(
                    "Feedback must not be blank."
                )

        if (
            self.action in {"changes", "answer"}
            and not self.feedback
        ):
            raise ValueError(
                "This action requires feedback."
            )

        if (
            self.action == "approve"
            and self.feedback is not None
        ):
            raise ValueError(
                "Use Request changes for proposal modifications."
            )

        return self


def create_initial_state(
    *,
    run_id: UUID,
    request: RunCreateRequest,
    dataset_bytes: bytes,
    runtime: TrainingRuntimeSettings | None = None,
) -> WorkflowState:
    load_training_frame(dataset_bytes)

    settings = (
        runtime
        if runtime is not None
        else TrainingRuntimeSettings()
    )

    settings = TrainingRuntimeSettings.model_validate(
        settings.model_dump()
    )

    request_data = request.model_dump(mode="json")
    request_data["language"] = (
        request_data.get("language") or "mr"
    )

    request_data = RunCreateRequest.model_validate(
        request_data
    ).model_dump(mode="json")

    return {
        "run_id": str(UUID(str(run_id))),
        "request": request_data,
        "dataset_sha256": hashlib.sha256(
            dataset_bytes
        ).hexdigest(),
        "runtime": settings.model_dump(mode="json"),
        "status": "queued",
        "planning_round": 0,
        "answers": [],
        "revision": 0,
        "proposal": None,
        "feedback": None,
        "training_decision": None,
    }


def _request_with_answers(state, answers):
    payload = dict(state["request"])

    if answers:
        payload["objective"] += (
            "\nUser clarifications:\n"
            + "\n".join(answers)
        )

    return RunCreateRequest.model_validate(payload)


def build_workflow(
    *,
    planner: PlannerAgent,
    engineer: DataEngineeringAgent,
    training_agent: ModelTrainingAgent,
    review_store: CleaningReviewStore,
    dataset_loader: Callable[[UUID], bytes],
    training_runner: Callable[..., Awaitable[dict]],
    checkpointer,
    max_revisions: int = 5,
):
    """
    dataset_loader(dataset_id) returns the stored original CSV bytes.

    training_runner receives TrainingExecutor.fit keyword arguments,
    executes training in a bounded worker process, saves artifacts,
    and returns {"report": result.report, "manifest": saved_manifest}.

    RunService must supply these callbacks, the same review store used
    by the agents, a live async-compatible checkpointer, and per-run
    concurrency control. Use a durable checkpointer for restart recovery.
    """
    if checkpointer is None:
        raise ValueError(
            "A checkpointer is required for human review."
        )

    if not 1 <= max_revisions <= 10:
        raise ValueError(
            "max_revisions must be between 1 and 10."
        )

    def guard(state, config):
        thread_id = config.get(
            "configurable", {}
        ).get("thread_id")

        if str(thread_id) != state["run_id"]:
            raise ValueError(
                "thread_id must equal the run_id."
            )

    async def dataset(state):
        dataset_id = UUID(
            state["request"]["dataset_id"]
        )

        raw = await asyncio.to_thread(
            dataset_loader,
            dataset_id,
        )

        if not isinstance(raw, bytes):
            raise TypeError(
                "dataset_loader must return bytes."
            )

        actual_hash = hashlib.sha256(raw).hexdigest()

        if actual_hash != state["dataset_sha256"]:
            raise ValueError(
                "Dataset bytes changed. Start a new run."
            )

        frame = await asyncio.to_thread(
            load_training_frame,
            raw,
        )

        return dataset_id, raw, frame

    async def planning(
        state: WorkflowState,
        config: RunnableConfig,
    ):
        guard(state, config)

        dataset_id, _, frame = await dataset(state)

        profile = await asyncio.to_thread(
            profile_dataset,
            frame=frame,
            dataset_id=dataset_id,
        )

        decision = await planner.plan(
            request=_request_with_answers(
                state,
                state["answers"],
            ),
            profile=profile,
            frame=frame,
        )

        result = decision.model_dump(mode="json")

        ready = (
            result["status"] == "ready"
            and result.get("plan") is not None
        )

        round_number = state["planning_round"] + 1

        status = (
            "plan_ready"
            if ready
            else "needs_review"
        )

        if (
            not ready
            and result.get("clarification_questions")
            and round_number < 5
        ):
            status = "awaiting_clarification"

        return {
            "planner_decision": result,
            "planning_round": round_number,
            "execution_plan": (
                result.get("plan") if ready else None
            ),
            "status": status,
        }

    def clarification(
        state: WorkflowState,
        config: RunnableConfig,
    ):
        guard(state, config)

        response = interrupt(
            {
                "kind": "planner_clarification",
                "revision": state["planning_round"],
                "questions": state["planner_decision"][
                    "clarification_questions"
                ],
            }
        )

        decision = ReviewDecision.model_validate(
            response
        )

        if (
            decision.revision != state["planning_round"]
            or decision.action not in {"answer", "reject"}
        ):
            raise ValueError(
                "Respond to the current Planner questions."
            )

        if decision.action == "reject":
            return {"status": "rejected"}

        answered = json.dumps(
            {
                "questions": state["planner_decision"][
                    "clarification_questions"
                ],
                "answer": decision.feedback,
            },
            ensure_ascii=False,
        )

        answers = state["answers"] + [answered]

        # Enforce the request length limit before storing the answer.
        _request_with_answers(state, answers)

        return {
            "answers": answers,
            "status": "replanning",
        }

    async def cleaning(
        state: WorkflowState,
        config: RunnableConfig,
    ):
        guard(state, config)

        dataset_id, _, frame = await dataset(state)

        plan = ExecutionPlan.model_validate(
            state["execution_plan"]
        )

        plan_hash = execution_plan_sha256(plan)

        try:
            existing = review_store.get_latest(
                run_id=UUID(state["run_id"])
            )
        except KeyError:
            existing = None

        # Recover a submission committed just before its graph checkpoint.
        if (
            existing
            and existing["revision"] == state["revision"] + 1
        ):
            if (
                existing["status"] != "pending"
                or existing["dataset_sha256"]
                != state["dataset_sha256"]
                or existing["plan_sha256"] != plan_hash
            ):
                raise ValueError(
                    "Stored review does not match this workflow stage."
                )

            return {
                "proposal": existing["proposal"],
                "revision": existing["revision"],
                "feedback": None,
                "training_decision": None,
                "status": "awaiting_review",
            }

        profile = await asyncio.to_thread(
            profile_dataset,
            frame=frame,
            dataset_id=dataset_id,
        )

        previous = (
            CleaningProposal.model_validate(
                state["proposal"]
            )
            if state["proposal"]
            else None
        )

        proposal = await engineer.propose(
            execution_plan=plan,
            profile=profile,
            frame=frame,
            language=state["request"]["language"],
            previous_proposal=previous,
            user_feedback=state["feedback"],
        )

        row = review_store.submit(
            run_id=UUID(state["run_id"]),
            proposal=proposal,
            dataset_sha256=state["dataset_sha256"],
            plan_sha256=plan_hash,
            expected_revision=state["revision"],
        )

        return {
            "proposal": row["proposal"],
            "revision": row["revision"],
            "feedback": None,
            "training_decision": None,
            "status": "awaiting_review",
        }

    def human_review(
        state: WorkflowState,
        config: RunnableConfig,
    ):
        guard(state, config)

        response = interrupt(
            {
                "kind": "cleaning_review",
                "revision": state["revision"],
                "proposal": state["proposal"],
            }
        )

        decision = ReviewDecision.model_validate(
            response
        )

        if (
            decision.revision != state["revision"]
            or decision.action == "answer"
        ):
            raise ValueError(
                "Review the current cleaning revision."
            )

        review_store.decide(
            run_id=UUID(state["run_id"]),
            revision=decision.revision,
            action=decision.action,
            feedback=decision.feedback,
        )

        if decision.action == "approve":
            return {"status": "approved"}

        if decision.action == "reject":
            return {"status": "rejected"}

        return {
            "feedback": decision.feedback,
            "training_decision": None,
            "status": (
                "revising"
                if state["revision"] < max_revisions
                else "needs_review"
            ),
        }

    async def training_settings(
        state: WorkflowState,
        config: RunnableConfig,
    ):
        guard(state, config)

        _, _, frame = await dataset(state)

        plan = ExecutionPlan.model_validate(
            state["execution_plan"]
        )

        runtime = TrainingRuntimeSettings.model_validate(
            state["runtime"]
        )

        split = await asyncio.to_thread(
            split_dataset,
            frame=frame,
            execution_plan=plan,
            random_seed=runtime.random_seed,
        )

        decision = await training_agent.propose(
            run_id=UUID(state["run_id"]),
            revision=state["revision"],
            dataset_sha256=state["dataset_sha256"],
            execution_plan=plan,
            training_frame=frame.iloc[list(split.train)],
            language=state["request"]["language"],
        )

        return {
            "training_decision": decision.model_dump(
                mode="json"
            ),
            "status": "training",
        }

    async def training(
        state: WorkflowState,
        config: RunnableConfig,
    ):
        from app.agents.model_training_agent import (
            TrainingDecision,
        )

        guard(state, config)

        dataset_id, raw, _ = await dataset(state)

        plan = ExecutionPlan.model_validate(
            state["execution_plan"]
        )

        plan_hash = execution_plan_sha256(plan)

        review_store.require_approved(
            run_id=UUID(state["run_id"]),
            revision=state["revision"],
            dataset_sha256=state["dataset_sha256"],
            plan_sha256=plan_hash,
        )

        output = await training_runner(
            run_id=UUID(state["run_id"]),
            revision=state["revision"],
            dataset_id=dataset_id,
            dataset_bytes=raw,
            execution_plan=plan,
            decision=TrainingDecision.model_validate(
                state["training_decision"]
            ),
            runtime=TrainingRuntimeSettings.model_validate(
                state["runtime"]
            ),
        )

        # Keep checkpoint state JSON-compatible and reject NaN/Infinity.
        output = json.loads(
            json.dumps(output, allow_nan=False)
        )

        report = output["report"]
        manifest = output["manifest"]

        for item in (report, manifest):
            if (
                item.get("run_id") != state["run_id"]
                or item.get("approved_revision")
                != state["revision"]
                or item.get("dataset_sha256")
                != state["dataset_sha256"]
                or item.get("execution_plan_sha256")
                != plan_hash
                or item.get("test_evaluated") is not False
                or item.get("audit_status") != "not_performed"
            ):
                raise ValueError(
                    "Training output does not match the approved run."
                )

        if (
            report.get("status") != "trained"
            or manifest.get("status") != "ready_for_audit"
        ):
            raise ValueError(
                "Training and artifact saving must both complete."
            )

        return {
            "training_report": report,
            "artifact": manifest,
            "status": "ready_for_audit",
        }

    graph = StateGraph(WorkflowState)

    graph.add_node("planner", planning)
    graph.add_node("clarification", clarification)
    graph.add_node("cleaning", cleaning)
    graph.add_node("human_review", human_review)
    graph.add_node("training_settings", training_settings)
    graph.add_node("training", training)

    graph.add_edge(START, "planner")

    graph.add_conditional_edges(
        "planner",
        lambda state: {
            "plan_ready": "cleaning",
            "awaiting_clarification": "clarification",
        }.get(state["status"], END),
    )

    graph.add_conditional_edges(
        "clarification",
        lambda state: (
            "planner"
            if state["status"] == "replanning"
            else END
        ),
    )

    graph.add_edge("cleaning", "human_review")

    graph.add_conditional_edges(
        "human_review",
        lambda state: {
            "approved": "training_settings",
            "revising": "cleaning",
        }.get(state["status"], END),
    )

    graph.add_edge("training_settings", "training")
    graph.add_edge("training", END)

    return graph.compile(checkpointer=checkpointer)