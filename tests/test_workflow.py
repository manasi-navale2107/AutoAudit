
from unittest.mock import MagicMock

import pytest
from langgraph.checkpoint.memory import MemorySaver

from app.graph.workflow import build_workflow


def make_workflow_kwargs():
    """Create lightweight dependencies for graph compilation tests."""
    return {
        "planner": MagicMock(),
        "engineer": MagicMock(),
        "training_agent": MagicMock(),
        "review_store": MagicMock(),
        "dataset_loader": MagicMock(),
        "training_runner": MagicMock(),
        "checkpointer": MemorySaver(),
    }


def test_build_workflow_returns_compiled_graph():
    workflow = build_workflow(**make_workflow_kwargs())

    assert workflow is not None
    assert callable(workflow.invoke)
    assert callable(workflow.get_graph)


def test_build_workflow_requires_checkpointer():
    kwargs = make_workflow_kwargs()
    kwargs["checkpointer"] = None

    with pytest.raises(
        ValueError,
        match="A checkpointer is required for human review",
    ):
        build_workflow(**kwargs)


@pytest.mark.parametrize("max_revisions", [0, -1, 11, 100])
def test_build_workflow_rejects_invalid_max_revisions(
    max_revisions,
):
    kwargs = make_workflow_kwargs()
    kwargs["max_revisions"] = max_revisions

    with pytest.raises(
        ValueError,
        match="max_revisions must be between 1 and 10",
    ):
        build_workflow(**kwargs)


@pytest.mark.parametrize("max_revisions", [1, 5, 10])
def test_build_workflow_accepts_valid_max_revisions(
    max_revisions,
):
    kwargs = make_workflow_kwargs()
    kwargs["max_revisions"] = max_revisions

    workflow = build_workflow(**kwargs)

    assert workflow is not None
    assert callable(workflow.invoke)
