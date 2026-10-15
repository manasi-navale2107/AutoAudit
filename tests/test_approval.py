
import pytest
from pydantic import ValidationError

from app.graph.workflow import ReviewDecision


def test_approve_decision_is_valid():
    decision = ReviewDecision.model_validate(
        {
            "revision": 1,
            "action": "approve",
            "feedback": None,
        }
    )

    assert decision.revision == 1
    assert decision.action == "approve"
    assert decision.feedback is None


@pytest.mark.parametrize("action", ["changes", "answer"])
def test_changes_and_answer_require_feedback(action):
    with pytest.raises(ValidationError):
        ReviewDecision.model_validate(
            {
                "revision": 1,
                "action": action,
                "feedback": None,
            }
        )


def test_changes_decision_accepts_feedback():
    decision = ReviewDecision.model_validate(
        {
            "revision": 2,
            "action": "changes",
            "feedback": "Please review the missing-value handling.",
        }
    )

    assert decision.action == "changes"
    assert decision.feedback == (
        "Please review the missing-value handling."
    )


def test_answer_decision_accepts_feedback():
    decision = ReviewDecision.model_validate(
        {
            "revision": 1,
            "action": "answer",
            "feedback": "Predict the outcome before the event.",
        }
    )

    assert decision.action == "answer"
    assert decision.feedback == (
        "Predict the outcome before the event."
    )


def test_reject_decision_is_valid():
    decision = ReviewDecision.model_validate(
        {
            "revision": 1,
            "action": "reject",
            "feedback": None,
        }
    )

    assert decision.action == "reject"


def test_feedback_is_trimmed():
    decision = ReviewDecision.model_validate(
        {
            "revision": 1,
            "action": "changes",
            "feedback": "  Fix the preprocessing  ",
        }
    )

    assert decision.feedback == "Fix the preprocessing"


def test_blank_feedback_is_rejected():
    with pytest.raises(ValidationError):
        ReviewDecision.model_validate(
            {
                "revision": 1,
                "action": "changes",
                "feedback": "   ",
            }
        )


def test_approve_with_feedback_is_rejected():
    with pytest.raises(ValidationError):
        ReviewDecision.model_validate(
            {
                "revision": 1,
                "action": "approve",
                "feedback": "Please make changes.",
            }
        )


@pytest.mark.parametrize("revision", [0, -1, 1.5, True])
def test_invalid_revision_is_rejected(revision):
    with pytest.raises(ValidationError):
        ReviewDecision.model_validate(
            {
                "revision": revision,
                "action": "approve",
                "feedback": None,
            }
        )


def test_unknown_action_is_rejected():
    with pytest.raises(ValidationError):
        ReviewDecision.model_validate(
            {
                "revision": 1,
                "action": "continue",
                "feedback": None,
            }
        )


def test_extra_fields_are_rejected():
    with pytest.raises(ValidationError):
        ReviewDecision.model_validate(
            {
                "revision": 1,
                "action": "approve",
                "feedback": None,
                "unexpected": "not allowed",
            }
        )
