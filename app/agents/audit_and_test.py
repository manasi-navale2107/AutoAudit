"""Small orchestration entry point for the two new downstream agents."""
from __future__ import annotations

from typing import Any

from app.agents.leakage_refactoring_agent import LeakageRefactoringAgent
from app.agents.testing_agent import TestingAgent


def audit_then_test(source: str, *, execute: bool = False) -> dict[str, Any]:
    """Audit source first, then test the repaired source (or original if no patch).

    The current refactoring stage is intentionally conservative: when findings
    need semantic changes, it reports recommendations and leaves source intact.
    """
    audit = LeakageRefactoringAgent().run(source, apply_repairs=True)
    candidate = audit.repaired_code if audit.repaired_code is not None else source
    testing = TestingAgent().run(candidate, execute=execute)
    return {
        "audit": audit.to_dict(),
        "testing": testing.to_dict(),
        "overall_status": "passed" if audit.syntax_valid and testing.status == "passed" else "needs_review",
        "note": "A passed test report does not prove absence of all data leakage. Review every audit finding.",
    }
