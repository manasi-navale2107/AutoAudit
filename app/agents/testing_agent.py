
"""Safe, static verification support for AutoAudit."""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class TestingResult:
    status: str
    syntax_valid: bool
    executed: bool
    checks: list[dict[str, Any]]
    test_targets: list[str]
    errors: list[str]
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TestingAgent:
    """Perform static source checks without executing submitted code."""

    def run(
        self,
        source: str,
        *,
        execute: bool = False,
    ) -> TestingResult:
        if not isinstance(source, str):
            return TestingResult(
                status="failed",
                syntax_valid=False,
                executed=False,
                checks=[],
                test_targets=[],
                errors=["Source must be a string."],
                summary="Verification failed: invalid input type.",
            )

        try:
            tree = ast.parse(source)
        except SyntaxError as exc:
            message = (
                f"Syntax error at line {exc.lineno}: {exc.msg}"
            )
            return TestingResult(
                status="failed",
                syntax_valid=False,
                executed=False,
                checks=[{
                    "name": "python_syntax",
                    "passed": False,
                    "detail": message,
                }],
                test_targets=[],
                errors=[message],
                summary="Static verification failed.",
            )

        targets = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                                 ast.ClassDef))
        ]

        checks = [{
            "name": "python_syntax",
            "passed": True,
            "detail": "Python source parsed successfully.",
        }]

        errors = []
        if execute:
            errors.append(
                "Runtime execution is not implemented. "
                "Run tests in a separately isolated environment."
            )

        return TestingResult(
            status="needs_review",
            syntax_valid=True,
            executed=False,
            checks=checks,
            test_targets=targets,
            errors=errors,
            summary=(
                "Syntax check passed. Runtime behavior has not been "
                "verified; additional tests and review are required."
            ),
        )
