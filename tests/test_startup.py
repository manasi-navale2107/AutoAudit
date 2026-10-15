
"""AutoAudit startup smoke test.

Run from project root:
    python -X utf8 tests/test_startup.py
    python -X utf8 tests/test_startup.py --live

The test does not contact an LLM or train models.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REQUIRED_FILES = (
    "app/main.py",
    "app/schemas/project.py",
    "app/schemas/run.py",
    "app/prompts/planner.txt",
    "app/agents/planner_agent.py",
    "app/graph/workflow.py",
    "app/services/project_workspace_service.py",
    "app/services/project_code_generation_service.py",
    "app/services/run_service.py",
    "app/api/routes/project_planning.py",
    "frontend/index.html",
    "frontend/js/api.js",
    "frontend/js/app.js",
    "frontend/js/review.js",
    "frontend/css/styles.css",
)

EXPECTED_ENDPOINTS = (
    ("GET", "/api/health"),
    ("GET", "/api/ready"),
    ("POST", "/api/datasets"),
    ("POST", "/api/runs"),
    ("POST", "/api/runs/{run_id}/reviews"),
    ("GET", "/api/runs/{run_id}/project"),
    ("POST", "/api/projects/plan"),
    ("GET", "/api/projects/plan/{run_id}"),
)

issues: list[str] = []


def report(label: str, success: bool, detail: str = "") -> None:
    print(
        f"[{'PASS' if success else 'FAIL'}] {label}"
        + (f": {detail}" if detail else "")
    )
    if not success:
        issues.append(
            label + (f": {detail}" if detail else "")
        )


def check_required_files() -> bool:
    missing = [
        name
        for name in REQUIRED_FILES
        if not (ROOT / name).is_file()
    ]

    report(
        "Required files",
        not missing,
        ", ".join(missing) if missing else "All present",
    )
    return not missing


def check_python() -> None:
    failures: list[str] = []
    count = 0

    for source in sorted((ROOT / "app").rglob("*.py")):
        count += 1

        try:
            compile(
                source.read_bytes(),
                str(source),
                "exec",
            )
        except (
            SyntaxError,
            OSError,
            UnicodeError,
        ) as exc:
            failures.append(
                f"{source.relative_to(ROOT)}: {exc}"
            )

    report(
        "Python syntax",
        count > 0 and not failures,
        (
            "; ".join(failures[:5])
            if failures
            else f"{count} files checked"
        ),
    )


def check_local_imports() -> None:
    missing: set[str] = set()
    checked = 0

    for source in (ROOT / "app").rglob("*.py"):
        try:
            tree = ast.parse(
                source.read_text(encoding="utf-8")
            )
        except (
            SyntaxError,
            OSError,
            UnicodeError,
        ):
            continue

        for node in ast.walk(tree):
            if (
                not isinstance(node, ast.ImportFrom)
                or not node.module
            ):
                continue

            if node.level != 0 or not (
                node.module == "app"
                or node.module.startswith("app.")
            ):
                continue

            checked += 1
            target = ROOT.joinpath(
                *node.module.split(".")
            )

            if not (
                target.with_suffix(".py").is_file()
                or target.is_dir()
            ):
                missing.add(node.module)

    report(
        "Local module paths",
        not missing,
        (
            ", ".join(sorted(missing))
            if missing
            else f"{checked} imports checked"
        ),
    )


def check_frontend() -> None:
    html = (
        ROOT / "frontend/index.html"
    ).read_text(encoding="utf-8")

    app_js = (
        ROOT / "frontend/js/app.js"
    ).read_text(encoding="utf-8")

    api_js = (
        ROOT / "frontend/js/api.js"
    ).read_text(encoding="utf-8")

    review_js = (
        ROOT / "frontend/js/review.js"
    ).read_text(encoding="utf-8")

    ids = re.findall(
        r'\bid\s*=\s*["\']([^"\']+)["\']',
        html,
    )

    duplicated = sorted({
        identifier
        for identifier in ids
        if ids.count(identifier) > 1
    })

    report(
        "Unique HTML IDs",
        not duplicated,
        (
            ", ".join(duplicated)
            if duplicated
            else f"{len(ids)} IDs found"
        ),
    )

    referenced = set(
        re.findall(
            r'\$\(["\']([a-zA-Z][\w-]*)["\']\)',
            app_js,
        )
    )

    missing_ids = sorted(
        referenced - set(ids)
    )

    report(
        "HTML references in app.js",
        not missing_ids,
        (
            ", ".join(missing_ids)
            if missing_ids
            else "All static IDs found"
        ),
    )

    methods = set(
        re.findall(
            r"^\s{2}async\s+([a-zA-Z]\w*)\s*\(",
            api_js,
            flags=re.MULTILINE,
        )
    )

    called = set(
        re.findall(
            r"\bapi\.([a-zA-Z]\w*)\s*\(",
            app_js + "\n" + review_js,
        )
    )

    missing_methods = sorted(
        called - methods
    )

    report(
        "Frontend API methods",
        not missing_methods,
        (
            ", ".join(missing_methods)
            if missing_methods
            else f"{len(called)} matched"
        ),
    )

    node = shutil.which("node")

    if node is None:
        print(
            "[SKIP] JavaScript syntax: Node.js unavailable"
        )
        return

    for name in (
        "app.js",
        "api.js",
        "review.js",
    ):
        file = ROOT / "frontend/js" / name

        try:
            output = subprocess.run(
                [
                    node,
                    "--input-type=module",
                    "--check",
                ],
                input=file.read_text(
                    encoding="utf-8"
                ),
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                timeout=20,
                check=False,
            )

            report(
                f"JavaScript syntax: {name}",
                output.returncode == 0,
                (
                    "Valid syntax"
                    if output.returncode == 0
                    else output.stderr.strip()[:500]
                ),
            )

        except (
            OSError,
            subprocess.TimeoutExpired,
        ) as exc:
            report(
                f"JavaScript syntax: {name}",
                False,
                str(exc),
            )


def check_api(*, live: bool) -> None:
    try:
        application = importlib.import_module(
            "app.main"
        ).app

    except Exception as exc:
        report(
            "FastAPI import",
            False,
            f"{type(exc).__name__}: {exc}",
        )
        return

    report(
        "FastAPI import",
        True,
        "app.main imported",
    )

    # FastAPI 0.137+ may store included routers as
    # _IncludedRouter objects inside app.routes.
    #
    # Those objects do not provide .path.
    #
    # Use OpenAPI to inspect actual registered
    # HTTP endpoints across FastAPI versions.

    try:
        openapi = application.openapi()
        paths = openapi.get("paths", {})

        registered = {
            (method.upper(), path)
            for path, operations in paths.items()
            for method in operations
            if method.lower() in {
                "get",
                "post",
                "put",
                "patch",
                "delete",
                "options",
                "head",
                "trace",
            }
        }

        missing = [
            f"{method} {path}"
            for method, path in EXPECTED_ENDPOINTS
            if (method, path) not in registered
        ]

        report(
            "API route registration",
            not missing,
            (
                ", ".join(missing)
                if missing
                else (
                    f"{len(EXPECTED_ENDPOINTS)} "
                    "expected endpoints found"
                )
            ),
        )

    except Exception as exc:
        report(
            "API route registration",
            False,
            (
                "OpenAPI failed: "
                f"{type(exc).__name__}: {exc}"
            ),
        )

    if not live:
        print(
            "[SKIP] FastAPI lifespan. "
            "Run with --live to check startup."
        )
        return

    try:
        from fastapi.testclient import TestClient

        with TestClient(application) as client:
            for endpoint in (
                "/api/health",
                "/api/ready",
            ):
                try:
                    response = client.get(
                        endpoint,
                        timeout=10,
                    )

                    report(
                        f"GET {endpoint}",
                        response.status_code == 200,
                        f"HTTP {response.status_code}",
                    )

                except Exception as exc:
                    report(
                        f"GET {endpoint}",
                        False,
                        (
                            f"{type(exc).__name__}: "
                            f"{exc}"
                        ),
                    )

    except Exception as exc:
        report(
            "FastAPI lifespan",
            False,
            f"{type(exc).__name__}: {exc}",
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="AutoAudit smoke test"
    )

    parser.add_argument(
        "--live",
        action="store_true",
        help="Run startup and HTTP readiness checks",
    )

    options = parser.parse_args()

    print("=" * 56)
    print("AUTOAUDIT STARTUP TEST")
    print("=" * 56)
    print("Project:", ROOT)
    print()

    if check_required_files():
        check_python()
        check_local_imports()

        try:
            check_frontend()
        except Exception as exc:
            report(
                "Frontend inspection",
                False,
                f"{type(exc).__name__}: {exc}",
            )

        check_api(live=options.live)

    print()
    print("=" * 56)

    if issues:
        print(
            f"RESULT: {len(issues)} problem(s) found"
        )
        print(
            "Share the first FAIL line "
            "and its traceback/output."
        )
    else:
        print(
            "RESULT: ALL PERFORMED CHECKS PASSED"
        )

    print("=" * 56)

    return 1 if issues else 0


if __name__ == "__main__":
    raise SystemExit(main())
