"""Deterministic and AST-based leakage auditor and safe refactoring assistant.

Audits machine learning pipelines, datasets, split manifests, trained models,
and project source code for data leakage, suspicious feature usage,
and maintainability problems.

Never executes arbitrary untrusted user code. Distinguishes confirmed leakage,
potential leakage, and informational findings.
"""
from __future__ import annotations

import ast
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LeakageFinding:
    rule_id: str
    severity: str  # critical, high, medium, low, informational
    message: str
    line: int | None
    recommendation: str
    auto_repairable: bool = False
    finding_id: str = ""
    category: str = "leakage"  # leakage, refactoring
    title: str = ""
    description: str = ""
    status: str = "confirmed"  # confirmed, potential, inconclusive
    source_file: str | None = None
    evidence: str = ""
    recommended_repair: str = ""
    auto_repair_applied: bool = False
    verification_result: str | None = None

    def __post_init__(self):
        # Default empty fields compatibly
        if not self.finding_id:
            object.__setattr__(self, "finding_id", self.rule_id)
        if not self.title:
            object.__setattr__(self, "title", self.message.split(":")[0] if ":" in self.message else self.message[:80])
        if not self.description:
            object.__setattr__(self, "description", self.message)
        if not self.recommended_repair:
            object.__setattr__(self, "recommended_repair", self.recommendation)


@dataclass(frozen=True)
class AuditResult:
    findings: list[LeakageFinding]
    syntax_valid: bool
    repaired_code: str | None = None
    repairs_applied: list[str] | None = None
    overall_status: str = "passed"  # passed, warning, failed, needs_review
    severity_counts: dict[str, int] = field(default_factory=dict)
    summary: str = ""

    def __post_init__(self):
        if not self.severity_counts:
            counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "informational": 0}
            for f in self.findings:
                sev = f.severity.lower()
                counts[sev] = counts.get(sev, 0) + 1
            object.__setattr__(self, "severity_counts", counts)
        if not self.summary:
            total = len(self.findings)
            crit = self.severity_counts.get("critical", 0)
            high = self.severity_counts.get("high", 0)
            if not self.syntax_valid:
                sm = "Syntax errors detected in source code."
                st = "failed"
            elif crit > 0:
                sm = f"Audit identified {crit} critical leakage issues that compromise model validity."
                st = "failed"
            elif high > 0:
                sm = f"Audit identified {high} high-severity findings requiring review."
                st = "warning"
            elif total > 0:
                sm = f"Audit identified {total} findings (no critical leakage)."
                st = "passed"
            else:
                sm = "Audit completed successfully: no data leakage or critical maintainability issues detected."
                st = "passed"
            object.__setattr__(self, "summary", sm)
            if self.overall_status == "passed" and st != "passed":
                object.__setattr__(self, "overall_status", st)

    def to_dict(self) -> dict[str, Any]:
        return {
            "syntax_valid": self.syntax_valid,
            "overall_status": self.overall_status,
            "summary": self.summary,
            "severity_counts": self.severity_counts,
            "findings": [asdict(item) for item in self.findings],
            "repaired_code": self.repaired_code,
            "repairs_applied": self.repairs_applied or [],
        }


class _Auditor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.findings: list[LeakageFinding] = []
        self.split_lines: list[int] = []
        self.fit_calls: list[ast.Call] = []
        self.fit_transform_calls: list[ast.Call] = []
        self.target_names: set[str] = set()

    @staticmethod
    def _call_name(node: ast.AST) -> str:
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.Attribute):
            return node.attr
        return ""

    def visit_Call(self, node: ast.Call) -> None:  # noqa: N802
        name = self._call_name(node.func)
        if name == "train_test_split":
            self.split_lines.append(node.lineno)
        if name in {"fit", "fit_transform"}:
            (self.fit_calls if name == "fit" else self.fit_transform_calls).append(node)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:  # noqa: N802
        value = node.value
        if isinstance(value, ast.Subscript) and isinstance(value.value, ast.Name):
            # Common target extraction pattern: y = df["target"] / df['label'].
            if isinstance(value.slice, ast.Constant) and isinstance(value.slice.value, str):
                if value.slice.value.lower() in {"target", "label", "class", "outcome", "y"}:
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            self.target_names.add(target.id)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:  # noqa: N802
        # Refactoring check: overly complex function (length > 80 lines)
        func_len = getattr(node, "end_lineno", node.lineno) - node.lineno
        if func_len > 80:
            self.findings.append(LeakageFinding(
                rule_id="REFACTOR_FUNCTION_COMPLEXITY",
                severity="low",
                message=f"Function {node.name!r} is {func_len} lines long; consider breaking it into modular sub-tasks.",
                line=node.lineno,
                recommendation=f"Decompose {node.name!r} into focused functions with single responsibilities.",
                category="refactoring",
                title=f"Function {node.name} exceeds 80 lines",
                description=f"Function {node.name} at line {node.lineno} is {func_len} lines long, making maintainability and unit testing difficult.",
                status="potential",
                evidence=f"Function span: lines {node.lineno} to {getattr(node, 'end_lineno', node.lineno)}",
            ))
        self.generic_visit(node)

    def visit_Try(self, node: ast.Try) -> None:  # noqa: N802
        # Refactoring check: bare except handler
        for handler in node.handlers:
            if handler.type is None:
                self.findings.append(LeakageFinding(
                    rule_id="REFACTOR_BARE_EXCEPT",
                    severity="medium",
                    message="Bare 'except:' clause catches all exceptions including system interrupts.",
                    line=handler.lineno,
                    recommendation="Specify explicit exception types (e.g. 'except ValueError:') rather than catching all exceptions.",
                    category="refactoring",
                    title="Bare except clause detected",
                    description="Bare except clauses can conceal bugs, cancellations, and keyboard interrupts.",
                    status="confirmed",
                    evidence=f"Bare except at line {handler.lineno}",
                ))
        self.generic_visit(node)


def audit_code(source: str, source_file: str | None = None) -> AuditResult:
    """Audit Python source using syntax checks and conservative static rules."""
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source must be a non-empty Python string")
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        finding = LeakageFinding(
            rule_id="PYTHON_SYNTAX",
            severity="critical",
            message=f"Python syntax error: {exc.msg}",
            line=int(exc.lineno or 1),
            recommendation="Fix syntax errors before leakage analysis.",
            auto_repairable=False,
            category="refactoring",
            title="Python Syntax Error",
            description=f"Syntax error at line {exc.lineno}: {exc.msg}",
            status="confirmed",
            source_file=source_file,
            evidence=str(exc),
        )
        return AuditResult([finding], False, overall_status="failed")

    visitor = _Auditor()
    visitor.visit(tree)
    findings: list[LeakageFinding] = list(visitor.findings)

    if visitor.split_lines:
        first_split = min(visitor.split_lines)
        for call in visitor.fit_transform_calls + visitor.fit_calls:
            if call.lineno < first_split:
                findings.append(LeakageFinding(
                    rule_id="PREPROCESSING_BEFORE_SPLIT",
                    severity="high",
                    message="A fit/fit_transform call appears before train_test_split; this may fit preprocessing on held-out data.",
                    line=call.lineno,
                    recommendation="Split first, then fit preprocessing only on training data; prefer sklearn Pipeline/ColumnTransformer.",
                    auto_repairable=False,
                    category="leakage",
                    title="Preprocessing Fitted Before Data Split",
                    description="Transformers fitted on the entire dataset before splitting cause data leakage from validation/test sets.",
                    status="confirmed",
                    source_file=source_file,
                    evidence=f"Call at line {call.lineno} occurs before train_test_split at line {first_split}.",
                ))

    # Report likely leakage-prone resampling order without assuming imported aliases.
    resample_lines = [
        n.lineno for n in ast.walk(tree)
        if isinstance(n, ast.Call) and _Auditor._call_name(n.func).lower() in {"fit_resample", "smote"}
    ]
    if resample_lines and visitor.split_lines and min(resample_lines) < min(visitor.split_lines):
        findings.append(LeakageFinding(
            rule_id="RESAMPLING_BEFORE_SPLIT",
            severity="high",
            message="Resampling appears before train_test_split, which can contaminate validation/test data.",
            line=min(resample_lines),
            recommendation="Split first and apply resampling only to each training fold, e.g. with imbalanced-learn Pipeline.",
            auto_repairable=False,
            category="leakage",
            title="Resampling Applied Before Data Split",
            description="Synthetic oversampling or resampling before data splitting duplicates or leaks patterns into held-out sets.",
            status="confirmed",
            source_file=source_file,
            evidence=f"Resampling at line {min(resample_lines)} occurs before train_test_split at line {min(visitor.split_lines)}.",
        ))

    # Duplicate exact findings by rule and line are suppressed for predictable reports.
    unique = {(f.rule_id, f.line): f for f in findings}
    return AuditResult(list(unique.values()), True)


class LeakageRefactoringAgent:
    """Agent 4: Audits trained ML pipelines and source code for leakage and maintainability."""

    def run(self, source: str, *, apply_repairs: bool = False) -> AuditResult:
        """Audit standalone source code string."""
        result = audit_code(source)
        repaired = source if apply_repairs and result.syntax_valid and not result.findings else None
        applied = ["No changes required; source passed implemented static rules."] if repaired is not None else []
        return AuditResult(
            result.findings,
            result.syntax_valid,
            repaired,
            applied,
            overall_status=result.overall_status,
            severity_counts=result.severity_counts,
            summary=result.summary,
        )

    def audit_pipeline(
        self,
        *,
        execution_plan: dict | None,
        cleaning_proposal: dict | None,
        training_report: dict | None,
        splits: dict | None,
        frame: pd.DataFrame | None,
        source_files: dict[str, str] | None = None,
    ) -> AuditResult:
        """
        Perform comprehensive deterministic leakage and refactoring audit on the pipeline artifacts.
        """
        findings: list[LeakageFinding] = []

        # ------------------------------------------------------------
        # 1. Target Column Leakage Check
        # ------------------------------------------------------------
        target_column = None
        if execution_plan:
            target_column = execution_plan.get("target_column")

        if target_column and cleaning_proposal:
            policies = cleaning_proposal.get("policies", [])
            for policy in policies:
                col = policy.get("column")
                action = policy.get("action")
                if col == target_column and action in {"keep", "KEEP"}:
                    findings.append(LeakageFinding(
                        rule_id="LEAK_TARGET_IN_FEATURES",
                        severity="critical",
                        message=f"Target column {target_column!r} is included in training features with action 'keep'.",
                        line=None,
                        recommendation=f"Exclude target column {target_column!r} from model input features.",
                        auto_repairable=True,
                        category="leakage",
                        title="Target Column Included in Predictors",
                        description=f"The outcome column {target_column!r} was configured as an input feature, guaranteeing trivial 100% target leakage.",
                        status="confirmed",
                        evidence=f"Cleaning policy for column '{col}': action='{action}'",
                        recommended_repair=f"Set action='exclude' for '{target_column}'.",
                    ))

        # ------------------------------------------------------------
        # 2. Suspicious Column / High-Cardinality Identifier Predictor Check
        # ------------------------------------------------------------
        if cleaning_proposal and frame is not None:
            policies = cleaning_proposal.get("policies", [])
            id_keywords = {"id", "uuid", "guid", "key", "index", "customerid", "client_id", "userid", "user_id"}
            n_rows = len(frame)

            for policy in policies:
                col = policy.get("column", "")
                action = policy.get("action", "")
                if action not in {"keep", "KEEP"} or col == target_column:
                    continue

                col_lower = col.lower().replace("_", "").replace("-", "")
                # Check for high-cardinality unique string columns (e.g. customer IDs)
                if col in frame.columns:
                    series = frame[col].dropna()
                    n_unique = series.nunique()
                    is_near_unique = (n_rows > 30 and n_unique / max(1, len(series)) >= 0.95)
                    has_id_name = any(kw in col_lower for kw in ["customerid", "clientid", "userid", "accountid", "rowid"]) or col_lower.endswith("id")

                    if is_near_unique and has_id_name:
                        findings.append(LeakageFinding(
                            rule_id="LEAK_IDENTIFIER_PREDICTOR",
                            severity="high",
                            message=f"Column {col!r} appears to be an entity identifier ({n_unique} unique values in {len(series)} rows) kept as a predictor.",
                            line=None,
                            recommendation=f"Exclude identifier column {col!r} from model predictors to avoid spurious memorization and data leakage.",
                            auto_repairable=True,
                            category="leakage",
                            title="Entity Identifier Used as Predictor",
                            description=f"Unique identifier '{col}' has {n_unique} distinct values. Identifiers cause memorization and fail to generalize.",
                            status="confirmed",
                            evidence=f"Column '{col}' has {n_unique}/{len(series)} unique entries ({n_unique/len(series):.1%}) with action='{action}'",
                            recommended_repair=f"Exclude '{col}' from cleaning policies and feature inputs.",
                        ))

                    # Check for target-derived columns (e.g. churn_date, churn_reason, cancel_status)
                    if target_column:
                        tgt_clean = target_column.lower().replace("_", "").replace("-", "")
                        if tgt_clean in col_lower and col_lower != tgt_clean:
                            findings.append(LeakageFinding(
                                rule_id="LEAK_TARGET_DERIVED_COLUMN",
                                severity="high",
                                message=f"Column {col!r} name suggests it is derived from or occurs after target {target_column!r}.",
                                line=None,
                                recommendation=f"Verify data collection timing for {col!r}; exclude if measured after outcome or derived from target.",
                                category="leakage",
                                title="Potential Target-Derived Feature",
                                description=f"Feature '{col}' contains target name '{target_column}' and may be a post-outcome field or proxy variable.",
                                status="potential",
                                evidence=f"Target: '{target_column}'; Feature: '{col}'",
                                recommended_repair=f"Review feature provenance and exclude if recorded post-outcome.",
                            ))

        # ------------------------------------------------------------
        # 3. Data Split Disjointness & Overlap Check
        # ------------------------------------------------------------
        if splits:
            train_idx = set(splits.get("train", []))
            val_idx = set(splits.get("validation", []))
            test_idx = set(splits.get("test", []))

            train_val_overlap = len(train_idx & val_idx)
            train_test_overlap = len(train_idx & test_idx)
            val_test_overlap = len(val_idx & test_idx)

            if train_val_overlap > 0 or train_test_overlap > 0 or val_test_overlap > 0:
                findings.append(LeakageFinding(
                    rule_id="LEAK_SPLIT_OVERLAP",
                    severity="critical",
                    message=(
                        f"Data partition overlap detected: "
                        f"train-val overlap={train_val_overlap}, "
                        f"train-test overlap={train_test_overlap}, "
                        f"val-test overlap={val_test_overlap}."
                    ),
                    line=None,
                    recommendation="Re-split dataset using strictly disjoint index partitioning.",
                    category="leakage",
                    title="Overlapping Rows Across Dataset Splits",
                    description="Training, validation, and test splits share row indices, allowing the model to train on its own evaluation data.",
                    status="confirmed",
                    evidence=f"Overlaps: train-val: {train_val_overlap}, train-test: {train_test_overlap}, val-test: {val_test_overlap}",
                    recommended_repair="Ensure index sets are mutually exclusive.",
                ))
            else:
                findings.append(LeakageFinding(
                    rule_id="INFO_SPLITS_DISJOINT",
                    severity="informational",
                    message="Dataset partitions (train, validation, test) are verified strictly disjoint.",
                    line=None,
                    recommendation="Maintain split seed reproducibility across re-runs.",
                    category="leakage",
                    title="Dataset Splits Disjoint",
                    description=f"Verified {len(train_idx)} train, {len(val_idx)} validation, and {len(test_idx)} test rows are mutually disjoint.",
                    status="confirmed",
                    evidence=f"Row counts: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}; 0 shared indices.",
                ))

            # Check for cross-split duplicate feature records
            if frame is not None and train_idx and (val_idx or test_idx):
                feature_cols = [c for c in frame.columns if c != target_column]
                if feature_cols:
                    train_rows = frame.iloc[list(train_idx)][feature_cols]
                    eval_idx = list(val_idx | test_idx)
                    eval_rows = frame.iloc[eval_idx][feature_cols]

                    # Fast duplicate check by row hashes
                    train_hashes = pd.util.hash_pandas_object(train_rows, index=False)
                    eval_hashes = pd.util.hash_pandas_object(eval_rows, index=False)
                    duplicate_hashes = set(train_hashes) & set(eval_hashes)

                    if duplicate_hashes:
                        findings.append(LeakageFinding(
                            rule_id="LEAK_DUPLICATE_ROWS_ACROSS_SPLITS",
                            severity="medium",
                            message=f"{len(duplicate_hashes)} identical feature records appear in both training and evaluation partitions.",
                            line=None,
                            recommendation="Deduplicate dataset before splitting or apply group-based splitting on entity keys.",
                            category="leakage",
                            title="Cross-Split Duplicate Feature Records",
                            description="Identical rows present in both training and test sets allow memorization of held-out records.",
                            status="potential",
                            evidence=f"{len(duplicate_hashes)} hash collisions found between train and eval partitions.",
                            recommended_repair="Inspect duplicates and deduplicate raw data prior to train-test partition.",
                        ))

        # ------------------------------------------------------------
        # 4. Held-Out Test Split Contamination Check
        # ------------------------------------------------------------
        if training_report:
            test_eval = training_report.get("test_evaluated", False)
            if test_eval is True:
                findings.append(LeakageFinding(
                    rule_id="LEAK_TEST_CONTAMINATION",
                    severity="critical",
                    message="The test partition was evaluated during training/model selection.",
                    line=None,
                    recommendation="Reserve test partition strictly for independent audit after all training and model choices are fixed.",
                    category="leakage",
                    title="Test Partition Used During Training",
                    description="Evaluating on test data during training causes selection bias and optimistic performance estimates.",
                    status="confirmed",
                    evidence=f"training_report.test_evaluated={test_eval}",
                    recommended_repair="Evaluate candidate models only on the validation split; evaluate test split only in testing agent.",
                ))
            else:
                findings.append(LeakageFinding(
                    rule_id="INFO_TEST_ISOLATION",
                    severity="informational",
                    message="Held-out test partition remained strictly unvisited during model training and selection.",
                    line=None,
                    recommendation="Proceed to independent testing on held-out test split.",
                    category="leakage",
                    title="Test Partition Properly Isolated",
                    description="Training report confirms candidate model selection was performed exclusively on validation data.",
                    status="confirmed",
                    evidence="training_report.test_evaluated is False",
                ))

        # ------------------------------------------------------------
        # 5. Suspiciously High Validation Performance Check
        # ------------------------------------------------------------
        if training_report:
            score = training_report.get("selected_validation_score")
            metric = training_report.get("primary_metric", "metric")
            if score is not None and isinstance(score, (int, float)):
                if score >= 0.999:
                    findings.append(LeakageFinding(
                        rule_id="LEAK_PERFECT_PERFORMANCE",
                        severity="high",
                        message=f"Model achieved near-perfect validation {metric} score ({score:.4f}). This strongly indicates label leakage or trivial synthetic data.",
                        line=None,
                        recommendation="Audit top contributing feature importances to verify features are genuinely available at prediction time.",
                        category="leakage",
                        title="Suspiciously Near-Perfect Performance",
                        description=f"Validation score of {score:.4f} is unusually high and frequently signifies target proxy leakage.",
                        status="potential",
                        evidence=f"selected_validation_score={score} for metric={metric}",
                        recommended_repair="Inspect feature correlations with target and exclude post-event predictors.",
                    ))
                else:
                    findings.append(LeakageFinding(
                        rule_id="INFO_MODEL_PERFORMANCE",
                        severity="informational",
                        message=f"Model validation score ({score:.4f} {metric}) is within standard plausible distribution bounds.",
                        line=None,
                        recommendation="Validate generalization on held-out test split.",
                        category="leakage",
                        title="Validation Score Plausible",
                        description=f"Validation {metric} is {score:.4f}, without immediate indicators of degenerate leakage.",
                        status="confirmed",
                        evidence=f"selected_validation_score={score}",
                    ))

        # ------------------------------------------------------------
        # 6. Source Code Refactoring and Maintainability Audit
        # ------------------------------------------------------------
        if source_files:
            for filepath, content in source_files.items():
                if filepath.endswith(".py"):
                    code_result = audit_code(content, source_file=filepath)
                    for f in code_result.findings:
                        # Adopt source file into finding
                        findings.append(LeakageFinding(
                            rule_id=f.rule_id,
                            severity=f.severity,
                            message=f"{filepath}:{f.line or 1}: {f.message}",
                            line=f.line,
                            recommendation=f.recommendation,
                            auto_repairable=f.auto_repairable,
                            category=f.category,
                            title=f.title,
                            description=f.description,
                            status=f.status,
                            source_file=filepath,
                            evidence=f.evidence or f"Source: {filepath}:{f.line}",
                            recommended_repair=f.recommended_repair,
                        ))

        # Compute counts and overall status
        counts = {"critical": 0, "high": 0, "medium": 0, "low": 0, "informational": 0}
        for f in findings:
            sev = f.severity.lower()
            counts[sev] = counts.get(sev, 0) + 1

        overall = "passed"
        if counts["critical"] > 0:
            overall = "failed"
        elif counts["high"] > 0:
            overall = "warning"
        elif counts["medium"] > 0:
            overall = "needs_review"

        summary_text = (
            f"Audit completed: {counts['critical']} critical, {counts['high']} high, "
            f"{counts['medium']} medium, {counts['low']} low, {counts['informational']} informational findings."
        )

        return AuditResult(
            findings=findings,
            syntax_valid=True,
            repaired_code=None,
            repairs_applied=[],
            overall_status=overall,
            severity_counts=counts,
            summary=summary_text,
        )
