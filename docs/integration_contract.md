# AutoAudit Integration Contract

## 1. Purpose

This document describes the input, output, and current behavior of the audit and static verification integration.

## 2. Audit-and-Test Entry Point

Module: `app/agents/audit_and_test.py`

Function:

`audit_then_test(source: str, *, execute: bool = False) -> dict[str, Any]`

### Input

- `source`: Python source code represented as a string.
- `execute`: Optional Boolean flag. Runtime execution is not implemented by the current TestingAgent.

### Processing

1. The function passes the source to `LeakageRefactoringAgent`.
2. The audit runs with `apply_repairs=True`.
3. If repaired code is available, it becomes the verification candidate. Otherwise, the original source is used.
4. The candidate is passed to `TestingAgent`.
5. The function returns a dictionary containing the audit result, testing result, overall status, and a note about reviewing leakage findings.

## 3. TestingAgent Contract

Module: `app/agents/testing_agent.py`

Method:

`TestingAgent.run(source: str, *, execute: bool = False) -> TestingResult`

### Result Fields

- `status`: Verification status, such as `failed` or `needs_review`.
- `syntax_valid`: Indicates whether Python syntax parsing succeeded.
- `executed`: Indicates whether the submitted source was executed.
- `checks`: List of structured verification checks.
- `test_targets`: Detected function, asynchronous function, and class names.
- `errors`: List of reported errors or limitations.
- `summary`: Human-readable result summary.

The result can be converted into a dictionary using `to_dict()`.

## 4. Current Behavior

### Valid Python Source

The agent parses the source and reports a successful syntax check. The status remains `needs_review` because runtime behavior has not been verified.

### Invalid Python Source

The agent returns a failed result, marks syntax as invalid, and reports the syntax error location and message when available.

### Invalid Input Type

If the input is not a string, the agent returns a failed result with an explanatory error.

### Runtime Execution Request

Setting `execute=True` does not execute the source. The agent reports that runtime execution is not implemented and recommends using a separately isolated environment for runtime testing.

## 5. Overall Status

The `audit_then_test()` integration currently reports `passed` only when the audit's `syntax_valid` value is true and the TestingAgent status is `passed`. Since the current TestingAgent returns `needs_review` for valid source, the integration's overall status will normally remain `needs_review` for that case.

A `needs_review` status is not equivalent to a failed syntax check; it indicates that further verification is required.

## 6. Safety and Verification Boundaries

- Do not treat static parsing as runtime testing.
- Do not execute untrusted source code in the main application process.
- Use a separately isolated environment for runtime tests.
- Review leakage findings before accepting refactored code.
- Passing automated tests does not prove that all data leakage has been eliminated.

## 7. Test Coverage

The current TestingAgent unit tests cover:

- Valid Python source.
- Invalid Python syntax.
- Non-string input.
- Detection of function names.

The recorded full test suite completed with 65 passing tests. Run the suite again after changes to confirm the current result.

## 8. Future Improvements

Potential improvements include isolated runtime test execution, stronger integration tests, clearer verification status definitions, and expanded coverage of audit-to-testing handoffs.
