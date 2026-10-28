# AutoAudit

## Overview

AutoAudit is a Python-based project that includes data engineering, leakage auditing, model training, workflow orchestration, and source verification components.

This project also includes a static `TestingAgent` that checks Python syntax and identifies functions, asynchronous functions, and classes without executing submitted source code.

## Technology Stack

- Python
- FastAPI
- Pytest
- LangGraph
- NumPy and Pandas
- Scikit-learn
- LangChain integrations

Refer to `requirements.txt` for the project's declared dependencies.

## Project Structure

```text
AutoAudit-main/
├── app/
│   ├── agents/
│   │   ├── audit_and_test.py
│   │   ├── leakage_refactoring_agent.py
│   │   ├── testing_agent.py
│   │   ├── data_engineering_agent.py
│   │   ├── model_training_agent.py
│   │   └── planner_agent.py
│   ├── api/
│   ├── core/
│   ├── graph/
│   ├── llm/
│   ├── ml/
│   ├── schemas/
│   ├── services/
│   └── main.py
├── docs/
│   ├── architecture.md
│   └── integration_contract.md
├── tests/
│   └── test_testing_agent.py
├── requirements.txt
└── README.md
```

## Environment Setup

Use the Python version specified by the project requirements.

1. Create and activate a virtual environment.
2. Install the dependencies.
3. Run the automated tests.

Example commands for Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If the virtual environment already exists, activate it rather than creating it again.

## Verification

Run the complete automated test suite:

```powershell
python -m pytest -v
```

Run only the TestingAgent tests:

```powershell
python -m pytest -v .\tests\test_testing_agent.py
```

### Verification results

The current recorded test run completed with **65 passed** and no test failures.

The four TestingAgent tests cover:

- Valid Python source.
- Invalid Python syntax.
- Non-string input.
- Function-name detection.

Test results should be recorded again whenever relevant source code changes.

## TestingAgent Behavior

The `TestingAgent` returns a structured verification result containing:

- Verification status.
- Python syntax validity.
- Whether submitted code was executed.
- Static checks.
- Detected function and class names.
- Error messages and a summary.

The `audit_then_test()` function first audits the source and then performs static verification on the repaired source when available, or on the original source otherwise.

## Current Limitations

- The TestingAgent does not execute submitted Python code.
- Valid syntax does not guarantee correct runtime behavior.
- Runtime tests, integration behavior, and security properties require separate verification.
- A successful test suite does not prove that all data leakage risks have been eliminated.
- Audit findings must be reviewed before considering code ready for use.

## Contribution and Maintenance

Before submitting changes:

1. Run the full test suite.
2. Review failures and warnings.
3. Update documentation when behavior changes.
4. Record known limitations and outstanding verification work.
