# AutoAudit Architecture

## 1. Purpose

AutoAudit is organized into components for data engineering, leakage auditing, model training, workflow orchestration, and verification.

This document describes the major source-code modules and the relationship between the leakage audit and static verification stages.

## 2. Main Components

### Application Entry Point

`app/main.py`

Contains the application's main entry point.

### Agent Layer

Located in `app/agents/`.

- `planner_agent.py`: planning-related agent logic.
- `data_engineering_agent.py`: data engineering agent logic.
- `leakage_refactoring_agent.py`: leakage auditing and refactoring support.
- `model_training_agent.py`: model training agent logic.
- `testing_agent.py`: static Python source verification.
- `audit_and_test.py`: coordinates the leakage audit and subsequent static verification.

### API Layer

Located in `app/api/`.

Contains route modules for application operations, including datasets, project planning, and runs.

### Core Configuration

Located in `app/core/`.

Contains configuration and logging modules.

### Workflow Layer

Located in `app/graph/`.

Contains workflow orchestration and related graph modules.

### Machine Learning Layer

Located in `app/ml/`.

Contains modules for preprocessing, profiling, splitting, training, and model creation.

### Schemas and Services

- `app/schemas/`: request, response, and data-model definitions.
- `app/services/`: application service logic.
- `app/storage/`: database and repository modules.
- `app/llm/`: language-model provider integrations.

## 3. Audit and Verification Flow

The function `audit_then_test()` in `app/agents/audit_and_test.py` performs these steps:

1. Receives Python source code.
2. Runs `LeakageRefactoringAgent` with `apply_repairs=True`.
3. Selects the repaired source when available; otherwise, it selects the original source.
4. Passes the selected source to `TestingAgent`.
5. Returns audit details, testing details, an overall status, and a review note.

## 4. TestingAgent

The TestingAgent uses Python's Abstract Syntax Tree (`ast`) parser to validate syntax and identify function, asynchronous function, and class definitions.

It returns a structured result with syntax validity, execution status, detected targets, checks, errors, and a summary.

The agent does not execute submitted source code. If runtime execution is requested, it reports that runtime execution is not implemented.

## 5. Verification Status

A recorded automated test run completed with 65 passing tests. Four of those tests directly cover TestingAgent behavior.

Passing tests indicate that the tested cases succeeded; they do not establish complete application correctness or production readiness.

## 6. Known Limitations

- Static syntax validation does not establish runtime correctness.
- Detected functions and classes are not equivalent to executed test cases.
- Audit findings still require review.
- Runtime and integration testing must be conducted separately where applicable.
