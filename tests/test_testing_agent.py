from app.agents.testing_agent import TestingAgent


def test_valid_python_source():
    result = TestingAgent().run(
        "def add(a, b):\n    return a + b"
    )
    assert result.syntax_valid is True
    assert result.executed is False
    assert result.status == "needs_review"


def test_invalid_python_source():
    result = TestingAgent().run(
        "def add(a, b)\n    return a + b"
    )
    assert result.syntax_valid is False
    assert result.status == "failed"
    assert result.executed is False


def test_non_string_source():
    result = TestingAgent().run(123)
    assert result.status == "failed"
    assert result.syntax_valid is False


def test_function_names_are_detected():
    result = TestingAgent().run(
        "def add(a, b):\n    return a + b"
    )
    assert "add" in result.test_targets
