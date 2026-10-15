
import pytest
from langchain_core.messages import AIMessage
from pydantic import BaseModel, ConfigDict

from app.llm.provider_factory import (
    LLMInvocationError,
    _validate_response,
)


class SampleSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    age: int


def validate_content(content: str, schema=SampleSchema):
    return _validate_response(
        AIMessage(content=content),
        schema,
    )


def test_valid_json_is_accepted():
    result = validate_content(
        '{"name": "Amit", "age": 22}'
    )

    assert result.name == "Amit"
    assert result.age == 22


@pytest.mark.parametrize(
    "content",
    [
        "",
        "not json",
        '{"name": "Amit", "age": }',
        '```json\n{"name": "Amit", "age": 22}\n```',
        '["Amit", 22]',
    ],
)
def test_invalid_json_is_rejected(content):
    with pytest.raises(LLMInvocationError) as exc_info:
        validate_content(content)

    assert exc_info.value.code == "model_output_invalid"


def test_duplicate_json_keys_are_rejected():
    content = '{"name": "Amit", "name": "Raj", "age": 22}'

    with pytest.raises(LLMInvocationError) as exc_info:
        validate_content(content)

    assert exc_info.value.code == "model_output_invalid"


def test_extra_fields_are_rejected():
    content = (
        '{"name": "Amit", "age": 22, "admin": true}'
    )

    with pytest.raises(LLMInvocationError):
        validate_content(content)


@pytest.mark.parametrize(
    "content",
    [
        '{"name": "Amit", "age": NaN}',
        '{"name": "Amit", "age": Infinity}',
        '{"name": "Amit", "age": -Infinity}',
    ],
)
def test_non_finite_json_values_are_rejected(content):
    with pytest.raises(LLMInvocationError):
        validate_content(content)


def test_wrong_field_type_is_rejected():
    content = '{"name": "Amit", "age": "22"}'

    with pytest.raises(LLMInvocationError):
        validate_content(content)


def test_truncated_response_is_rejected():
    response = AIMessage(
        content='{"name": "Amit", "age": 22}',
        response_metadata={"finish_reason": "length"},
    )

    with pytest.raises(LLMInvocationError):
        _validate_response(response, SampleSchema)


def test_tool_call_response_is_rejected():
    response = AIMessage(
        content='{"name": "Amit", "age": 22}',
        tool_calls=[
            {
                "name": "example_tool",
                "args": {},
                "id": "call_123",
                "type": "tool_call",
            }
        ],
    )

    with pytest.raises(LLMInvocationError):
        _validate_response(response, SampleSchema)
