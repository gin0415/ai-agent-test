from typing import Any, Dict, List, Tuple, Optional, Literal

from pydantic import (
    BaseModel,
    ValidationError,
    field_validator,
    model_validator,
)


class ToolCall(BaseModel):
    action: Literal["search", "answer"]
    q: Optional[str] = None
    k: Optional[int] = 3

    @field_validator("k", mode="before")
    @classmethod
    def coerce_k(cls, v):
        if v is None:
            return 3
        if isinstance(v, bool):
            raise ValueError("k must be an integer between 1 and 5")
        if isinstance(v, str):
            v = v.strip()
            if not v.lstrip("-").isdigit():
                raise ValueError("k must be an integer between 1 and 5")
            v = int(v)
        if not isinstance(v, int) or not (1 <= v <= 5):
            raise ValueError("k must be an integer between 1 and 5")
        return v

    @field_validator("q", mode="before")
    @classmethod
    def trim_q(cls, v):
        if isinstance(v, str):
            return v.strip()
        return v

    @model_validator(mode="after")
    def check_q_conditional(self):
        if self.action == "search":
            if not isinstance(self.q, str) or not self.q.strip():
                raise ValueError("Missing or empty 'q' for action='search'")
        elif self.action == "answer":
            # 'q' is ignored for action='answer'
            self.q = None
        return self


def validate_tool_call(payload: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """
    Returns (clean, errors). 'clean' strictly follows the
    schema with defaults applied.
    - Trim strings; coerce numeric strings to ints.
    - Remove unknown keys.
    - If action=='answer', ignore 'q' if present (no error).
    - On fatal errors (e.g., missing/invalid 'action', or
    missing/empty 'q' for search), return ({}, errors).
    """
    errors: List[str] = []
    try:
        obj = ToolCall.model_validate(payload)
    except ValidationError as e:
        errors = [err["msg"] for err in e.errors()]
        return {}, errors

    clean = obj.model_dump(exclude_none=(obj.action == "answer"))
    return clean, errors


if __name__ == "__main__":
    print(validate_tool_call({
        "action": "search",
        "q": "What is the capital of France?",
        "k": "2",
    }))

    print(validate_tool_call({
        "action": "answer",
        "q": "What is the capital of France?",
        "k": "2",
    }))
