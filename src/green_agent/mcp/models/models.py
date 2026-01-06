from typing import Any, Dict, Optional

from pydantic import BaseModel


class ToolCallRequest(BaseModel):
    name: str
    arguments: Dict[str, Any]


class ToolCallResponse(BaseModel):
    result: Any
    error: Optional[str] = None
