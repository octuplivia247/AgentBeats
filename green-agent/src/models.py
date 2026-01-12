"""Data models for MCP server."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class ToolCallRequest(BaseModel):
    """Request to call an MCP tool."""
    name: str
    arguments: dict[str, Any] = {}


class ToolCallResponse(BaseModel):
    """Response from an MCP tool call."""
    result: Any
    error: Optional[str] = None


@dataclass
class ActionLog:
    """Log entry for a device action."""
    timestamp: str
    device_name: str
    action: str
    parameters: dict[str, Any]
    success: bool
    error: Optional[str] = None

    @classmethod
    def create(
        cls,
        device_name: str,
        action: str,
        parameters: dict[str, Any],
        success: bool,
        error: Optional[str] = None,
    ) -> "ActionLog":
        """Create an action log with current timestamp."""
        return cls(
            timestamp=datetime.now().isoformat(),
            device_name=device_name,
            action=action,
            parameters=parameters,
            success=success,
            error=error,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "device_name": self.device_name,
            "action": self.action,
            "parameters": self.parameters,
            "success": self.success,
            "error": self.error,
        }
