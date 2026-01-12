"""MCP Client for Green Agent."""

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class MCPClient:
    """Client for interacting with the MCP server."""

    def __init__(self, base_url: str = "http://localhost:9006"):
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=120.0)
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None

    async def health_check(self) -> bool:
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception:
            return False

    async def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        """Call an MCP tool."""
        url = f"{self.base_url}/tools/call"
        payload = {"name": tool_name, "arguments": arguments}

        client = await self._get_client()
        response = await client.post(url, json=payload)
        response.raise_for_status()
        result = response.json()
        
        if isinstance(result, dict) and "result" in result:
            return result["result"]
        return result

    async def compute_accuracy_metrics(self, predictions: list[str], expected: list[str]) -> dict[str, float]:
        return await self.call_tool("compute_accuracy_metrics", {"predictions": predictions, "expected": expected})

    async def parse_operations_from_response(self, response: str) -> list[str]:
        return await self.call_tool("parse_operations_from_response", {"response": response})

    async def log_device_action(
        self,
        device_name: str,
        action: str,
        parameters: dict[str, Any] | None = None,
        success: bool = True,
        error: str | None = None
    ) -> dict[str, Any]:
        return await self.call_tool("log_device_action", {
            "device_name": device_name,
            "action": action,
            "parameters": parameters or {},
            "success": success,
            "error": error
        })

    async def compute_batch_metrics(self, results: list[dict[str, Any]]) -> dict[str, Any]:
        return await self.call_tool("compute_batch_metrics", {"results": results})
