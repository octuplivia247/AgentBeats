"""
MCP Client for Purple Agent

This client connects to the MCP server to get smart home tools and execute them.
"""

from typing import Any, Dict, List, Optional

import httpx


class PurpleMCPClient:
    """Client for purple agent to interact with MCP server tools."""

    def __init__(self, base_url: str = "http://localhost:9006"):
        """
        Initialize the MCP client.

        Args:
            base_url: Base URL of the MCP server
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=120.0)
        self._tools_cache = None

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        Get list of available tools from MCP server.

        Returns:
            List of tool definitions
        """
        if self._tools_cache is not None:
            return self._tools_cache

        try:
            url = f"{self.base_url}/tools"
            response = await self.client.get(url)
            response.raise_for_status()
            data = response.json()
            self._tools_cache = data.get("tools", [])
            return self._tools_cache
        except httpx.HTTPError as e:
            print(f"Error fetching tools from MCP server: {e}")
            return []

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call an MCP tool.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            The result from the tool

        Raises:
            httpx.HTTPError: If the tool call fails
        """
        url = f"{self.base_url}/tools/call"
        payload = {"name": tool_name, "arguments": arguments}

        print(f"Purple agent calling MCP tool: {tool_name}")
        print(f"Arguments: {arguments}")

        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()

            # Extract the actual result from the response
            if isinstance(result, dict) and "result" in result:
                return result["result"]
            return result

        except httpx.HTTPError as e:
            print(f"Error calling MCP tool {tool_name}: {e}")
            if hasattr(e, "response") and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            raise

    async def get_environment_state(self, environment_id: str = "default") -> Dict[str, Any]:
        """
        Get the current state of the smart home environment.

        Args:
            environment_id: ID of the environment

        Returns:
            Environment state
        """
        try:
            url = f"{self.base_url}/state"
            response = await self.client.get(url)
            response.raise_for_status()
            state = response.json()

            # Extract environment info
            environments = state.get("environments", {})
            if environment_id in environments:
                return environments[environment_id]

            return {}
        except httpx.HTTPError as e:
            print(f"Error fetching environment state: {e}")
            return {}
