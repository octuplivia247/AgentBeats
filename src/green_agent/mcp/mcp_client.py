"""
MCP Client for Green Agent

This client connects to the MCP server and provides methods to call the tools.
"""

from typing import Any, Dict, List

import httpx


class MCPClient:
    """Client for interacting with the MCP server."""

    def __init__(self, base_url: str = "http://localhost:9006"):
        """
        Initialize the MCP client.

        Args:
            base_url: Base URL of the MCP server
        """
        self.base_url = base_url.rstrip("/")
        self.client = httpx.AsyncClient(timeout=120.0)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call an MCP tool.

        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool

        Returns:
            The result from the tool
        """
        url = f"{self.base_url}/tools/call"
        payload = {"name": tool_name, "arguments": arguments}

        try:
            response = await self.client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result
        except httpx.HTTPError as e:
            print(f"Error calling MCP tool {tool_name}: {e}")
            raise

    async def run_homebench_evaluation(
        self, purple_agent_url: str, evaluation_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute complete HomeBench evaluation for a purple agent.

        Args:
            purple_agent_url: URL of the purple agent to evaluate
            evaluation_config: Configuration for the evaluation

        Returns:
            Dictionary containing evaluation results
        """
        return await self.call_tool(
            "run_homebench_evaluation",
            {"purple_agent_url": purple_agent_url, "evaluation_config": evaluation_config},
        )

    async def initialize_smart_home(self, environment_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set up the smart home simulation with all devices and initial states.

        Args:
            environment_config: Configuration for the smart home environment

        Returns:
            Status of initialization
        """
        return await self.call_tool(
            "initialize_smart_home", {"environment_config": environment_config}
        )

    async def load_home_by_id(
        self, home_id: int, jsonl_path: str = "data/home_status_method_all.jsonl"
    ) -> Dict[str, Any]:
        """
        Load a specific home environment by its ID from a multi-home JSONL file.

        Args:
            home_id: The home ID to load
            jsonl_path: Path to the JSONL file containing home configurations

        Returns:
            Status of initialization with device information
        """
        return await self.call_tool(
            "load_home_by_id", {"home_id": home_id, "jsonl_path": jsonl_path}
        )

    async def list_available_homes(
        self, jsonl_path: str = "data/home_status_method_all.jsonl"
    ) -> Dict[str, Any]:
        """
        List all available home IDs from a multi-home JSONL file.

        Args:
            jsonl_path: Path to the JSONL file containing home configurations

        Returns:
            List of available home IDs
        """
        return await self.call_tool("list_available_homes", {"jsonl_path": jsonl_path})

    async def assign_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Send a HomeBench task instruction to the purple agent.

        Args:
            task: The task definition containing instruction and expected actions

        Returns:
            The response from the purple agent
        """
        return await self.call_tool("assign_task", {"task": task})

    async def monitor_purple_agent_actions(
        self, monitoring_config: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Track and validate actions taken by the purple agent.

        Args:
            monitoring_config: Configuration for monitoring

        Returns:
            List of actions taken by the agent
        """
        return await self.call_tool(
            "monitor_purple_agent_actions", {"monitoring_config": monitoring_config}
        )

    async def evaluate_task_completion(self, evaluation_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess whether the purple agent successfully completed the task.

        Args:
            evaluation_input: Dictionary containing task, agent actions, and final device states

        Returns:
            Evaluation result including success status and score
        """
        return await self.call_tool(
            "evaluate_task_completion", {"evaluation_input": evaluation_input}
        )

    async def compute_accuracy_metrics(
        self, predictions: List[Any], expected: List[Any]
    ) -> Dict[str, float]:
        """
        Compute evaluation metrics for a set of predicted vs expected operations.

        Args:
            predictions: List of predicted operations
            expected: List of expected operations

        Returns:
            Dictionary with EM, Precision, Recall, F1 scores
        """
        return await self.call_tool(
            "compute_accuracy_metrics", {"predictions": predictions, "expected": expected}
        )

    async def categorize_examples(self, dataset_path: str) -> Dict[str, int]:
        """
        Load dataset entries and group them into evaluation categories.

        Args:
            dataset_path: Path to the dataset file

        Returns:
            Dictionary with counts per category
        """
        return await self.call_tool("categorize_examples", {"dataset_path": dataset_path})

    async def evaluate_all_categories(self, results_path: str) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics for each instruction category.

        Args:
            results_path: Path to the results file

        Returns:
            Dictionary mapping categories to their metrics
        """
        return await self.call_tool("evaluate_all_categories", {"results_path": results_path})

    async def write_error_logs(
        self, predictions: List[Any], expected: List[Any], output_dir: str
    ) -> str:
        """
        Generate JSONL files containing mismatched predictions.

        Args:
            predictions: List of predicted operations
            expected: List of expected operations
            output_dir: Directory to save error logs

        Returns:
            Path to the generated error logs
        """
        return await self.call_tool(
            "write_error_logs",
            {"predictions": predictions, "expected": expected, "output_dir": output_dir},
        )
