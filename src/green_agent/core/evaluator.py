"""
Green Agent Evaluator - HomeBench Evaluation Framework

This module provides the core evaluation infrastructure for assessing purple agents
against HomeBench tasks.
"""

import asyncio
import json
import time
from typing import Any, Dict, List, Optional

from src.green_agent.core.metrics_calculator import MetricsCalculator
from src.green_agent.core.models.task_result import TaskResult
from src.green_agent.core.smart_home_env_manager import SmartHomeEnvironment
from src.utils import my_a2a


class AgentCommunicator:
    """
    Handles communication with purple agents via A2A protocol.

    This class:
    - Sends task instructions to agents via A2A
    - Maintains conversation context
    - Formats messages with tool/device information
    - Parses A2A responses
    """

    def __init__(self, agent_url: str):
        """Initialize communicator with agent URL."""
        self.agent_url = agent_url
        self.context_id: Optional[str] = None
        self.task_id: Optional[str] = None

    async def wait_ready(self, timeout: int = 30) -> bool:
        """
        Wait for the agent to be ready.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            True if agent is ready, False otherwise
        """
        return await my_a2a.wait_agent_ready(self.agent_url, timeout=timeout)

    async def send_task(
        self,
        task_instruction: str,
        tools_info: Optional[List[Dict]] = None,
        environment_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Send task instruction to agent.

        Args:
            task_instruction: Natural language instruction
            tools_info: Optional list of available tools/devices
            environment_id: Optional environment identifier

        Returns:
            {"status": "success", "response": "agent's text response"}
            or {"status": "error", "error": "error message"}
        """
        # Format message with tools info
        message = task_instruction

        if tools_info:
            message += f"\n\n<tools>\n{json.dumps(tools_info, indent=2)}\n</tools>"

        if environment_id:
            message += f"\n\n<environment_id>\n{environment_id}\n</environment_id>"

        message += "\n\nPlease use the available MCP tools to complete this task."

        try:
            response = await my_a2a.send_message(
                self.agent_url,
                message,
                task_id=self.task_id,
                context_id=self.context_id,
            )

            # Extract response text
            if hasattr(response, "result") and response.result:
                result = response.result
                if hasattr(result, "parts") and result.parts:
                    response_text = ""
                    for part in result.parts:
                        if hasattr(part, "root") and hasattr(part.root, "text"):
                            response_text += part.root.text
                        elif hasattr(part, "text"):
                            response_text += part.text
                    return {"status": "success", "response": response_text}

            return {"status": "success", "response": str(response)}

        except Exception as e:
            return {"status": "error", "error": str(e)}

    def parse_operations_from_response(self, response: str) -> List[str]:
        """
        Parse device operations from agent response.

        Looks for patterns like:
        - <tool_call>{"name": "control_device", "arguments": {...}}</tool_call>
        - room_name.device_name.operation(args)

        Args:
            response: Agent's text response

        Returns:
            List of operation strings in format "device.operation(args)"
        """
        import re

        operations = []

        # Parse <tool_call> format
        tool_call_pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(tool_call_pattern, response, re.DOTALL)

        for match in matches:
            try:
                call = json.loads(match.strip())
                if call.get("name") == "control_device":
                    args = call.get("arguments", {})
                    device = args.get("device_name", "unknown")
                    action = args.get("action", "unknown")
                    params = args.get("parameters", {})

                    if params:
                        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
                        operations.append(f"{device}.{action}({param_str})")
                    else:
                        operations.append(f"{device}.{action}()")
            except json.JSONDecodeError:
                continue

        # Parse direct operation format: room.device.operation(args)
        op_pattern = r"(\w+)\.(\w+)\.(\w+)\(([^)]*)\)"
        for match in re.finditer(op_pattern, response):
            room, device, operation, args = match.groups()
            if args:
                operations.append(f"{room}.{device}.{operation}({args})")
            else:
                operations.append(f"{room}.{device}.{operation}()")

        return operations


class HomeBenchEvaluator:
    """
    Main orchestrator for HomeBench task evaluation.

    This class coordinates the entire evaluation process:
    1. Manages a SmartHomeEnvironment
    2. Communicates with purple agents via A2A protocol
    3. Parses agent responses to extract operations
    4. Executes operations in the environment
    5. Computes evaluation metrics
    6. Stores and returns results

    Usage:
        evaluator = HomeBenchEvaluator("http://localhost:9000")
        await evaluator.initialize_environment(config)
        result = await evaluator.evaluate_task(task)
        all_results = evaluator.get_results()
        metrics = evaluator.get_aggregate_metrics()
    """

    def __init__(self, purple_agent_url: str):
        """
        Initialize the evaluator.

        Args:
            purple_agent_url: URL of the purple agent to evaluate
        """
        self.purple_agent_url = purple_agent_url
        self.environment = SmartHomeEnvironment()
        self.communicator = AgentCommunicator(purple_agent_url)
        self.metrics_calc = MetricsCalculator()
        self.results: List[TaskResult] = []

    async def initialize_environment(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Initialize the smart home environment.

        Args:
            config: Environment configuration

        Returns:
            Initialization result
        """
        return self.environment.initialize(config)

    async def evaluate_task(
        self, task: Dict[str, Any], max_steps: int = 30, wait_time: float = 2.0
    ) -> TaskResult:
        """
        Evaluate a single task.

        This is the main evaluation flow:
        1. Reset the environment
        2. Get available tools/devices info
        3. Send task instruction to purple agent via A2A
        4. Parse agent's response to extract operations
        5. Execute each operation in the environment
        6. Compare predicted vs expected operations
        7. Compute metrics and create TaskResult
        8. Store result and return

        Args:
            task: Task dictionary containing:
                - task_id: Unique task identifier
                - instruction: Natural language instruction
                - expected_operations: List of expected operations
                - category: Optional task category
            max_steps: Maximum interaction steps allowed
            wait_time: Time to wait for agent response

        Returns:
            TaskResult object with evaluation details
        """
        start_time = time.time()

        task_id = task.get("task_id", "unknown")
        instruction = task.get("instruction", "")
        expected_operations = task.get("expected_operations", [])
        category = task.get("category", "normal_single")
        environment_id = task.get("environment_id")

        # Get tools info
        tools_info = self.environment.get_tools_info()

        # Wait for agent to be ready
        if not await self.communicator.wait_ready(timeout=10):
            result = TaskResult(
                task_id=task_id,
                instruction=instruction,
                expected_operations=expected_operations,
                predicted_operations=[],
                actions_taken=[],
                success=False,
                score=0.0,
                feedback="Agent not ready",
                category=category,
                execution_time=time.time() - start_time,
            )
            self.results.append(result)
            return result

        # Send task to agent
        response = await self.communicator.send_task(
            instruction, tools_info, environment_id
        )

        if response["status"] == "error":
            result = TaskResult(
                task_id=task_id,
                instruction=instruction,
                expected_operations=expected_operations,
                predicted_operations=[],
                actions_taken=[],
                success=False,
                score=0.0,
                feedback=f"Communication error: {response.get('error', 'Unknown')}",
                category=category,
                execution_time=time.time() - start_time,
            )
            self.results.append(result)
            return result

        # Give agent time to execute via MCP
        await asyncio.sleep(wait_time)

        # Get actions from environment log
        action_log = self.environment.get_action_log()

        # Extract predicted operations from actions
        predicted_operations = []
        for action in action_log:
            if action.success:
                if action.parameters:
                    param_str = ", ".join(
                        f"{k}={v}" for k, v in action.parameters.items()
                    )
                    predicted_operations.append(
                        f"{action.device_name}.{action.action}({param_str})"
                    )
                else:
                    predicted_operations.append(
                        f"{action.device_name}.{action.action}()"
                    )

        # Also parse operations from response (backup)
        if not predicted_operations and response.get("response"):
            predicted_operations = self.communicator.parse_operations_from_response(
                response["response"]
            )

        # Compute metrics
        metrics = self.metrics_calc.compute_metrics(
            predicted_operations, expected_operations
        )

        success = metrics["exact_match"] == 1.0
        score = metrics["f1"]

        # Generate feedback
        if success:
            feedback = "Task completed successfully"
        else:
            feedback = f"F1: {score:.2f}, EM: {metrics['exact_match']:.2f}"
            if metrics["precision"] < 1.0:
                feedback += f", Precision: {metrics['precision']:.2f}"
            if metrics["recall"] < 1.0:
                feedback += f", Recall: {metrics['recall']:.2f}"

        result = TaskResult(
            task_id=task_id,
            instruction=instruction,
            expected_operations=expected_operations,
            predicted_operations=predicted_operations,
            actions_taken=[
                {
                    "timestamp": a.timestamp,
                    "device": a.device_name,
                    "action": a.action,
                    "parameters": a.parameters,
                    "success": a.success,
                    "error": a.error,
                }
                for a in action_log
            ],
            success=success,
            score=score,
            feedback=feedback,
            category=category,
            execution_time=time.time() - start_time,
        )

        self.results.append(result)
        return result

    def get_results(self) -> List[TaskResult]:
        """
        Get all evaluation results collected so far.

        Returns:
            List of TaskResult objects
        """
        return list(self.results)

    def get_aggregate_metrics(self) -> Dict[str, float]:
        """
        Get aggregate metrics across all evaluated tasks.

        Returns:
            Dictionary with aggregate metrics
        """
        return self.metrics_calc.compute_aggregate_metrics(self.results)

    def get_results_by_category(self) -> Dict[str, List[TaskResult]]:
        """
        Group results by task category.

        Returns:
            Dictionary mapping category names to results
        """
        by_category: Dict[str, List[TaskResult]] = {}
        for result in self.results:
            if result.category not in by_category:
                by_category[result.category] = []
            by_category[result.category].append(result)
        return by_category

    def get_metrics_by_category(self) -> Dict[str, Dict[str, float]]:
        """
        Get metrics for each category.

        Returns:
            Dictionary mapping category names to their metrics
        """
        by_category = self.get_results_by_category()
        return {
            category: self.metrics_calc.compute_aggregate_metrics(results)
            for category, results in by_category.items()
        }

    def clear_results(self) -> None:
        """Clear all stored results."""
        self.results.clear()
