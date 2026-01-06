#!/usr/bin/env python3
"""
Green Agent MCP Server

This server exposes the HomeBench evaluation tools via both:
1. MCP protocol (SSE transport)
2. Simple HTTP REST API for easier testing
"""

import asyncio
import json
import logging
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastmcp import FastMCP

from src.green_agent.core.categorizer import TaskCategorizer
from src.green_agent.core.evaluator import AgentCommunicator
from src.green_agent.core.metrics_calculator import MetricsCalculator
from src.green_agent.core.models.task_result import TaskResult
from src.green_agent.mcp.state import GlobalState

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


from src.green_agent.mcp.models.models import ToolCallRequest, ToolCallResponse

mcp_server = FastMCP("HomeBench Green Agent MCP", version="1.0.0")

http_app = FastAPI(title="HomeBench MCP HTTP Wrapper")


# Tool Registry
TOOLS = {}


def register_tool(func):
    """Decorator to register tools in both MCP and HTTP API."""
    TOOLS[func.__name__] = func
    return func


# Global state instance
state = GlobalState()



@mcp_server.tool
@register_tool
def control_device(
    device_name: str, action: str, parameters: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Control a smart home device.

    Args:
        device_name: Name of the device (e.g., "living_room_light")
        action: Action to perform (e.g., "turn_on", "turn_off", "set_temperature")
        parameters: Optional parameters for the action (e.g., {"temperature": 72})

    Returns:
        Result of the device control operation
    """
    logger.info(f"Controlling device: {device_name}, action: {action}, parameters: {parameters}")

    try:
        env_id = state.current_env_id or "default"
        environment = state.get_or_create_environment(env_id)

        if device_name not in environment.devices:
            logger.warning(f"Device {device_name} not found in environment {env_id}")
            return {
                "success": False,
                "error": f"Device '{device_name}' not found",
                "device": device_name,
                "action": action,
            }

        # Log the action
        from src.green_agent.mcp.state import ActionLog

        action_log = ActionLog(
            timestamp=datetime.now().isoformat(),
            device_name=device_name,
            action=action,
            parameters=parameters or {},
            success=True,
            error=None,
        )
        environment.action_log.append(action_log)

        logger.info(f"Device {device_name} {action} executed successfully")

        return {
            "success": True,
            "device": device_name,
            "action": action,
            "parameters": parameters,
            "message": f"Successfully executed {action} on {device_name}",
        }

    except NotImplementedError as e:
        logger.warning(f"Device control not fully implemented: {e}")
        return {
            "success": False,
            "device": device_name,
            "action": action,
            "error": "not_implemented",
            "message": str(e),
        }
    except Exception as e:
        logger.error(f"Error controlling device {device_name}: {e}")
        logger.debug(traceback.format_exc())
        return {"success": False, "device": device_name, "action": action, "error": str(e)}


@mcp_server.tool
@register_tool
def get_device_state(device_name: str) -> Dict[str, Any]:
    """
    Get the current state of a smart home device.

    Args:
        device_name: Name of the device

    Returns:
        Current state of the device
    """
    logger.info(f"Getting state for device: {device_name}")

    try:
        env_id = state.current_env_id or "default"
        environment = state.get_or_create_environment(env_id)

        if device_name not in environment.devices:
            return {"success": False, "error": f"Device '{device_name}' not found"}

        device = environment.devices[device_name]

        return {
            "success": True,
            "device": device_name,
            "state": {
                "type": getattr(device, "type", "unknown"),
                "status": getattr(device, "status", "unknown"),
            },
        }

    except Exception as e:
        logger.error(f"Error getting device state: {e}")
        return {"success": False, "device": device_name, "error": str(e)}


@mcp_server.tool
@register_tool
def list_devices(room: Optional[str] = None) -> Dict[str, Any]:
    """
    List all available smart home devices.

    Args:
        room: Optional room name to filter devices

    Returns:
        List of available devices
    """
    logger.info(f"Listing devices{' in room: ' + room if room else ''}")

    try:
        env_id = state.current_env_id or "default"
        environment = state.get_or_create_environment(env_id)

        devices = list(environment.devices.keys())

        if room:
            devices = [d for d in devices if d.startswith(room)]

        return {"success": True, "devices": devices, "count": len(devices)}

    except Exception as e:
        logger.error(f"Error listing devices: {e}")
        return {"success": False, "error": str(e)}


# ==============================================================================
# EVALUATION TOOLS (for Green Agent)
# ==============================================================================


@mcp_server.tool
@register_tool
def run_homebench_evaluation(
    purple_agent_url: str, evaluation_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Execute complete HomeBench evaluation for a purple agent across all task categories.

    Args:
        purple_agent_url: URL of the purple agent to evaluate
        evaluation_config: Configuration for the evaluation

    Returns:
        Dictionary containing evaluation results
    """
    print(f"Starting HomeBench evaluation for {purple_agent_url}")
    print(f"Config: {evaluation_config}")

    eval_id = f"eval_{purple_agent_url}_{hash(json.dumps(evaluation_config))}"
    evaluator = state.get_or_create_evaluator(purple_agent_url, eval_id)

    dataset_path = evaluation_config.get("dataset_path", "data/home_status_data.jsonl")

    env_config = evaluation_config.get(
        "environment_config", {"rooms": ["living_room", "bedroom", "kitchen"], "devices": {}}
    )

    evaluator.environment.initialize(env_config)

    return {
        "status": "started",
        "eval_id": eval_id,
        "agent_url": purple_agent_url,
        "message": "Evaluation started.",
    }


@mcp_server.tool
@register_tool
def initialize_smart_home(environment_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Set up the smart home simulation with all devices and initial states.

    Args:
        environment_config: Configuration for the smart home environment

    Returns:
        Status of initialization
    """
    try:
        logger.info("Initializing smart home environment...")
        logger.info(f"Rooms: {environment_config.get('rooms', [])}")
        logger.info(f"Devices: {len(environment_config.get('devices', {}))} devices configured")

        env_id = environment_config.get("environment_id", "default")
        environment = state.get_or_create_environment(env_id)

        # Create simple mock devices directly in the environment
        # This is a basic implementation until SmartHomeEnvironment.initialize() is fully implemented
        devices_config = environment_config.get("devices", {})

        # Simple device class
        class MockDevice:
            def __init__(self, name, device_type, config):
                self.name = name
                self.type = device_type
                self.config = config
                self.status = config.get("state", "unknown")

        # Create devices
        for device_name, device_config in devices_config.items():
            device_type = device_config.get("type", "unknown")
            environment.devices[device_name] = MockDevice(device_name, device_type, device_config)
            logger.info(f"Created device: {device_name} (type: {device_type})")

        environment.initialized = True
        state.current_env_id = env_id

        device_count = len(environment.devices)
        logger.info(f"Environment {env_id} initialized with {device_count} devices")

        return {
            "status": "success",
            "device_count": device_count,
            "devices": list(environment.devices.keys()),
            "environment_id": env_id,
            "initialized": True,
        }
    except NotImplementedError as e:
        logger.warning(f"Method not implemented: {e}")
        return {
            "status": "not_implemented",
            "error": str(e),
            "message": "This method needs to be implemented by the team. See src/green_agent/evaluator.py",
            "environment_id": environment_config.get("environment_id", "default"),
        }
    except Exception as e:
        logger.error(f"Error initializing environment: {e}")
        logger.debug(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "environment_id": environment_config.get("environment_id", "default"),
        }


@mcp_server.tool
@register_tool
def assign_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """
    Send a HomeBench task instruction to the purple agent.

    Args:
        task: The task definition containing instruction and expected actions

    Returns:
        The response from the purple agent
    """
    print(f"Assigning task: {task.get('task_id')}")
    print(f"Instruction: {task.get('instruction')}")

    agent_url = task.get("agent_url", "http://localhost:9000")

    communicator = AgentCommunicator(agent_url)

    env = state.get_or_create_environment(state.current_env_id or "default")
    tools_info = [
        {
            "device": device_name,
            "operations": list(device.operations.keys()) if hasattr(device, "operations") else [],
        }
        for device_name, device in env.devices.items()
    ]

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():

        return {
            "task_id": task.get("task_id"),
            "status": "queued",
            "message": "Task queued for async execution",
        }
    else:
        response = loop.run_until_complete(
            communicator.send_task(task.get("instruction", ""), tools_info)
        )

        return {
            "task_id": task.get("task_id"),
            "status": response.get("status", "unknown"),
            "agent_response": response.get("response", ""),
            "error": response.get("error"),
        }


@mcp_server.tool
@register_tool
def monitor_purple_agent_actions(monitoring_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Track and validate actions taken by the purple agent on smart home devices.

    Args:
        monitoring_config: Configuration for monitoring (timeout, allowed devices, etc.)

    Returns:
        List of actions taken by the agent
    """
    try:
        logger.info(f"Monitoring actions for task: {monitoring_config.get('task_id')}")

        env_id = monitoring_config.get("environment_id", state.current_env_id or "default")
        environment = state.get_or_create_environment(env_id)

        actions = environment.get_action_log()

        logger.info(f"Retrieved {len(actions)} actions from environment {env_id}")

        return [
            {
                "timestamp": action.timestamp,
                "device": action.device_name,
                "action": action.action,
                "parameters": action.parameters,
                "success": action.success,
                "error": action.error,
            }
            for action in actions
        ]
    except NotImplementedError as e:
        logger.warning(f"Method not implemented: {e}")
        return [
            {
                "error": "not_implemented",
                "message": str(e),
                "note": "See src/green_agent/evaluator.py to implement this method",
            }
        ]
    except Exception as e:
        logger.error(f"Error monitoring actions: {e}")
        logger.debug(traceback.format_exc())
        return []


@mcp_server.tool
@register_tool
def evaluate_task_completion(evaluation_input: Dict[str, Any]) -> Dict[str, Any]:
    """
    Assess whether the purple agent successfully completed the task based on device states and instruction validity.

    Args:
        evaluation_input: Dictionary containing task, agent actions, and final device states

    Returns:
        Evaluation result including success status and score
    """
    # Handle case where evaluation_input might be a string (parse it)
    if isinstance(evaluation_input, str):
        try:
            evaluation_input = json.loads(evaluation_input)
        except json.JSONDecodeError:
            logger.error(f"Failed to parse evaluation_input as JSON: {evaluation_input}")
            return {
                "task_id": "unknown",
                "success": False,
                "score": 0.0,
                "feedback": "Error: Invalid evaluation_input format",
                "error": "evaluation_input must be a dictionary or valid JSON string",
            }

    task = evaluation_input.get("task", {})
    task_id = task.get("task_id", "unknown") if isinstance(task, dict) else "unknown"
    print(f"Evaluating completion for task: {task_id}")

    agent_url = evaluation_input.get("agent_url", "http://localhost:9000")
    eval_id = evaluation_input.get("eval_id", "default")

    evaluator = state.get_or_create_evaluator(agent_url, eval_id)

    env_id = evaluation_input.get("environment_id", state.current_env_id or "default")
    evaluator.environment = state.get_or_create_environment(env_id)

    # Run async evaluation
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    if loop.is_running():

        expected_ops = task.get("expected_operations", []) if isinstance(task, dict) else []
        actions = evaluation_input.get("actions", [])

        # Handle actions that might be dicts with error info
        predicted_ops = []
        for a in actions:
            if isinstance(a, dict) and a.get("success"):
                device = a.get("device", "unknown")
                action = a.get("action", "unknown")
                predicted_ops.append(f"{device}.{action}()")

        # Simple fallback metrics if MetricsCalculator not implemented
        try:
            metrics = MetricsCalculator.compute_metrics(predicted_ops, expected_ops)
        except NotImplementedError:
            # Simple exact match fallback
            metrics = {
                "exact_match": 1.0 if set(predicted_ops) == set(expected_ops) else 0.0,
                "precision": (
                    1.0 if predicted_ops and set(predicted_ops).issubset(set(expected_ops)) else 0.0
                ),
                "recall": (
                    1.0 if expected_ops and set(expected_ops).issubset(set(predicted_ops)) else 0.0
                ),
                "f1": 1.0 if set(predicted_ops) == set(expected_ops) else 0.0,
            }

        return {
            "task_id": task_id,
            "success": metrics["exact_match"] == 1.0,
            "score": metrics["f1"],
            "feedback": f"F1: {metrics['f1']:.2f}, EM: {metrics['exact_match']:.2f}",
            "metrics": metrics,
            "predicted_operations": predicted_ops,
            "expected_operations": expected_ops,
        }
    else:
        result = loop.run_until_complete(evaluator.evaluate_task(task))

        if eval_id not in state.task_results:
            state.task_results[eval_id] = []
        state.task_results[eval_id].append(result)

        return {
            "task_id": result.task_id,
            "success": result.success,
            "score": result.score,
            "feedback": result.feedback,
            "predicted_operations": result.predicted_operations,
            "expected_operations": result.expected_operations,
            "execution_time": result.execution_time,
        }


@mcp_server.tool
@register_tool
def compute_accuracy_metrics(predictions: List[Any], expected: List[Any]) -> Dict[str, float]:
    """
    Compute evaluation metrics for a set of predicted vs expected operations.

    Args:
        predictions: List of predicted operations
        expected: List of expected operations

    Returns:
        Dictionary with EM, Precision, Recall, F1 scores
    """
    try:
        logger.info("Computing accuracy metrics...")
        logger.info(f"Predictions: {len(predictions)} operations")
        logger.info(f"Expected: {len(expected)} operations")

        # Convert to strings if needed
        pred_strs = [str(p) for p in predictions]
        exp_strs = [str(e) for e in expected]

        metrics = MetricsCalculator.compute_metrics(pred_strs, exp_strs)

        logger.info(f"Results: EM={metrics['exact_match']:.2f}, F1={metrics['f1']:.2f}")

        return metrics
    except NotImplementedError as e:
        logger.warning(f"Method not implemented: {e}")
        return {
            "exact_match": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "error": "not_implemented",
            "message": str(e),
            "note": "See src/green_agent/evaluator.py to implement MetricsCalculator",
        }
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        logger.debug(traceback.format_exc())
        return {"exact_match": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "error": str(e)}


@mcp_server.tool
@register_tool
def categorize_examples(dataset_path: str) -> Dict[str, int]:
    """
    Load dataset entries and group them into the same evaluation categories used in HomeBench.

    Args:
        dataset_path: Path to the dataset file

    Returns:
        Dictionary with counts per category
    """
    print(f"Categorizing examples from {dataset_path}")

    categorized = TaskCategorizer.load_and_categorize_dataset(dataset_path)

    counts = {category: len(examples) for category, examples in categorized.items()}

    print(f"Categorization complete: {sum(counts.values())} total examples")
    for category, count in counts.items():
        if count > 0:
            print(f"  {category}: {count}")

    return counts


@mcp_server.tool
@register_tool
def evaluate_all_categories(results_path: str) -> Dict[str, Dict[str, float]]:
    """
    Compute metrics for each instruction category.

    Args:
        results_path: Path to the results file

    Returns:
        Dictionary mapping categories to their metrics
    """
    print(f"Evaluating all categories from {results_path}")

    try:
        results_by_category: Dict[str, List[TaskResult]] = {}

        with open(results_path, "r") as f:
            for line in f:
                if line.strip():
                    result_dict = json.loads(line)
                    # Reconstruct TaskResult (simplified)
                    category = result_dict.get("category", "normal_single")
                    if category not in results_by_category:
                        results_by_category[category] = []

                    # For metrics, we only need predictions and expected
                    predicted = result_dict.get("predicted_operations", [])
                    expected = result_dict.get("expected_operations", [])

                    results_by_category[category].append(
                        {"predicted": predicted, "expected": expected}
                    )

        # Compute metrics for each category
        category_metrics = {}
        all_predicted = []
        all_expected = []

        for category, results in results_by_category.items():
            cat_predicted = []
            cat_expected = []

            for result in results:
                cat_predicted.extend(result["predicted"])
                cat_expected.extend(result["expected"])
                all_predicted.extend(result["predicted"])
                all_expected.extend(result["expected"])

            if cat_predicted or cat_expected:
                category_metrics[category] = MetricsCalculator.compute_metrics(
                    cat_predicted, cat_expected
                )

        if all_predicted or all_expected:
            category_metrics["all"] = MetricsCalculator.compute_metrics(all_predicted, all_expected)

        print(f"Evaluated {len(results_by_category)} categories")

        return category_metrics

    except FileNotFoundError:
        print(f"Results file not found: {results_path}")
        print("Returning metrics from in-memory state instead")

        # Fall back to in-memory results
        category_metrics = {}

        for eval_id, results in state.task_results.items():
            # Group by category
            results_by_cat: Dict[str, List[TaskResult]] = {}
            for result in results:
                cat = result.category
                if cat not in results_by_cat:
                    results_by_cat[cat] = []
                results_by_cat[cat].append(result)

            # Compute metrics per category
            for category, cat_results in results_by_cat.items():
                if category not in category_metrics:
                    category_metrics[category] = MetricsCalculator.compute_aggregate_metrics(
                        cat_results
                    )

        # Compute overall
        all_results = []
        for results in state.task_results.values():
            all_results.extend(results)

        if all_results:
            category_metrics["all"] = MetricsCalculator.compute_aggregate_metrics(all_results)

        return category_metrics


@mcp_server.tool
@register_tool
def write_error_logs(predictions: List[Any], expected: List[Any], output_dir: str) -> str:
    """
    Generate JSONL files containing mismatched predictions for each category.

    Args:
        predictions: List of predicted operations
        expected: List of expected operations
        output_dir: Directory to save error logs

    Returns:
        Path to the generated error logs
    """
    print(f"Writing error logs to {output_dir}")

    from pathlib import Path

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    errors = []
    for i, (pred, exp) in enumerate(zip(predictions, expected)):
        pred_str = str(pred)
        exp_str = str(exp)

        if MetricsCalculator.normalize_operation(pred_str) != MetricsCalculator.normalize_operation(
            exp_str
        ):
            errors.append(
                {
                    "index": i,
                    "predicted": pred_str,
                    "expected": exp_str,
                    "timestamp": datetime.now().isoformat(),
                }
            )

    if len(predictions) != len(expected):
        errors.append(
            {
                "error": "length_mismatch",
                "predicted_count": len(predictions),
                "expected_count": len(expected),
                "timestamp": datetime.now().isoformat(),
            }
        )

    # Write to file
    error_log_path = output_path / "error_logs.jsonl"
    with open(error_log_path, "w") as f:
        for error in errors:
            f.write(json.dumps(error) + "\n")

    print(f"Wrote {len(errors)} errors to {error_log_path}")

    return str(error_log_path)


# HTTP API endpoints
@http_app.get("/")
async def root():
    """Root endpoint showing available tools."""
    return {
        "server": "HomeBench MCP HTTP Wrapper",
        "version": "1.0.0",
        "tools": list(TOOLS.keys()),
        "endpoints": {"list_tools": "GET /tools", "call_tool": "POST /tools/call"},
    }


@http_app.get("/tools")
async def list_tools():
    """List all available tools."""
    return {"tools": [{"name": name, "description": func.__doc__} for name, func in TOOLS.items()]}


@http_app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environments": len(state.environments),
        "evaluators": len(state.evaluators),
        "total_results": sum(len(results) for results in state.task_results.values()),
    }


@http_app.get("/state")
async def get_state():
    """Get current server state."""
    return {
        "environments": {
            env_id: {
                "initialized": env.initialized,
                "device_count": len(env.devices),
                "devices": list(env.devices.keys()),
                "action_count": len(env.action_log),
            }
            for env_id, env in state.environments.items()
        },
        "evaluators": list(state.evaluators.keys()),
        "current_env_id": state.current_env_id,
        "task_results": {eval_id: len(results) for eval_id, results in state.task_results.items()},
    }


@http_app.post("/state/reset")
async def reset_state():
    """Reset all server state."""
    logger.info("Resetting all state")
    state.clear_all()
    return {"status": "reset", "message": "All state cleared"}


@http_app.post("/tools/call")
async def call_tool(request: ToolCallRequest):
    """Call a specific tool by name with provided arguments."""
    tool_name = request.name
    arguments = request.arguments

    logger.info(f"Tool call requested: {tool_name}")
    logger.debug(f"Arguments: {arguments}")

    if tool_name not in TOOLS:
        logger.error(f"Tool not found: {tool_name}")
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")

    try:
        tool_func = TOOLS[tool_name]
        result = tool_func(**arguments)
        logger.info(f"Tool {tool_name} executed successfully")
        return ToolCallResponse(result=result, error=None)
    except TypeError as e:
        logger.error(f"Invalid arguments for {tool_name}: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=400, detail=f"Invalid arguments for tool '{tool_name}': {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error executing {tool_name}: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error executing tool '{tool_name}': {str(e)}")


if __name__ == "__main__":
    print("=" * 70)
    print("Starting HomeBench Green Agent MCP Server")
    print("=" * 70)
    print(f"HTTP API: http://localhost:9006")
    print(f"  - Root: http://localhost:9006/")
    print(f"  - List tools: http://localhost:9006/tools")
    print(f"  - Call tool: POST http://localhost:9006/tools/call")
    print("=" * 70)

    # Run HTTP API server (FastAPI with uvicorn)
    uvicorn.run(http_app, host="localhost", port=9006)
