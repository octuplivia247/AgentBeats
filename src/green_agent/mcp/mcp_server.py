#!/usr/bin/env python3
"""
Green Agent MCP Server

This server exposes the HomeBench evaluation tools via both:
1. MCP protocol (SSE transport)
2. Simple HTTP REST API for easier testing
"""

from typing import Dict, List, Any, Optional
from fastmcp import FastMCP
from fastapi import FastAPI, HTTPException
import uvicorn
import asyncio
import json
import logging
from pathlib import Path
from datetime import datetime
import sys
import traceback

from src.green_agent.core.categorizer import TaskCategorizer
from src.green_agent.core.metrics_calculator import MetricsCalculator
from src.green_agent.core.models.task_result import TaskResult
from src.green_agent.core.smart_home_env_manager import SmartHomeEnvironment
from src.green_agent.core.evaluator import AgentCommunicator, HomeBenchEvaluator
from src.green_agent.mcp.state import GlobalState

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
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
def run_homebench_evaluation(purple_agent_url: str, evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
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
    
    dataset_path = evaluation_config.get(
        "dataset_path",
        "data/home_status_data.jsonl"
    )
    
    env_config = evaluation_config.get("environment_config", {
        "rooms": ["living_room", "bedroom", "kitchen"],
        "devices": {}
    })
    
    evaluator.environment.initialize(env_config)

    return {
        "status": "started",
        "eval_id": eval_id,
        "agent_url": purple_agent_url,
        "message": "Evaluation started."
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
        
        result = environment.initialize(environment_config)
        state.current_env_id = env_id
        
        logger.info(f"Environment {env_id} initialized with {result['device_count']} devices")
        
        return {
            **result,
            "environment_id": env_id
        }
    except NotImplementedError as e:
        logger.warning(f"Method not implemented: {e}")
        return {
            "status": "not_implemented",
            "error": str(e),
            "message": "This method needs to be implemented by the team. See src/green_agent/evaluator.py",
            "environment_id": environment_config.get("environment_id", "default")
        }
    except Exception as e:
        logger.error(f"Error initializing environment: {e}")
        logger.debug(traceback.format_exc())
        return {
            "status": "error",
            "error": str(e),
            "environment_id": environment_config.get("environment_id", "default")
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
            "operations": list(device.operations.keys()) if hasattr(device, 'operations') else []
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
            "message": "Task queued for async execution"
        }
    else:
        response = loop.run_until_complete(
            communicator.send_task(task.get("instruction", ""), tools_info)
        )
        
        return {
            "task_id": task.get("task_id"),
            "status": response.get("status", "unknown"),
            "agent_response": response.get("response", ""),
            "error": response.get("error")
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
                "error": action.error
            }
            for action in actions
        ]
    except NotImplementedError as e:
        logger.warning(f"Method not implemented: {e}")
        return [{
            "error": "not_implemented",
            "message": str(e),
            "note": "See src/green_agent/evaluator.py to implement this method"
        }]
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
    task = evaluation_input.get("task", {})
    print(f"Evaluating completion for task: {task.get('task_id')}")
    
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

        expected_ops = task.get("expected_operations", [])
        actions = evaluation_input.get("actions", [])
        
        predicted_ops = [f"{a['device']}.{a['action']}()" for a in actions if a.get('success')]
        metrics = MetricsCalculator.compute_metrics(predicted_ops, expected_ops)
        
        return {
            "task_id": task.get("task_id"),
            "success": metrics["exact_match"] == 1.0,
            "score": metrics["f1"],
            "feedback": f"F1: {metrics['f1']:.2f}, EM: {metrics['exact_match']:.2f}",
            "metrics": metrics
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
            "execution_time": result.execution_time
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
            "note": "See src/green_agent/evaluator.py to implement MetricsCalculator"
        }
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        logger.debug(traceback.format_exc())
        return {
            "exact_match": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "error": str(e)
        }

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
    
    counts = {
        category: len(examples)
        for category, examples in categorized.items()
    }
    
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
        
        with open(results_path, 'r') as f:
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
                    
                    results_by_category[category].append({
                        "predicted": predicted,
                        "expected": expected
                    })
        
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
        
        # Compute overall metrics
        if all_predicted or all_expected:
            category_metrics["all"] = MetricsCalculator.compute_metrics(
                all_predicted, all_expected
            )
        
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
                    category_metrics[category] = MetricsCalculator.compute_aggregate_metrics(cat_results)
        
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
        
        if MetricsCalculator.normalize_operation(pred_str) != MetricsCalculator.normalize_operation(exp_str):
            errors.append({
                "index": i,
                "predicted": pred_str,
                "expected": exp_str,
                "timestamp": datetime.now().isoformat()
            })
    
    # Also check for length mismatches
    if len(predictions) != len(expected):
        errors.append({
            "error": "length_mismatch",
            "predicted_count": len(predictions),
            "expected_count": len(expected),
            "timestamp": datetime.now().isoformat()
        })
    
    # Write to file
    error_log_path = output_path / "error_logs.jsonl"
    with open(error_log_path, 'w') as f:
        for error in errors:
            f.write(json.dumps(error) + '\n')
    
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
        "endpoints": {
            "list_tools": "GET /tools",
            "call_tool": "POST /tools/call"
        }
    }


@http_app.get("/tools")
async def list_tools():
    """List all available tools."""
    return {
        "tools": [
            {"name": name, "description": func.__doc__}
            for name, func in TOOLS.items()
        ]
    }


@http_app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "environments": len(state.environments),
        "evaluators": len(state.evaluators),
        "total_results": sum(len(results) for results in state.task_results.values())
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
                "action_count": len(env.action_log)
            }
            for env_id, env in state.environments.items()
        },
        "evaluators": list(state.evaluators.keys()),
        "current_env_id": state.current_env_id,
        "task_results": {
            eval_id: len(results)
            for eval_id, results in state.task_results.items()
        }
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
            status_code=400,
            detail=f"Invalid arguments for tool '{tool_name}': {str(e)}"
        )
    except Exception as e:
        logger.error(f"Error executing {tool_name}: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Error executing tool '{tool_name}': {str(e)}"
        )


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
