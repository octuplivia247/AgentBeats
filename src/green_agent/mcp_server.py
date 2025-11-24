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
from pydantic import BaseModel
import uvicorn
import asyncio

# Initialize FastMCP server
mcp_server = FastMCP("HomeBench Green Agent MCP", version="1.0.0")

# Initialize FastAPI for HTTP wrapper
http_app = FastAPI(title="HomeBench MCP HTTP Wrapper")


# Request/Response models for HTTP API
class ToolCallRequest(BaseModel):
    name: str
    arguments: Dict[str, Any]


class ToolCallResponse(BaseModel):
    result: Any
    error: Optional[str] = None

# Tool Registry
TOOLS = {}

def register_tool(func):
    """Decorator to register tools in both MCP and HTTP API."""
    TOOLS[func.__name__] = func
    return func


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
    
    # Mock implementation
    return {
        "status": "completed",
        "summary": {
            "total_tasks": 40,
            "completed_tasks": 38,
            "success_rate": 0.95
        },
        "details_url": "http://localhost:8000/results/123"
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
    print("Initializing smart home environment...")
    print(f"Rooms: {environment_config.get('rooms', [])}")
    print(f"Devices: {len(environment_config.get('devices', {}))} devices configured")
    
    return {
        "status": "initialized",
        "environment_id": "env_001",
        "device_count": len(environment_config.get('devices', {}))
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
    
    # Mock response
    return {
        "task_id": task.get("task_id"),
        "status": "accepted",
        "agent_response": "I will execute this task."
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
    print(f"Monitoring actions for task: {monitoring_config.get('task_id')}")
    
    # Mock actions
    return [
        {"timestamp": "2023-01-01T12:00:01", "device": "living_room_light", "action": "turn_on", "success": True},
        {"timestamp": "2023-01-01T12:00:02", "device": "living_room_thermostat", "action": "set_temperature", "value": 72, "success": True}
    ]

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
    
    # Mock evaluation logic
    return {
        "task_id": task.get("task_id"),
        "success": True,
        "score": 1.0,
        "feedback": "All expected actions were performed correctly."
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
    print("Computing accuracy metrics...")
    
    # Mock metrics
    return {
        "exact_match": 0.85,
        "precision": 0.90,
        "recall": 0.88,
        "f1": 0.89
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
    
    return {
        "normal_single": 100,
        "unexist_single": 20,
        "unexist_device_single": 15,
        "unexist_attribute_single": 10,
        "normal_multi": 50,
        "mix_multi": 30,
        "error_multi": 25
    }

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
    
    metrics = {
        "exact_match": 0.85,
        "precision": 0.90,
        "recall": 0.88,
        "f1": 0.89
    }
    
    return {
        "all": metrics,
        "normal_single": metrics,
        "unexist_single": metrics,
        "normal_multi": metrics,
        "mix_multi": metrics,
        "error_multi": metrics
    }

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
    
    return f"{output_dir}/error_logs.jsonl"

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


@http_app.post("/tools/call")
async def call_tool(request: ToolCallRequest):
    """Call a specific tool by name with provided arguments."""
    tool_name = request.name
    arguments = request.arguments
    
    if tool_name not in TOOLS:
        raise HTTPException(status_code=404, detail=f"Tool '{tool_name}' not found")
    
    try:
        tool_func = TOOLS[tool_name]
        result = tool_func(**arguments)
        return ToolCallResponse(result=result)
    except TypeError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid arguments for tool '{tool_name}': {str(e)}"
        )
    except Exception as e:
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
