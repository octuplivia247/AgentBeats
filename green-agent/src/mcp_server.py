#!/usr/bin/env python3
"""Green Agent MCP Server - Exposes HomeBench evaluation tools via HTTP."""

import logging
import re
from datetime import datetime
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException

from metrics_calculator import MetricsCalculator, HomeBenchMetricsCalculator
from models import ToolCallRequest, ToolCallResponse, ActionLog

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

http_app = FastAPI(title="HomeBench Green Agent MCP Server")
TOOLS: dict[str, callable] = {}


def register_tool(func):
    TOOLS[func.__name__] = func
    return func


class GlobalState:
    def __init__(self):
        self.action_logs: list[ActionLog] = []
    
    def log_action(self, device_name: str, action: str, parameters: dict, success: bool, error: Optional[str] = None) -> ActionLog:
        log = ActionLog.create(device_name=device_name, action=action, parameters=parameters, success=success, error=error)
        self.action_logs.append(log)
        return log
    
    def get_action_logs(self) -> list[dict]:
        return [log.to_dict() for log in self.action_logs]
    
    def clear(self):
        self.action_logs.clear()


state = GlobalState()
metrics = MetricsCalculator()
homebench_metrics = HomeBenchMetricsCalculator()


# Tools
@register_tool
def compute_accuracy_metrics(predictions: list[str], expected: list[str]) -> dict[str, float]:
    """Compute EM, precision, recall, F1 for predicted vs expected operations."""
    return metrics.compute_metrics(predictions, expected)


@register_tool
def evaluate_task_completion(evaluation_input: dict[str, Any]) -> dict[str, Any]:
    """Evaluate if purple agent completed task successfully."""
    task = evaluation_input.get("task", {})
    task_id = task.get("task_id", "unknown")
    expected_ops = task.get("expected_operations", [])
    predicted_ops = evaluation_input.get("predicted_operations", [])
    
    result = metrics.compute_metrics(predicted_ops, expected_ops)
    return {
        "task_id": task_id,
        "success": result["exact_match"] == 1.0,
        "score": result["f1"],
        "metrics": result,
        "predicted_operations": predicted_ops,
        "expected_operations": expected_ops,
    }


@register_tool
def evaluate_homebench_task(task_data: dict[str, Any]) -> dict[str, Any]:
    """Evaluate task in HomeBench triple-quote format."""
    expected_ops = homebench_metrics.parse_homebench_output(task_data.get("expected_output", ""))
    predicted_ops = homebench_metrics.parse_homebench_output(task_data.get("predicted_output", ""))
    result = metrics.compute_metrics(predicted_ops, expected_ops)
    return {"metrics": result, "predicted_operations": predicted_ops, "expected_operations": expected_ops, "success": result["exact_match"] == 1.0}


@register_tool
def parse_operations_from_response(response: str) -> list[str]:
    """Extract device operations from agent response text."""
    pattern = r'[\w_]+(?:\.[\w_]+)+\([^)]*\)'
    return re.findall(pattern, response)


@register_tool
def log_device_action(device_name: str, action: str, parameters: dict[str, Any] | None = None, success: bool = True, error: str | None = None) -> dict[str, Any]:
    """Log a device action."""
    return state.log_action(device_name, action, parameters or {}, success, error).to_dict()


@register_tool
def get_action_logs() -> list[dict[str, Any]]:
    """Get all logged actions."""
    return state.get_action_logs()


@register_tool
def clear_action_logs() -> dict[str, str]:
    """Clear action logs."""
    state.clear()
    return {"status": "cleared"}


@register_tool
def compute_batch_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute aggregate metrics across multiple tasks."""
    all_pred, all_exp = [], []
    per_task = []
    
    for i, r in enumerate(results):
        pred, exp = r.get("predicted_operations", []), r.get("expected_operations", [])
        all_pred.extend(pred)
        all_exp.extend(exp)
        m = metrics.compute_metrics(pred, exp)
        per_task.append({"task_index": i, "task_id": r.get("task_id", f"task_{i}"), **m})
    
    agg = metrics.compute_metrics(all_pred, all_exp)
    exact_matches = sum(1 for m in per_task if m["exact_match"] == 1.0)
    
    return {"aggregate": agg, "total_tasks": len(results), "exact_match_count": exact_matches, "exact_match_rate": exact_matches / len(results) if results else 0.0, "per_task": per_task}


# HTTP Endpoints
@http_app.get("/")
async def root():
    return {"server": "HomeBench MCP Server", "tools": list(TOOLS.keys())}


@http_app.get("/tools")
async def list_tools():
    return {"tools": [{"name": n, "description": f.__doc__ or ""} for n, f in TOOLS.items()]}


@http_app.get("/health")
async def health():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


@http_app.post("/tools/call")
async def call_tool(request: ToolCallRequest) -> ToolCallResponse:
    if request.name not in TOOLS:
        raise HTTPException(status_code=404, detail=f"Tool '{request.name}' not found")
    try:
        result = TOOLS[request.name](**request.arguments)
        return ToolCallResponse(result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def run_server(host: str = "0.0.0.0", port: int = 9006):
    print(f"Starting MCP Server on http://{host}:{port}")
    uvicorn.run(http_app, host=host, port=port)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9006)
    args = parser.parse_args()
    run_server(args.host, args.port)
