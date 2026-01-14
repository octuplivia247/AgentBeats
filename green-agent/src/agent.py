"""HomeBench Green Agent - Smart Home Evaluation Orchestrator with MCP Tools."""

import json
import logging
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, HttpUrl, ValidationError
from dotenv import load_dotenv

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from mcp_client import MCPClient

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("homebench_green_agent")

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:9006")


# Models
class EvalRequest(BaseModel):
    participants: dict[str, HttpUrl]
    config: dict[str, Any]


class TaskResult(BaseModel):
    task_id: str
    success: bool
    score: float
    predicted_operations: list[str]
    expected_operations: list[str]
    error: str | None = None
    category: str = "unknown"


class CategoryMetrics(BaseModel):
    exact_match: float
    precision: float
    recall: float
    f1: float
    total_tasks: int


class EvalResult(BaseModel):
    agent_url: str
    total_tasks: int
    successful_tasks: int
    success_rate: float
    overall_metrics: CategoryMetrics
    category_metrics: dict[str, CategoryMetrics]
    task_results: list[TaskResult]


class Agent:
    """HomeBench Green Agent for evaluating smart home assistants."""
    
    required_roles = ["purple_agent"]

    def __init__(self):
        self.messenger = Messenger()
        self.mcp = MCPClient(MCP_SERVER_URL)
        self._mcp_ok: bool | None = None

    async def _mcp_available(self) -> bool:
        if self._mcp_ok is None:
            self._mcp_ok = await self.mcp.health_check()
        return self._mcp_ok

    async def run(self, message: Message, updater: TaskUpdater) -> None:
        input_text = get_message_text(message)

        try:
            request = EvalRequest.model_validate_json(input_text)
            if missing := set(self.required_roles) - set(request.participants.keys()):
                await updater.reject(new_agent_text_message(f"Missing roles: {missing}"))
                return
        except ValidationError as e:
            await updater.reject(new_agent_text_message(f"Invalid request: {e}"))
            return

        await updater.update_status(TaskState.working, new_agent_text_message("Starting evaluation..."))

        purple_url = str(request.participants["purple_agent"])
        config = request.config

        if not await self.messenger.wait_agent_ready(purple_url, timeout=30):
            await updater.failed(new_agent_text_message(f"Purple agent not responding: {purple_url}"))
            return

        tasks = self._load_tasks(config)
        await updater.update_status(TaskState.working, new_agent_text_message(f"Loaded {len(tasks)} tasks"))

        results: list[TaskResult] = []
        for i, task in enumerate(tasks):
            task_id = task.get("task_id", f"task_{i}")
            await updater.update_status(TaskState.working, new_agent_text_message(f"Evaluating {i+1}/{len(tasks)}: {task_id}"))
            
            result = await self._evaluate_task(task, purple_url, config)
            results.append(result)
            
            icon = "✅" if result.success else "❌"
            await updater.update_status(TaskState.working, new_agent_text_message(f"{icon} {task_id}: {result.score:.2f}"))

        eval_result = await self._compute_results(purple_url, results)
        report = self._generate_report(eval_result)
        
        await updater.add_artifact(
            parts=[Part(root=TextPart(text=report)), Part(root=DataPart(data=eval_result.model_dump()))],
            name="Evaluation Result",
        )
        await self.mcp.close()

    def _load_tasks(self, config: dict[str, Any]) -> list[dict]:
        path = config.get("dataset_path", "data/test_data.jsonl")
        if Path(path).exists():
            return self._load_from_file(path, config.get("task_ids"))
        if "tasks" in config:
            return config["tasks"]
        return [
            {"task_id": "demo_1", "instruction": "Turn on the living room light", "expected_operations": ["living_room.light.turn_on()"], "category": "valid_single"},
            {"task_id": "demo_2", "instruction": "Set the bedroom thermostat to 72 degrees", "expected_operations": ["bedroom.thermostat.set_temperature(72)"], "category": "valid_single"},
        ]

    def _load_from_file(self, path: str, task_ids: list[int] | None) -> list[dict]:
        tasks = []
        with open(path) as f:
            for i, line in enumerate(f):
                if task_ids and i not in task_ids:
                    continue
                if line.strip():
                    task = json.loads(line)
                    task["task_id"] = task.get("task_id", f"task_{i}")
                    tasks.append(task)
        return tasks

    async def _evaluate_task(self, task: dict, purple_url: str, config: dict) -> TaskResult:
        task_id = task.get("task_id", "unknown")
        expected = task.get("expected_operations", [])
        category = task.get("category", "unknown")
        
        try:
            msg = self._format_message(task)
            response = await self.messenger.talk_to_agent(msg, purple_url, new_conversation=True, timeout=config.get("timeout_seconds", 60))
            logger.info(f"Response for {task_id}: {response[:200]}...")
            
            predicted = await self._parse_ops(response)
            success, score = self._compute_score(predicted, expected)
            
            return TaskResult(task_id=task_id, success=success, score=score, predicted_operations=predicted, expected_operations=expected, category=category)
        except Exception as e:
            logger.error(f"Error on {task_id}: {e}")
            return TaskResult(task_id=task_id, success=False, score=0.0, predicted_operations=[], expected_operations=expected, error=str(e), category=category)

    def _format_message(self, task: dict) -> str:
        return f"""You are a smart home assistant. Execute the following instruction:
Instruction: {task.get("instruction", "")}
Task ID: {task.get("task_id", "")}
Home ID: {task.get("home_id", 0)}
Respond with device operations formatted as: namespace.device.operation(args)
Example: [living_room.light.turn_on()]
Only respond with operations inside square brackets."""

    async def _parse_ops(self, response: str) -> list[str]:
        if await self._mcp_available():
            try:
                return await self.mcp.parse_operations_from_response(response)
            except Exception:
                pass
        
        operations = re.findall(r'[\w_]+(?:\.[\w_]+)+\([^)]*\)', response)
        
        # Filter out common template patterns
        template_patterns = {
            "device.method(args)",
            "example.operation(args)",
            "namespace.device.operation(args)",
        }
        
        filtered = [op for op in operations if op not in template_patterns]
        return filtered if filtered else operations

    def _compute_score(self, predicted: list[str], expected: list[str]) -> tuple[bool, float]:
        if not expected:
            return len(predicted) == 0, 1.0 if not predicted else 0.0
        
        pred, exp = Counter(predicted), Counter(expected)
        exact = pred == exp
        tp = sum((pred & exp).values())
        prec = tp / len(predicted) if predicted else 0.0
        rec = tp / len(expected) if expected else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return exact, f1

    async def _compute_results(self, url: str, results: list[TaskResult]) -> EvalResult:
        total = len(results)
        successful = sum(1 for r in results if r.success)
        
        all_pred = [op for r in results for op in r.predicted_operations]
        all_exp = [op for r in results for op in r.expected_operations]
        overall = self._metrics(all_pred, all_exp, total)
        
        by_cat: dict[str, list[TaskResult]] = {}
        for r in results:
            by_cat.setdefault(r.category, []).append(r)
        
        cat_metrics = {}
        for cat, cat_results in by_cat.items():
            p = [op for r in cat_results for op in r.predicted_operations]
            e = [op for r in cat_results for op in r.expected_operations]
            cat_metrics[cat] = self._metrics(p, e, len(cat_results))
        
        return EvalResult(
            agent_url=url, total_tasks=total, successful_tasks=successful,
            success_rate=successful / total if total else 0.0,
            overall_metrics=overall, category_metrics=cat_metrics, task_results=results
        )

    def _metrics(self, pred: list[str], exp: list[str], count: int) -> CategoryMetrics:
        p, e = Counter(pred), Counter(exp)
        exact = 1.0 if p == e else 0.0
        tp = sum((p & e).values())
        prec = tp / len(pred) if pred else 0.0
        rec = tp / len(exp) if exp else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        return CategoryMetrics(exact_match=exact, precision=prec, recall=rec, f1=f1, total_tasks=count)

    def _generate_report(self, r: EvalResult) -> str:
        cats = "\n".join(f"  [{c}] Tasks: {m.total_tasks}, EM: {m.exact_match:.0%}, F1: {m.f1:.0%}" for c, m in r.category_metrics.items())
        tasks = "\n".join(f"  {'✅' if t.success else '❌'} {t.task_id}: {t.score:.2f}{' ('+t.error+')' if t.error else ''}" for t in r.task_results)
        return f"""
══════════════════════════════════════════════════════════════
HOMEBENCH EVALUATION COMPLETE
══════════════════════════════════════════════════════════════
Agent: {r.agent_url}
Total: {r.total_tasks} | Success: {r.successful_tasks} ({r.success_rate:.0%})

Overall: EM={r.overall_metrics.exact_match:.0%} P={r.overall_metrics.precision:.0%} R={r.overall_metrics.recall:.0%} F1={r.overall_metrics.f1:.0%}

Categories:
{cats}

Tasks:
{tasks}
"""
