"""HomeBench Green Agent - Smart Home Evaluation Orchestrator with MCP Tools."""

import json
import logging
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from pydantic import BaseModel, HttpUrl, ValidationError
from dotenv import load_dotenv

from a2a.server.tasks import TaskUpdater
from a2a.types import Message, TaskState, Part, TextPart, DataPart
from a2a.utils import get_message_text, new_agent_text_message

from messenger import Messenger
from mcp_client import MCPClient
from metrics_calculator import HomeBenchMetricsCalculator

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
        self._home_data_cache: dict[int, dict] | None = None

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
        reduced_tests = config.get("reduced_tests", False)
        max_tasks = config.get("max_tasks", 50 if reduced_tests else None)

        # Try multiple path resolutions for different deployment scenarios
        resolved_path = self._resolve_data_path(path)

        if resolved_path and Path(resolved_path).exists():
            return self._load_from_file(resolved_path, config.get("task_ids"), reduced_tests, max_tasks)
        if "tasks" in config:
            return config["tasks"]

        # Fallback to demo tasks for testing
        logger.warning(f"Dataset file not found: {path}. Using demo tasks.")
        return [
            {"task_id": "demo_1", "instruction": "Turn on the living room light", "expected_operations": ["living_room.light.turn_on()"], "category": "valid_single"},
            {"task_id": "demo_2", "instruction": "Set the bedroom thermostat to 72 degrees", "expected_operations": ["bedroom.thermostat.set_temperature(72)"], "category": "valid_single"},
        ]

    def _resolve_data_path(self, path: str) -> str | None:
        """Resolve data file path for different deployment scenarios."""
        # 1. Try relative to current working directory (Docker with volume mount at /home/agent/data)
        if Path(path).exists():
            return path

        # 2. Try relative to agent.py location parent (local development from green-agent/src)
        agent_parent = Path(__file__).parent.parent / path
        if agent_parent.exists():
            return str(agent_parent)

        # 3. Try relative to project root (local development from root)
        project_root = Path(__file__).parent.parent.parent / path
        if project_root.exists():
            return str(project_root)

        # 4. Try absolute path if provided
        abs_path = Path(path)
        if abs_path.is_absolute() and abs_path.exists():
            return str(abs_path)

        return None

    def _load_from_file(
        self,
        path: str,
        task_ids: list[int] | None,
        reduced_tests: bool = False,
        max_tasks: int | None = None
    ) -> list[dict]:
        """Load tasks from JSONL file with optional sampling for reduced tests."""
        all_tasks = []
        with open(path) as f:
            for i, line in enumerate(f):
                if task_ids and i not in task_ids:
                    continue
                if line.strip():
                    task = json.loads(line)
                    task["task_id"] = task.get("task_id", task.get("id", f"task_{i}"))

                    # Parse HomeBench output format to expected_operations
                    if "expected_operations" not in task and "output" in task:
                        task["expected_operations"] = HomeBenchMetricsCalculator.parse_homebench_output(task["output"])

                    # Set category from type if not present
                    if "category" not in task and "type" in task:
                        task["category"] = task["type"]

                    all_tasks.append(task)
        
        # If no reduction needed, return all tasks
        if not reduced_tests and max_tasks is None:
            return all_tasks
        
        # Sample tasks stratified by category/type
        if reduced_tests or max_tasks:
            return self._sample_tasks_by_category(all_tasks, max_tasks or 50)
        
        return all_tasks

    def _sample_tasks_by_category(self, tasks: list[dict], max_tasks: int) -> list[dict]:
        """Sample tasks evenly across categories for representative testing."""
        by_category: dict[str, list[dict]] = defaultdict(list)
        for task in tasks:
            category = task.get("type", task.get("category", "unknown"))
            by_category[category].append(task)
        
        # Calculate tasks per category
        num_categories = len(by_category)
        if num_categories == 0:
            return tasks[:max_tasks]
        
        tasks_per_cat = max(1, max_tasks // num_categories)
        sampled = []
        
        for category, cat_tasks in by_category.items():
            sample_size = min(tasks_per_cat, len(cat_tasks))
            sampled.extend(random.sample(cat_tasks, sample_size))
        
        # If we have room for more, add random samples from remaining
        remaining = max_tasks - len(sampled)
        if remaining > 0:
            used_ids = {t.get("task_id") or t.get("id") for t in sampled}
            available = [t for t in tasks if (t.get("task_id") or t.get("id")) not in used_ids]
            if available:
                sampled.extend(random.sample(available, min(remaining, len(available))))
        
        return sampled[:max_tasks]

    def _load_home_data(self, config: dict[str, Any]) -> dict[int, dict]:
        """Load home status/method data and cache by home_id."""
        if self._home_data_cache is not None:
            return self._home_data_cache

        path = config.get("home_data_path", "data/home_status_method_all.jsonl")
        homes: dict[int, dict] = {}

        # Use same path resolution logic as task loading
        resolved_path = self._resolve_data_path(path)

        if not resolved_path or not Path(resolved_path).exists():
            logger.warning(f"Home data file not found: {path}")
            self._home_data_cache = homes
            return homes

        path = resolved_path
        
        with open(path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    home_id = data.get("home_id")
                    if home_id is not None:
                        homes[home_id] = data
        
        logger.info(f"Loaded home data for {len(homes)} homes")
        self._home_data_cache = homes
        return homes

    def _get_home_for_task(self, task: dict, config: dict[str, Any]) -> dict | None:
        """Get home data for a specific task based on home_id."""
        home_id = task.get("home_id")
        if home_id is None:
            return None
        
        homes = self._load_home_data(config)
        return homes.get(home_id)

    def _format_home_context(self, home_data: dict) -> str:
        """Format home data into a context string for the prompt."""
        methods = home_data.get("method", [])
        home_status = home_data.get("home_status", {})
        
        if not methods:
            return ""
        
        # Group methods by room and device
        room_devices: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for method in methods:
            room = method.get("room_name", "unknown")
            device = method.get("device_name", "unknown")
            operation = method.get("operation", "")
            params = method.get("parameters", [])
            
            # Format operation with parameter info
            if params:
                param_str = ", ".join(p.get("name", "") for p in params)
                op_formatted = f"{operation}({param_str})"
            else:
                op_formatted = f"{operation}()"
            
            room_devices[room][device].append(op_formatted)
        
        # Build context string
        lines = ["Available devices and operations in this home:"]
        for room, devices in sorted(room_devices.items()):
            lines.append(f"\n{room}:")
            for device, operations in sorted(devices.items()):
                ops_str = ", ".join(operations)
                lines.append(f"  - {device}: {ops_str}")
            
            # Add current state info if available
            if room in home_status:
                room_state = home_status[room]
                state_items = []
                for device in devices:
                    if device in room_state and isinstance(room_state[device], dict):
                        device_state = room_state[device]
                        state = device_state.get("state", "unknown")
                        attrs = device_state.get("attributes", {})
                        attr_vals = []
                        for attr_name, attr_data in attrs.items():
                            if isinstance(attr_data, dict) and "value" in attr_data:
                                attr_vals.append(f"{attr_name.strip()}={attr_data['value']}")
                        if attr_vals:
                            state_items.append(f"{device}: {state} ({', '.join(attr_vals)})")
                        else:
                            state_items.append(f"{device}: {state}")
                if state_items:
                    lines.append(f"    Current state: {'; '.join(state_items)}")
        
        return "\n".join(lines)

    async def _evaluate_task(self, task: dict, purple_url: str, config: dict) -> TaskResult:
        task_id = task.get("task_id", "unknown")
        expected = task.get("expected_operations", [])
        category = task.get("category", task.get("type", "unknown"))
        
        try:
            home_data = self._get_home_for_task(task, config)
            msg = self._format_message(task, home_data)
            response = await self.messenger.talk_to_agent(msg, purple_url, new_conversation=True, timeout=config.get("timeout_seconds", 60))
            logger.info(f"Response for {task_id}: {response[:200]}...")
            
            predicted = await self._parse_ops(response)
            success, score = self._compute_score(predicted, expected)
            
            return TaskResult(task_id=task_id, success=success, score=score, predicted_operations=predicted, expected_operations=expected, category=category)
        except Exception as e:
            logger.error(f"Error on {task_id}: {e}")
            return TaskResult(task_id=task_id, success=False, score=0.0, predicted_operations=[], expected_operations=expected, error=str(e), category=category)

    def _format_message(self, task: dict, home_data: dict | None = None) -> str:
        """Format the task message with optional home context."""
        instruction = task.get("instruction", task.get("input", ""))
        task_id = task.get("task_id", task.get("id", ""))
        home_id = task.get("home_id", 0)
        
        # Build home context section
        context_section = ""
        if home_data:
            context_section = f"""
{self._format_home_context(home_data)}

"""
        
        return f"""You are a smart home assistant. Execute the following instruction by generating the appropriate device operations.
{context_section}Instruction: {instruction}
Task ID: {task_id}
Home ID: {home_id}

IMPORTANT:
- Only use devices and operations that are available in this home (listed above).
- If a requested device or operation does not exist, respond with "error_input" for that part.
- Use the exact format: room_name.device_name.operation(parameters)

Respond ONLY with a JSON list of operations, no other text. Example format:
["living_room.light.turn_on()", "bedroom.thermostat.set_temperature(72)"]
For invalid requests, include "error_input" in the list.
"""

    async def _parse_ops(self, response: str) -> list[str]:
        if await self._mcp_available():
            try:
                return await self.mcp.parse_operations_from_response(response)
            except Exception:
                pass
                
        template_patterns = {
            "device.method(args)",
            "example.operation(args)",
            "namespace.device.operation(args)",
        }
        operations = []
        # Try to parse as JSON first
        try:
            operations = json.loads(response)
        except json.JSONDecodeError:
            # Fallback to regex
            operations = re.findall(r'[\w_]+(?:\.[\w_]+)+\([^)]*\)', response)
        
        # Filter out common template patterns
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
