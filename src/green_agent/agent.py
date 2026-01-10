"""
Green Agent - Evaluation Orchestrator for Smart Home Assessment

This agent evaluates purple agents on HomeBench tasks by:
1. Setting up smart home environments via MCP
2. Assigning tasks to purple agents via A2A
3. Monitoring agent actions via MCP
4. Computing evaluation metrics
"""

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import uvicorn

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard
from a2a.utils import new_agent_text_message

from src.green_agent.core.evaluator import HomeBenchEvaluator
from src.green_agent.core.smart_home_env_manager import SmartHomeEnvironment
from src.green_agent.core.task_loader import load_homebench_config, load_tasks_from_dataset
from src.green_agent.mcp.mcp_client import MCPClient
from src.utils import my_a2a, parse_tags


def load_agent_card_toml(agent_name: str) -> Dict[str, Any]:
    """Load agent card configuration from TOML file."""
    current_dir = Path(__file__).parent
    with open(current_dir / f"{agent_name}.toml", "rb") as f:
        return tomllib.load(f)


class GreenAgentEvaluator(AgentExecutor):
    """
    Green agent executor for evaluating purple agent via A2A.

    This executor:
    1. Receives evaluation requests from users
    2. Sets up the smart home environment via MCP
    3. Assigns tasks to purple agent via A2A
    4. Monitors purple agent's actions via MCP
    5. Evaluates completion and computes metrics
    """

    def __init__(self, mcp_url: str = "http://localhost:9006"):
        """Initialize the green agent evaluator."""
        self.mcp_url = mcp_url
        self.mcp_client: Optional[MCPClient] = None

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute evaluation of purple agent."""
        if self.mcp_client is None:
            self.mcp_client = MCPClient(self.mcp_url)

        try:
            print("=" * 70)
            print("Green Agent: Starting Evaluation")
            print("=" * 70)

            # Parse evaluation request
            user_input = context.get_user_input()
            tags = parse_tags(user_input)

            purple_agent_url = tags.get("purple_agent_url") or tags.get("white_agent_url")
            if not purple_agent_url:
                raise ValueError("purple_agent_url must be specified")

            # Parse configuration
            evaluation_config = self._parse_evaluation_config(tags)

            print(f"Evaluating purple agent at: {purple_agent_url}")
            print(f"Configuration: {json.dumps(evaluation_config, indent=2)}")

            await event_queue.enqueue_event(
                new_agent_text_message(f"Starting evaluation of purple agent at {purple_agent_url}")
            )

            # Load environment config
            env_config = self._load_environment_config(evaluation_config)
            env_id = f"eval_{int(time.time())}"
            env_config["environment_id"] = env_id

            # Initialize environment via MCP
            init_result = await self.mcp_client.initialize_smart_home(env_config)
            print(f"Environment initialized: {init_result}")

            # Load tasks
            tasks = self._load_tasks(evaluation_config)

            await event_queue.enqueue_event(
                new_agent_text_message(f"Loaded {len(tasks)} task(s) for evaluation")
            )

            # Evaluate each task
            all_results = []
            for i, task in enumerate(tasks):
                task["environment_id"] = env_id

                await event_queue.enqueue_event(
                    new_agent_text_message(f"Evaluating task {i+1}/{len(tasks)}: {task['task_id']}")
                )

                result = await self._evaluate_single_task(
                    task, purple_agent_url, env_id, evaluation_config
                )
                all_results.append(result)

                status = "✅" if result.get("success") else "❌"
                await event_queue.enqueue_event(
                    new_agent_text_message(
                        f"Task {task['task_id']}: {status} (Score: {result.get('score', 0.0):.2f})"
                    )
                )

            # Compute aggregate metrics
            metrics = await self._compute_aggregate_metrics(all_results)

            # Generate final report
            final_message = self._generate_report(
                purple_agent_url, env_id, all_results, metrics
            )

            await event_queue.enqueue_event(new_agent_text_message(final_message))

        except Exception as e:
            error_msg = f"Error during evaluation: {str(e)}"
            print(error_msg)
            import traceback
            traceback.print_exc()
            await event_queue.enqueue_event(new_agent_text_message(error_msg))
        finally:
            if self.mcp_client:
                await self.mcp_client.close()
                self.mcp_client = None

    def _parse_evaluation_config(self, tags: Dict[str, str]) -> Dict[str, Any]:
        """Parse evaluation configuration from tags."""
        if "evaluation_config" in tags:
            return json.loads(tags["evaluation_config"])
        elif "env_config" in tags:
            env_config = json.loads(tags["env_config"])
            return {
                "environment": env_config.get("env", "smart_home"),
                "task_ids": env_config.get("task_ids", [0]),
                "max_steps_per_task": 30,
                "dataset_path": env_config.get("dataset_path", "data/home_status_data.jsonl"),
                "method_path": env_config.get("method_path", "data/home_status_method.jsonl"),
            }
        else:
            return {
                "environment": "smart_home",
                "max_steps_per_task": 30,
                "task_ids": [0],
                "dataset_path": "data/home_status_data.jsonl",
                "method_path": "data/home_status_method.jsonl",
            }

    def _load_environment_config(self, evaluation_config: Dict[str, Any]) -> Dict[str, Any]:
        """Load environment configuration."""
        method_path = evaluation_config.get("method_path", "data/home_status_method.jsonl")

        if Path(method_path).exists():
            return load_homebench_config(method_path)
        else:
            # Fallback to simple config
            return {
                "rooms": ["living_room", "bedroom", "kitchen"],
                "devices": {
                    "living_room.light": {"type": "light", "state": "off"},
                    "living_room.thermostat": {"type": "heating", "state": "off"},
                    "bedroom.light": {"type": "light", "state": "off"},
                    "kitchen.light": {"type": "light", "state": "off"},
                },
            }

    def _load_tasks(self, evaluation_config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Load tasks for evaluation."""
        dataset_path = evaluation_config.get("dataset_path", "data/home_status_data.jsonl")
        task_ids = evaluation_config.get("task_ids")

        if Path(dataset_path).exists():
            return load_tasks_from_dataset(dataset_path, task_ids)
        else:
            # Fallback to default tasks
            return [
                {
                    "task_id": "task_1",
                    "instruction": "Turn on the living room light",
                    "expected_operations": ["living_room.light.turn_on()"],
                    "category": "normal_single",
                },
            ]

    async def _evaluate_single_task(
        self,
        task: Dict[str, Any],
        purple_agent_url: str,
        env_id: str,
        evaluation_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Evaluate a single task."""
        # Wait for agent to be ready
        if not await my_a2a.wait_agent_ready(purple_agent_url, timeout=10):
            return {
                "task_id": task["task_id"],
                "success": False,
                "score": 0.0,
                "error": "Agent not ready",
                "predicted_operations": [],
                "expected_operations": task.get("expected_operations", []),
            }

        # Format and send task
        task_message = self._format_task_message(task, env_id)
        await my_a2a.send_message(purple_agent_url, task_message)

        # Wait for execution
        wait_time = evaluation_config.get("wait_time", 3.0)
        await asyncio.sleep(wait_time)

        # Monitor actions
        monitoring_config = {
            "task_id": task["task_id"],
            "environment_id": env_id,
            "timeout": evaluation_config.get("max_steps_per_task", 30),
        }
        actions = await self.mcp_client.monitor_purple_agent_actions(monitoring_config)

        # Evaluate completion
        eval_input = {
            "task": task,
            "actions": actions,
            "agent_url": purple_agent_url,
            "environment_id": env_id,
        }
        return await self.mcp_client.evaluate_task_completion(eval_input)

    def _format_task_message(self, task: Dict[str, Any], env_id: str) -> str:
        """Format task as A2A message for purple agent."""
        return f"""{task['instruction']}

<environment_id>
{env_id}
</environment_id>

<task_id>
{task['task_id']}
</task_id>

Please use the available MCP tools to complete this task.
"""

    async def _compute_aggregate_metrics(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """Compute aggregate metrics from results."""
        predictions = [r.get("predicted_operations", []) for r in results]
        expected = [r.get("expected_operations", []) for r in results]

        flat_predictions = [item for sublist in predictions for item in sublist]
        flat_expected = [item for sublist in expected for item in sublist]

        return await self.mcp_client.compute_accuracy_metrics(flat_predictions, flat_expected)

    def _generate_report(
        self,
        agent_url: str,
        env_id: str,
        results: List[Dict[str, Any]],
        metrics: Dict[str, float],
    ) -> str:
        """Generate final evaluation report."""
        success_count = sum(1 for r in results if r.get("success"))
        total_count = len(results)
        success_rate = success_count / total_count if total_count > 0 else 0

        return f"""
╔═══════════════════════════════════════════════════════════════╗
║              EVALUATION COMPLETE                              ║
╚═══════════════════════════════════════════════════════════════╝

Purple Agent: {agent_url}
Environment: {env_id}

Task Results:
  • Completed: {success_count}/{total_count}
  • Success Rate: {success_rate:.1%}

Performance Metrics:
  • Exact Match: {metrics.get('exact_match', 0):.2%}
  • Precision: {metrics.get('precision', 0):.2%}
  • Recall: {metrics.get('recall', 0):.2%}
  • F1 Score: {metrics.get('f1', 0):.2%}

Evaluation ID: {env_id}
"""

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the evaluation."""
        if self.mcp_client:
            await self.mcp_client.close()
            self.mcp_client = None


def start_green_agent(
    agent_name: str = "SmartHome_green_agent",
    host: str = "localhost",
    port: int = 9001,
    mcp_url: Optional[str] = None,
) -> None:
    """
    Start the green agent.

    Args:
        agent_name: Name of the agent configuration file (without .toml extension)
        host: Host to bind the server
        port: Port to bind the server
        mcp_url: MCP server URL for monitoring (required for evaluation)
    """
    print("=" * 70)
    print("Starting Green Agent (Evaluation Orchestrator)")
    print("=" * 70)
    print(f"Agent URL: http://{host}:{port}")
    if mcp_url:
        print(f"MCP Server: {mcp_url}")
    else:
        print("WARNING: No MCP URL provided. Evaluation will not work properly.")
        mcp_url = "http://localhost:9006"
    print("=" * 70)

    agent_card_dict = load_agent_card_toml(agent_name)
    url = f"http://{host}:{port}"
    agent_card_dict["url"] = url

    executor = GreenAgentEvaluator(mcp_url=mcp_url)

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )

    print("Green agent is ready to orchestrate evaluations")
    uvicorn.run(app.build(), host=host, port=port)
