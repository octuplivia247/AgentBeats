"""
Green Agent - Assessment Manager for BenchPress Smart Home Assessment
"""
import asyncio

import uvicorn

try:
    import tomllib
except ImportError:
    import tomli as tomllib

import json
import time

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.apps import A2AStarletteApplication
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard
from a2a.utils import new_agent_text_message

from src.green_agent.mcp.mcp_client import MCPClient
from src.utils import parse_tags, my_a2a


def get_env(env_name, user_strategy, user_model, task_split, user_provider=None, task_index=0):
    """Mock implementation of get_env for testing."""

    class MockEnv:
        def __init__(self):
            self.tools_info = []
            self.wiki = "Smart home environment"

    print(f"Creating mock environment: {env_name}, task {task_index}")
    return MockEnv()


async def ask_agent_to_solve(white_agent_url, env, task_index, max_num_steps=30):
    """Mock implementation of ask_agent_to_solve for testing."""
    print(f"Mock evaluation of agent at {white_agent_url}")

    class SolveResult:
        def __init__(self, reward, info):
            self.reward = reward
            self.info = info

    # Simulate evaluation
    return SolveResult(reward=1, info={"steps": 5})


def load_agent_card_toml(agent_name):
    current_dir = __file__.rsplit("/", 1)[0]
    with open(f"{current_dir}/{agent_name}.toml", "rb") as f:
        return tomllib.load(f)


class GreenAgentExecutor(AgentExecutor):

    def __init__(self):
        pass

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        # parse the task
        print("Green agent: Received a task, parsing...")
        user_input = context.get_user_input()
        tags = parse_tags(user_input)
        white_agent_url = tags["white_agent_url"]
        env_config_str = tags["env_config"]
        env_config = json.loads(env_config_str)

        # set up the environment
        # migrate from https://github.com/sierra-research/tau-bench/blob/4754e6b406507dbcbce8e8b3855dcf80aaec18ac/tau_bench/run.py#L20
        print("Green agent: Setting up the environment...")
        assert len(env_config["task_ids"]) == 1, "Only single task supported for demo purpose"
        task_index = env_config["task_ids"][0]
        env = get_env(
            env_name=env_config["env"],
            user_strategy=env_config["user_strategy"],
            user_model=env_config["user_model"],
            task_split=env_config["task_split"],
            user_provider=env_config.get("user_provider", None),
            task_index=task_index,
        )
        metrics = {}

        print("Green agent: Starting evaluation...")
        timestamp_started = time.time()
        # TODO: replace
        # agent = ToolCallingAgent(
        #     tools_info=env.tools_info,
        #     wiki=env.wiki,
        #     model="openai/gpt-4o",
        #     provider="openai",
        # )
        # res = agent.solve(
        #     env=env,
        #     task_index=task_index,
        # )
        res = await ask_agent_to_solve(white_agent_url, env, task_index)

        metrics["time_used"] = time.time() - timestamp_started
        result_bool = metrics["success"] = res.reward == 1
        result_emoji = "✅" if result_bool else "❌"

        print("Green agent: Evaluation complete.")
        await event_queue.enqueue_event(
            new_agent_text_message(
                f"Finished. White agent success: {result_emoji}\nMetrics: {metrics}\n"
            )
        )  # alternative, impl as a task-generating agent

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError


class GreenAgentMCPExecutor(AgentExecutor):
    """Green agent executor using MCP tools for HomeBench evaluation."""

    def __init__(self, mcp_url: str = "http://localhost:9006"):
        """
        Initialize the MCP-based executor.

        Args:
            mcp_url: URL of the MCP server
        """
        self.mcp_url = mcp_url
        self.mcp_client = None

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Execute evaluation using MCP tools."""
        # Initialize MCP client
        self.mcp_client = MCPClient(self.mcp_url)

        try:
            print("Green agent (MCP): Received a task, parsing...")
            user_input = context.get_user_input()
            tags = parse_tags(user_input)

            # Parse input - support both old format (white_agent_url) and new format (purple_agent_url)
            purple_agent_url = tags.get("purple_agent_url") or tags.get("white_agent_url")

            # Parse configuration
            if "evaluation_config" in tags:
                # New format
                config_str = tags["evaluation_config"]
                evaluation_config = json.loads(config_str)
            elif "env_config" in tags:
                # Old format - convert to new format
                env_config_str = tags["env_config"]
                env_config = json.loads(env_config_str)
                evaluation_config = {
                    "environment": env_config.get("env", "smart_home"),
                    "task_ids": env_config.get("task_ids", [1]),
                    "max_steps_per_task": 30,
                }
            else:
                # Default configuration
                evaluation_config = {"environment": "smart_home", "max_steps_per_task": 30}

            print(f"Green agent (MCP): Evaluating agent at {purple_agent_url}")
            print(f"Green agent (MCP): Configuration: {json.dumps(evaluation_config, indent=2)}")

            # Step 1: Initialize smart home environment
            await event_queue.enqueue_event(
                new_agent_text_message("Initializing smart home environment...")
            )

            env_config = {
                "rooms": ["living_room", "bedroom", "kitchen"],
                "devices": {
                    "living_room_light": {"type": "light", "state": "off"},
                    "living_room_thermostat": {"type": "thermostat", "temperature": 70},
                    "bedroom_light": {"type": "light", "state": "off"},
                    "kitchen_light": {"type": "light", "state": "off"},
                },
            }

            init_result = await self.mcp_client.initialize_smart_home(env_config)
            print(f"Environment initialized: {init_result}")

            # Step 2: Run evaluation for each task
            task_ids = evaluation_config.get("task_ids", [1])
            all_results = []

            await event_queue.enqueue_event(
                new_agent_text_message(f"Running evaluation for {len(task_ids)} task(s)...")
            )

            for task_id in task_ids:
                # Create a sample task
                task = {
                    "task_id": f"task_{task_id}",
                    "instruction": f"Turn on the living room light (Task {task_id})",
                    "expected_actions": ["turn_on_light"],
                }

                # Assign task
                print(f"Assigning task {task_id}...")
                assign_result = await self.mcp_client.assign_task(task)

                # Monitor actions
                print(f"Monitoring agent actions for task {task_id}...")
                monitoring_config = {
                    "task_id": task["task_id"],
                    "timeout": evaluation_config.get("max_steps_per_task", 30),
                }
                actions = await self.mcp_client.monitor_purple_agent_actions(monitoring_config)

                # Evaluate completion
                print(f"Evaluating task completion for task {task_id}...")
                eval_input = {
                    "task": task,
                    "actions": actions,
                    "final_states": {"living_room_light": {"state": "on"}},
                }
                eval_result = await self.mcp_client.evaluate_task_completion(eval_input)

                all_results.append(eval_result)

                status = "✅" if eval_result.get("success") else "❌"
                await event_queue.enqueue_event(
                    new_agent_text_message(
                        f"Task {task_id}: {status} (Score: {eval_result.get('score', 0.0)})"
                    )
                )

            # Step 3: Compute overall metrics
            print("Computing accuracy metrics...")
            predictions = [r.get("predicted_action", "none") for r in all_results]
            expected = [r.get("expected_action", "none") for r in all_results]

            metrics = await self.mcp_client.compute_accuracy_metrics(predictions, expected)

            # Step 4: Send final results
            success_count = sum(1 for r in all_results if r.get("success"))
            total_count = len(all_results)
            success_rate = success_count / total_count if total_count > 0 else 0

            final_message = f"""
Evaluation Complete! 🎉

Results:
- Tasks Completed: {success_count}/{total_count}
- Success Rate: {success_rate:.1%}

Metrics:
- Exact Match: {metrics.get('exact_match', 0):.2f}
- Precision: {metrics.get('precision', 0):.2f}
- Recall: {metrics.get('recall', 0):.2f}
- F1 Score: {metrics.get('f1', 0):.2f}

Agent URL: {purple_agent_url}
"""

            print("Green agent (MCP): Evaluation complete.")
            await event_queue.enqueue_event(new_agent_text_message(final_message))

        except Exception as e:
            print(f"Green agent (MCP): Error during execution: {e}")
            import traceback

            traceback.print_exc()
            await event_queue.enqueue_event(
                new_agent_text_message(f"Error during evaluation: {str(e)}")
            )
        finally:
            if self.mcp_client:
                await self.mcp_client.close()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the execution."""
        if self.mcp_client:
            await self.mcp_client.close()
        raise NotImplementedError("Cancellation not fully implemented")


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
        self.mcp_client = None

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
            if "evaluation_config" in tags:
                evaluation_config = json.loads(tags["evaluation_config"])
            else:
                evaluation_config = {
                    "environment": "smart_home",
                    "max_steps_per_task": 30,
                    "task_ids": [1],
                }

            print(f"Evaluating purple agent at: {purple_agent_url}")
            print(f"Configuration: {json.dumps(evaluation_config, indent=2)}")

            await event_queue.enqueue_event(
                new_agent_text_message(f"Starting evaluation of purple agent at {purple_agent_url}")
            )

            # Initialize environment
            env_config = {
                "rooms": ["living_room", "bedroom", "kitchen"],
                "devices": {
                    "living_room_light": {"type": "light", "state": "off"},
                    "living_room_thermostat": {"type": "thermostat", "temperature": 70},
                    "bedroom_light": {"type": "light", "state": "off"},
                    "kitchen_light": {"type": "light", "state": "off"},
                },
                "environment_id": f"eval_{int(time.time())}",
            }

            init_result = await self.mcp_client.initialize_smart_home(env_config)
            env_id = init_result.get("environment_id", "default")

            # Run evaluation for each task
            task_ids = evaluation_config.get("task_ids", [1])
            all_results = []

            for task_id in task_ids:
                task = self._create_task(task_id, env_id)

                # Assign task via A2A
                if not await my_a2a.wait_agent_ready(purple_agent_url, timeout=10):
                    raise RuntimeError(f"Purple agent at {purple_agent_url} is not ready")

                task_message = self._format_task_for_purple_agent(task, env_config)
                await my_a2a.send_message(purple_agent_url, task_message)
                await asyncio.sleep(2)  # Give time to execute

                # Monitor and evaluate
                monitoring_config = {
                    "task_id": task["task_id"],
                    "environment_id": env_id,
                    "timeout": evaluation_config.get("max_steps_per_task", 30),
                }

                actions = await self.mcp_client.monitor_purple_agent_actions(monitoring_config)

                eval_input = {
                    "task": task,
                    "actions": actions,
                    "agent_url": purple_agent_url,
                    "environment_id": env_id,
                    "eval_id": f"eval_{task_id}",
                }

                eval_result = await self.mcp_client.evaluate_task_completion(eval_input)
                all_results.append(eval_result)

            # Compute metrics
            predictions = [r.get("predicted_operations", []) for r in all_results]
            expected = [r.get("expected_operations", []) for r in all_results]
            flat_predictions = [item for sublist in predictions for item in sublist]
            flat_expected = [item for sublist in expected for item in sublist]

            metrics = await self.mcp_client.compute_accuracy_metrics(
                flat_predictions, flat_expected
            )

            # Generate report
            success_count = sum(1 for r in all_results if r.get("success"))
            total_count = len(all_results)
            success_rate = success_count / total_count if total_count > 0 else 0

            final_message = f"""
╔═══════════════════════════════════════════════════════════════╗
║              EVALUATION COMPLETE                              ║
╚═══════════════════════════════════════════════════════════════╝

Purple Agent: {purple_agent_url}
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

    def _create_task(self, task_id: int, env_id: str) -> dict:
        """Create a task definition."""
        tasks = {
            1: {
                "task_id": f"task_{task_id}",
                "instruction": "Turn on the living room light",
                "expected_operations": ["living_room_light.turn_on()"],
                "category": "normal_single",
            },
            2: {
                "task_id": f"task_{task_id}",
                "instruction": "Set the living room thermostat to 72 degrees",
                "expected_operations": ["living_room_thermostat.set_temperature(72)"],
                "category": "normal_single",
            },
            3: {
                "task_id": f"task_{task_id}",
                "instruction": "Turn on all the lights in the house",
                "expected_operations": [
                    "living_room_light.turn_on()",
                    "bedroom_light.turn_on()",
                    "kitchen_light.turn_on()",
                ],
                "category": "normal_multiple",
            },
        }
        task = tasks.get(task_id, tasks[1])
        task["environment_id"] = env_id
        return task

    def _format_task_for_purple_agent(self, task: dict, env_config: dict) -> str:
        """Format task as A2A message for purple agent."""
        tools_info = []
        for device_name, device_config in env_config.get("devices", {}).items():
            device_type = device_config.get("type", "unknown")
            if device_type == "light":
                tools_info.append(
                    {"device": device_name, "operations": ["turn_on", "turn_off", "get_state"]}
                )
            elif device_type == "thermostat":
                tools_info.append(
                    {"device": device_name, "operations": ["set_temperature", "get_temperature"]}
                )

        return f"""{task['instruction']}

<tools>
{json.dumps(tools_info, indent=2)}
</tools>

<environment_id>
{task.get('environment_id', 'default')}
</environment_id>

Please use the available MCP tools to complete this task.
"""

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the evaluation."""
        if self.mcp_client:
            await self.mcp_client.close()
        raise NotImplementedError("Cancellation not fully implemented")


def start_green_agent(
    agent_name="SmartHome_green_agent", host="localhost", port=9001, mcp_url=None
):
    """
    Start the green agent.

    Args:
        agent_name: Name of the agent configuration file (without .toml extension)
        host: Host to bind the server
        port: Port to bind the server
        mcp_url: Optional MCP server URL. If provided, uses MCP-based executor.
    """
    print("=" * 70)
    print("Starting Green Agent (Evaluation Orchestrator)")
    print("=" * 70)
    print(f"Agent URL: http://{host}:{port}")
    if mcp_url:
        print(f"MCP Server (for monitoring): {mcp_url}")
    print("=" * 70)

    agent_card_dict = load_agent_card_toml(agent_name)
    url = f"http://{host}:{port}"
    agent_card_dict["url"] = url

    # Choose executor
    if mcp_url:
        executor = GreenAgentEvaluator(mcp_url=mcp_url)
    else:
        print("Using standard executor (mock mode)")
        executor = GreenAgentExecutor()

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
