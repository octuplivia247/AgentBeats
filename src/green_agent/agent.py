"""
Green Agent - Assessment Manager for BenchPress Smart Home Assessment
"""

import uvicorn
try:
    import tomllib
except ImportError:
    import tomli as tomllib
import json
import time

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, SendMessageSuccessResponse, Message
from a2a.utils import new_agent_text_message, get_text_parts
from src.utils import parse_tags
from src.green_agent.mcp.mcp_client import MCPClient


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
    



# async def ask_agent_to_solve(white_agent_url, env, task_index, max_num_steps=30):
#     # migrated from https://github.com/sierra-research/tau-bench/blob/4754e6b406507dbcbce8e8b3855dcf80aaec18ac/tau_bench/agents/tool_calling_agent.py#L27
#     total_cost = 0.0
#     env_reset_res = env.reset(task_index=task_index)
#     obs = env_reset_res.observation
#     info = env_reset_res.info.model_dump()
#     reward = 0.0

#     # messages = [
#     #     {"role": "system", "content": env.wiki},
#     #     {"role": "user", "content": obs},
#     # ]

#     # Here, instead of calling white agent like calling an LLM, we need to present
#     #   the assessment scenario to the white agent as if it is a independent task
#     # Specifically, here we provide the tool information for the agent to reply with
#     task_description = f"""
#         {env.wiki}
#         Here's a list of tools you can use (you can use at most one tool at a time):
#         {json.dumps(env.tools_info, indent=2)}
#         Please response in the JSON format. Please wrap the JSON part with <json>...</json> tags.
#         The JSON should contain:
#         - "name": the tool call function name, or "{RESPOND_ACTION_NAME}" if you want to respond directly.
#         - "kwargs": the arguments for the tool call, or {{"content": "your message here"}} if you want to respond directly.

#         Next, I'll provide you with the user message and tool call results.
#         User message: {obs}
#     """

#     next_green_message = task_description
#     context_id = None
#     for _ in range(max_num_steps):
#         # # --> messages (message history)
#         # res = completion(
#         #     messages=messages,
#         #     model=self.model,
#         #     custom_llm_provider=self.provider,
#         #     tools=self.tools_info,
#         #     temperature=self.temperature,
#         # )
#         # next_message = res.choices[0].message.model_dump()
#         # total_cost += res._hidden_params["response_cost"] or 0
#         # action = message_to_action(next_message)
#         # # --> action (to be executed in the environment)
#         print(
#             f"@@@ Green agent: Sending message to white agent{'ctx_id=' + str(context_id) if context_id else ''}... -->\n{next_green_message}"
#         )
#         white_agent_response = await my_a2a.send_message(
#             white_agent_url, next_green_message, context_id=context_id
#         )
#         res_root = white_agent_response.root
#         assert isinstance(res_root, SendMessageSuccessResponse)
#         res_result = res_root.result
#         assert isinstance(
#             res_result, Message
#         )  # though, a robust design should also support Task
#         if context_id is None:
#             context_id = res_result.context_id
#         else:
#             assert context_id == res_result.context_id, (
#                 "Context ID should remain the same in a conversation"
#             )

#         text_parts = get_text_parts(res_result.parts)
#         assert len(text_parts) == 1, (
#             "Expecting exactly one text part from the white agent"
#         )
#         white_text = text_parts[0]
#         print(f"@@@ White agent response:\n{white_text}")
#         # parse the action out
#         white_tags = parse_tags(white_text)
#         action_json = white_tags["json"]
#         action_dict = json.loads(action_json)
#         action = Action(**action_dict)

#         env_response = env.step(action)
#         reward = env_response.reward
#         info = {**info, **env_response.info.model_dump()}

#         # instead of maintain history, just prepare the next message with the latest observation
#         if action.name != RESPOND_ACTION_NAME:
#             next_green_message = f"""
#                 Tool call result:
#                 {env_response.observation}
#             """
#         else:
#             next_green_message = f"""
#                 User message:
#                 {env_response.observation}
#             """
#         if env_response.done:
#             break

#     return SolveResult(
#         reward=reward,
#         info=info,
#         messages=[],  # incompatible, thus removed
#         total_cost=total_cost,
#     )


class GreenAgentExecutor(AgentExecutor):
    """Original green agent executor using mock tau-bench approach."""
    
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
        assert len(env_config["task_ids"]) == 1, (
            "Only single task supported for demo purpose"
        )
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
                    "max_steps_per_task": 30
                }
            else:
                # Default configuration
                evaluation_config = {
                    "environment": "smart_home",
                    "max_steps_per_task": 30
                }
            
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
                    "kitchen_light": {"type": "light", "state": "off"}
                }
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
                    "expected_actions": ["turn_on_light"]
                }
                
                # Assign task
                print(f"Assigning task {task_id}...")
                assign_result = await self.mcp_client.assign_task(task)
                
                # Monitor actions
                print(f"Monitoring agent actions for task {task_id}...")
                monitoring_config = {
                    "task_id": task["task_id"],
                    "timeout": evaluation_config.get("max_steps_per_task", 30)
                }
                actions = await self.mcp_client.monitor_purple_agent_actions(monitoring_config)
                
                # Evaluate completion
                print(f"Evaluating task completion for task {task_id}...")
                eval_input = {
                    "task": task,
                    "actions": actions,
                    "final_states": {"living_room_light": {"state": "on"}}
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



def start_green_agent(
    agent_name="SmartHome_green_agent",
    host="localhost",
    port=9001,
    mcp_url=None
):
    """
    Start the green agent.
    
    Args:
        agent_name: Name of the agent configuration file (without .toml extension)
        host: Host to bind the server
        port: Port to bind the server
        mcp_url: Optional MCP server URL. If provided, uses MCP-based executor.
    """
    print("Starting green agent...")
    agent_card_dict = load_agent_card_toml(agent_name)
    url = f"http://{host}:{port}"
    agent_card_dict["url"] = url  # complete all required card fields

    # Choose executor based on whether MCP URL is provided
    if mcp_url:
        print(f"Using MCP executor with server at: {mcp_url}")
        executor = GreenAgentMCPExecutor(mcp_url=mcp_url)
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

    uvicorn.run(app.build(), host=host, port=port)
    

