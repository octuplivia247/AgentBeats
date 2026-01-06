"""
Purple Agent - Smart Home Task Execution Agent

This agent receives tasks via A2A protocol and uses MCP tools to interact with
the smart home environment. It is the agent being evaluated by the green agent.
"""

import json

import dotenv
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
from litellm import completion

from src.purple_agent.mcp_client import PurpleMCPClient

dotenv.load_dotenv()


def load_agent_card_toml(agent_name):
    """Load agent card configuration from TOML file."""
    current_dir = __file__.rsplit("/", 1)[0]
    with open(f"{current_dir}/{agent_name}.toml", "rb") as f:
        return tomllib.load(f)


class PurpleAgentExecutor(AgentExecutor):
    """
    Purple agent executor with MCP integration for smart home tasks.

    This executor receives tasks from the green agent via A2A protocol,
    uses an LLM to reason about the task, and calls MCP tools to interact
    with the smart home environment.
    """

    def __init__(self, mcp_url: str = "http://localhost:9006"):
        """
        Initialize the purple agent executor.

        Args:
            mcp_url: URL of the MCP server providing smart home tools
        """
        self.mcp_url = mcp_url
        self.mcp_client = None
        self.ctx_id_to_messages = {}
        self.ctx_id_to_tools = {}

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """
        Execute a task using MCP tools and LLM reasoning.

        Args:
            context: The request context containing the task
            event_queue: Queue for sending events back to the caller
        """
        if self.mcp_client is None:
            self.mcp_client = PurpleMCPClient(self.mcp_url)

        try:
            user_input = context.get_user_input()
            print(f"Purple agent received task: {user_input}")

            if context.context_id not in self.ctx_id_to_messages:
                self.ctx_id_to_messages[context.context_id] = []

                available_tools = await self.mcp_client.get_available_tools()
                self.ctx_id_to_tools[context.context_id] = available_tools

                system_message = self._build_system_message(available_tools)
                self.ctx_id_to_messages[context.context_id].append(
                    {"role": "system", "content": system_message}
                )

            messages = self.ctx_id_to_messages[context.context_id]
            messages.append(
                {
                    "role": "user",
                    "content": user_input,
                }
            )

            if "<tools>" in user_input and "</tools>" in user_input:
                tools_json = user_input[
                    user_input.find("<tools>") + 7 : user_input.find("</tools>")
                ]
                try:
                    tools_info = json.loads(tools_json)
                    print(f"Task includes {len(tools_info)} tool(s)")
                except json.JSONDecodeError:
                    print("Could not parse tools information")

            # Use LLM to reason about the task and decide on actions
            # The LLM will output JSON with tool calls
            response = completion(
                messages=messages,
                model="openai/gpt-4o",
                custom_llm_provider="openai",
                temperature=0.0,
            )

            next_message = response.choices[0].message.model_dump()
            assistant_response = next_message["content"]

            print(f"Purple agent LLM response: {assistant_response}")

            # Parse LLM response for tool calls
            tool_calls = self._parse_tool_calls(assistant_response)

            if tool_calls:
                # Execute tool calls via MCP
                tool_results = []
                for tool_call in tool_calls:
                    tool_name = tool_call.get("name")
                    tool_args = tool_call.get("arguments", {})

                    print(f"Executing MCP tool: {tool_name} with args: {tool_args}")

                    try:
                        result = await self.mcp_client.call_tool(tool_name, tool_args)
                        tool_results.append({"tool": tool_name, "result": result, "success": True})
                        print(f"Tool {tool_name} executed successfully: {result}")
                    except Exception as e:
                        tool_results.append({"tool": tool_name, "error": str(e), "success": False})
                        print(f"Tool {tool_name} failed: {e}")

                # Format tool results for response
                results_text = self._format_tool_results(tool_results)
                final_response = f"{assistant_response}\n\nTool Execution Results:\n{results_text}"
            else:
                final_response = assistant_response

            # Add assistant message to history
            messages.append(
                {
                    "role": "assistant",
                    "content": final_response,
                }
            )

            # Send response back via A2A
            await event_queue.enqueue_event(
                new_agent_text_message(final_response, context_id=context.context_id)
            )

        except Exception as e:
            error_msg = f"Error in purple agent execution: {str(e)}"
            print(error_msg)
            import traceback

            traceback.print_exc()
            await event_queue.enqueue_event(
                new_agent_text_message(error_msg, context_id=context.context_id)
            )

    def _build_system_message(self, available_tools):
        """Build system message with tool information."""
        tools_desc = "\n".join(
            [
                f"- {tool['name']}: {tool.get('description', 'No description')}"
                for tool in available_tools
            ]
        )

        return f"""You are a smart home assistant with direct access to MCP device control tools.

CRITICAL: You must ACTUALLY EXECUTE actions using MCP tools - do NOT just describe what you would do!

Available MCP Tools:
{tools_desc}

HOW TO CONTROL DEVICES:
When you receive a task like "Turn on the living room light", you should:
1. Identify the device name (e.g., "living_room_light")
2. Identify the action (e.g., "turn_on")
3. Call the control_device tool using the format below

TOOL CALL FORMAT:
<tool_call>
{{
  "name": "control_device",
  "arguments": {{
    "device_name": "living_room_light",
    "action": "turn_on",
    "parameters": {{}}
  }}
}}
</tool_call>

EXAMPLES:
Task: "Turn on the living room light"
→ <tool_call>{{"name": "control_device", "arguments": {{"device_name": "living_room_light", "action": "turn_on"}}}}</tool_call>

Task: "Set thermostat to 72 degrees"
→ <tool_call>{{"name": "control_device", "arguments": {{"device_name": "living_room_thermostat", "action": "set_temperature", "parameters": {{"temperature": 72}}}}}}</tool_call>

Task: "Turn on all lights"
→ Multiple <tool_call> blocks, one for each light

IMPORTANT RULES:
1. ALWAYS use tool calls - NEVER just say what you would do
2. Use control_device for all device operations
3. Extract device names from the <tools> section
4. Wrap EVERY tool call in <tool_call></tool_call> tags
5. Make tool calls immediately - don't just plan"""

    def _parse_tool_calls(self, response: str):
        """Parse tool calls from LLM response."""
        tool_calls = []

        # Look for <tool_call> tags
        import re

        pattern = r"<tool_call>(.*?)</tool_call>"
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            try:
                tool_call = json.loads(match.strip())
                tool_calls.append(tool_call)
            except json.JSONDecodeError as e:
                print(f"Failed to parse tool call: {e}")
                continue

        pattern = r"<json>(.*?)</json>"
        matches = re.findall(pattern, response, re.DOTALL)

        for match in matches:
            try:
                action_data = json.loads(match.strip())
                # Convert to tool call format
                if "name" in action_data:
                    tool_calls.append(
                        {"name": action_data["name"], "arguments": action_data.get("kwargs", {})}
                    )
            except json.JSONDecodeError as e:
                print(f"Failed to parse JSON action: {e}")
                continue

        return tool_calls

    def _format_tool_results(self, tool_results):
        """Format tool execution results for display."""
        lines = []
        for result in tool_results:
            if result.get("success"):
                lines.append(f"✓ {result['tool']}: {json.dumps(result['result'], indent=2)}")
            else:
                lines.append(f"✗ {result['tool']}: {result.get('error', 'Unknown error')}")
        return "\n".join(lines)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel the execution."""
        if self.mcp_client:
            await self.mcp_client.close()
        raise NotImplementedError("Cancellation not fully implemented")


def start_purple_agent(
    agent_name="smarthome_purple_agent",
    host="localhost",
    port=9000,
    mcp_url="http://localhost:9006",
):
    """
    Start the purple agent server.

    Args:
        agent_name: Name of the agent configuration file (without .toml extension)
        host: Host to bind the server
        port: Port to bind the server
        mcp_url: URL of the MCP server
    """
    print("=" * 70)
    print("Starting Purple Agent (Smart Home Task Executor)")
    print("=" * 70)
    print(f"Agent URL: http://{host}:{port}")
    print(f"MCP Server: {mcp_url}")
    print("=" * 70)

    agent_card_dict = load_agent_card_toml(agent_name)
    url = f"http://{host}:{port}"
    agent_card_dict["url"] = url

    executor = PurpleAgentExecutor(mcp_url=mcp_url)

    request_handler = DefaultRequestHandler(
        agent_executor=executor,
        task_store=InMemoryTaskStore(),
    )

    app = A2AStarletteApplication(
        agent_card=AgentCard(**agent_card_dict),
        http_handler=request_handler,
    )

    print("Purple agent is ready to receive tasks via A2A protocol")
    uvicorn.run(app.build(), host=host, port=port)
