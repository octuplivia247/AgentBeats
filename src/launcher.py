"""
System Launcher - Starts the complete AgentBeats system

Architecture:
1. MCP Server - Provides smart home tools (port 9006)
2. Purple Agent - Task executor using MCP tools (port 9000)
3. Green Agent - Evaluation orchestrator via A2A (port 9001)

Flow:
User → Green Agent (A2A) → Purple Agent (A2A) → MCP Server (HTTP)
              ↓ (monitoring)
        MCP Server
"""

import asyncio
import json
import multiprocessing
import time

from src.green_agent.agent import start_green_agent
from src.purple_agent.agent import start_purple_agent
from src.utils import my_a2a


def start_mcp_server():
    """Start the MCP server in a separate process."""
    import uvicorn

    from src.green_agent.mcp.mcp_server import http_app

    print("Starting MCP Server...")
    uvicorn.run(http_app, host="localhost", port=9006)


async def launch_evaluation_refactored():
    """
    Launch the complete refactored evaluation system.

    System architecture:
    - MCP Server (9006): Provides smart home tools
    - Purple Agent (9000): Executes tasks using MCP tools
    - Green Agent (9001): Orchestrates evaluation via A2A
    """
    print("\n" + "=" * 70)
    print("=" * 70)
    print("Architecture:")
    print("  1. MCP Server (port 9006) - Smart home tools")
    print("  2. Purple Agent (port 9000) - Task executor")
    print("  3. Green Agent (port 9001) - Evaluation orchestrator")
    print("=" * 70 + "\n")

    # Start MCP server
    print("Step 1: Starting MCP Server...")
    mcp_address = ("localhost", 9006)
    mcp_url = f"http://{mcp_address[0]}:{mcp_address[1]}"
    p_mcp = multiprocessing.Process(target=start_mcp_server)
    p_mcp.start()

    print("Waiting for MCP server to be ready...")
    await asyncio.sleep(3)

    import httpx

    for i in range(10):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{mcp_url}/health")
                if response.status_code == 200:
                    print("✓ MCP Server is ready")
                    break
        except Exception:
            print(f"  Waiting for MCP server... ({i+1}/10)")
            await asyncio.sleep(1)
    else:
        print("✗ MCP Server failed to start")
        p_mcp.terminate()
        return

    print("\nStep 2: Starting Purple Agent...")
    purple_address = ("localhost", 9000)
    purple_url = f"http://{purple_address[0]}:{purple_address[1]}"
    p_purple = multiprocessing.Process(
        target=start_purple_agent, args=("smarthome_purple_agent", *purple_address, mcp_url)
    )
    p_purple.start()

    if not await my_a2a.wait_agent_ready(purple_url, timeout=15):
        print("✗ Purple agent not ready in time")
        p_purple.terminate()
        p_mcp.terminate()
        return
    print("✓ Purple Agent is ready")

    print("\nStep 3: Starting Green Agent...")
    green_address = ("localhost", 9001)
    green_url = f"http://{green_address[0]}:{green_address[1]}"
    p_green = multiprocessing.Process(
        target=start_green_agent, args=("SmartHome_green_agent", *green_address, mcp_url)
    )
    p_green.start()

    if not await my_a2a.wait_agent_ready(green_url, timeout=15):
        print("✗ Green agent not ready in time")
        p_green.terminate()
        p_purple.terminate()
        p_mcp.terminate()
        return
    print("✓ Green Agent is ready")

    print("\n" + "=" * 70)
    print("ALL AGENTS READY")
    print("=" * 70)
    print(f"MCP Server:    {mcp_url}")
    print(f"Purple Agent:  {purple_url}")
    print(f"Green Agent:   {green_url}")
    print("=" * 70 + "\n")

    # Send evaluation request to green agent
    print("Step 4: Sending evaluation request to Green Agent...")

    evaluation_config = {
        "environment": "smart_home",
        "task_ids": [1, 2, 3],  # Multiple tasks
        "max_steps_per_task": 30,
    }

    task_text = f"""
Please evaluate the purple agent located at:
<purple_agent_url>
{purple_url}
</purple_agent_url>

Use the following evaluation configuration:
<evaluation_config>
{json.dumps(evaluation_config, indent=2)}
</evaluation_config>

The purple agent will use MCP tools to interact with the smart home environment.
Please orchestrate the evaluation and report the results.
"""

    print("\n--- Evaluation Request ---")
    print(task_text)
    print("--- End Request ---\n")

    print("Sending request to green agent...")
    response = await my_a2a.send_message(green_url, task_text)

    print("\n" + "=" * 70)
    print("GREEN AGENT RESPONSE")
    print("=" * 70)

    # Parse response
    from a2a.types import Message, SendMessageSuccessResponse
    from a2a.utils import get_text_parts

    res_root = response.root
    if isinstance(res_root, SendMessageSuccessResponse):
        res_result = res_root.result
        if isinstance(res_result, Message):
            text_parts = get_text_parts(res_result.parts)
            if text_parts:
                for part in text_parts:
                    print(part)
        else:
            print(f"Unexpected result type: {type(res_result)}")
    else:
        print(f"Response: {response}")

    print("=" * 70 + "\n")

    print("Evaluation complete. Press Ctrl+C to terminate all agents...")

    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n\nShutting down...")

    # Cleanup
    print("Terminating agents...")
    p_green.terminate()
    p_green.join()
    p_purple.terminate()
    p_purple.join()
    p_mcp.terminate()
    p_mcp.join()
    print("All agents terminated.")


async def quick_test():
    """
    Quick test to verify purple agent can use MCP tools.
    """
    print("\n" + "=" * 70)
    print("QUICK TEST: Purple Agent with MCP Tools")
    print("=" * 70 + "\n")

    # Start MCP server
    print("Starting MCP Server...")
    mcp_address = ("localhost", 9006)
    mcp_url = f"http://{mcp_address[0]}:{mcp_address[1]}"
    p_mcp = multiprocessing.Process(target=start_mcp_server)
    p_mcp.start()

    await asyncio.sleep(3)

    # Start purple agent
    print("Starting Purple Agent...")
    purple_address = ("localhost", 9000)
    purple_url = f"http://{purple_address[0]}:{purple_address[1]}"
    p_purple = multiprocessing.Process(
        target=start_purple_agent, args=("smarthome_purple_agent", *purple_address, mcp_url)
    )
    p_purple.start()

    if not await my_a2a.wait_agent_ready(purple_url, timeout=15):
        print("✗ Purple agent not ready")
        p_purple.terminate()
        p_mcp.terminate()
        return

    print("✓ Purple Agent ready\n")

    # Send a simple task
    task = "Turn on the living room light"
    print(f"Sending task: {task}")

    response = await my_a2a.send_message(purple_url, task)

    # Print response
    from a2a.types import Message, SendMessageSuccessResponse
    from a2a.utils import get_text_parts

    print("\nPurple Agent Response:")
    print("-" * 70)
    res_root = response.root
    if isinstance(res_root, SendMessageSuccessResponse):
        res_result = res_root.result
        if isinstance(res_result, Message):
            text_parts = get_text_parts(res_result.parts)
            if text_parts:
                print(text_parts[0])
    print("-" * 70 + "\n")

    # Cleanup
    print("Test complete. Cleaning up...")
    p_purple.terminate()
    p_purple.join()
    p_mcp.terminate()
    p_mcp.join()
    print("Done.")


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "test":
        asyncio.run(quick_test())
    else:
        asyncio.run(launch_evaluation_refactored())
