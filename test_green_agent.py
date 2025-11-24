#!/usr/bin/env python3
"""
Test script for the Green Agent
Sends a test message using the A2A protocol
"""

import asyncio
import sys
from src.utils.my_a2a import send_message, wait_agent_ready


async def test_green_agent():
    """Send a test message to the green agent"""
    
    green_url = "http://localhost:9001"
    
    # Wait for agent to be ready
    print("Checking if green agent is ready...")
    is_ready = await wait_agent_ready(green_url, timeout=10)
    
    if not is_ready:
        print("❌ Green agent is not ready. Make sure it's running with:")
        print("   agentbeats run src/green_agent/SmartHome_green_agent_mcp.toml --mcp http://localhost:9006/sse")
        return
    
    print("✅ Green agent is ready!")
    
    # Send a test message
    print("\nSending test message to green agent...")
    test_message = """
Your task is to evaluate the agent located at:
<white_agent_url>
http://localhost:9002/
</white_agent_url>

You should use the following env configuration:
<env_config>
{
  "env": "retail",
  "user_strategy": "llm",
  "user_model": "openai/gpt-4o",
  "user_provider": "openai",
  "task_split": "test",
  "task_ids": [1]
}
</env_config>
"""
    
    try:
        response = await send_message(green_url, test_message)
        print("\n✅ Response from green agent:")
        print(response)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    print("=" * 60)
    print("Green Agent Test Script")
    print("=" * 60)
    print("\nMake sure the following are running:")
    print("1. MCP Server: python src/green_agent/mcp_server.py")
    print("2. Green Agent: agentbeats run src/green_agent/SmartHome_green_agent_mcp.toml --mcp http://localhost:9006/sse")
    print("=" * 60)
    print()
    
    asyncio.run(test_green_agent())

