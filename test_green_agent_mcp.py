#!/usr/bin/env python3
"""
End-to-end test for Green Agent with MCP integration

This script tests the complete workflow:
1. MCP server is running
2. Green agent is running with MCP enabled
3. Send evaluation request to green agent
4. Verify results
"""

import asyncio
import sys
from src.utils.my_a2a import send_message, wait_agent_ready


async def test_green_agent_with_mcp():
    """Test the green agent with MCP integration."""
    
    print("=" * 70)
    print("Green Agent with MCP - End-to-End Test")
    print("=" * 70)
    print("\nPrerequisites:")
    print("1. MCP Server running: python src/green_agent/mcp_server.py")
    print("2. Green Agent running with MCP:")
    print("   agentbeats run src/green_agent/SmartHome_green_agent_mcp.toml \\")
    print("     --mcp http://localhost:9006")
    print("=" * 70)
    print()
    
    green_url = "http://localhost:9001"
    
    # Wait for agent to be ready
    print("[Step 1] Checking if green agent is ready...")
    is_ready = await wait_agent_ready(green_url, timeout=10)
    
    if not is_ready:
        print("❌ Green agent is not ready.")
        print("\nMake sure both services are running:")
        print("  Terminal 1: python src/green_agent/mcp_server.py")
        print("  Terminal 2: agentbeats run src/green_agent/SmartHome_green_agent_mcp.toml --mcp http://localhost:9006")
        return False
    
    print("✅ Green agent is ready!")
    
    # Test Case 1: Send evaluation request with new format
    print("\n[Step 2] Sending evaluation request (new format)...")
    test_message_new = """
Run HomeBench evaluation for the agent at:
<purple_agent_url>
http://localhost:8000/
</purple_agent_url>

Use this configuration:
<evaluation_config>
{
  "environment": "smart_home",
  "task_ids": [1, 2],
  "max_steps_per_task": 30
}
</evaluation_config>
"""
    
    try:
        response = await send_message(green_url, test_message_new)
        print("\n✅ Response received from green agent:")
        print("-" * 70)
        
        # Parse the response
        if hasattr(response, 'root'):
            root = response.root
            
            # Check if it's an error
            if hasattr(root, 'error'):
                print(f"❌ Error: {root.error}")
                return False
            
            # Check if it's a success response
            if hasattr(root, 'result'):
                result = root.result
                if hasattr(result, 'parts'):
                    for part in result.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            print(part.root.text)
                else:
                    print(result)
            else:
                print(root)
        else:
            print(response)
        
        print("-" * 70)
        print("\n✅ Test Case 1 (new format) passed!")
        
    except Exception as e:
        print(f"\n❌ Test Case 1 failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test Case 2: Send evaluation request with old format (for backward compatibility)
    print("\n[Step 3] Sending evaluation request (old format for compatibility)...")
    test_message_old = """
Your task is to evaluate the agent located at:
<white_agent_url>
http://localhost:8000/
</white_agent_url>

You should use the following env configuration:
<env_config>
{
  "env": "smart_home",
  "user_strategy": "llm",
  "user_model": "openai/gpt-4o",
  "user_provider": "openai",
  "task_split": "test",
  "task_ids": [1]
}
</env_config>
"""
    
    try:
        response = await send_message(green_url, test_message_old)
        print("\n✅ Response received from green agent:")
        print("-" * 70)
        
        # Parse the response
        if hasattr(response, 'root'):
            root = response.root
            
            if hasattr(root, 'error'):
                print(f"❌ Error: {root.error}")
                return False
            
            if hasattr(root, 'result'):
                result = root.result
                if hasattr(result, 'parts'):
                    for part in result.parts:
                        if hasattr(part, 'root') and hasattr(part.root, 'text'):
                            print(part.root.text)
                else:
                    print(result)
            else:
                print(root)
        else:
            print(response)
        
        print("-" * 70)
        print("\n✅ Test Case 2 (old format) passed!")
        
    except Exception as e:
        print(f"\n❌ Test Case 2 failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("🎉 All tests passed!")
    print("=" * 70)
    return True


async def main():
    """Main test runner."""
    success = await test_green_agent_with_mcp()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())

