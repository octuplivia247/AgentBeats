#!/usr/bin/env python3
"""
Test script for direct MCP tool calls

This script tests the MCP server tools directly without going through the green agent.
"""

import asyncio
import httpx
import json
from typing import Any, Dict


class SimpleMCPClient:
    """Simple MCP client for testing."""
    
    def __init__(self, base_url: str = "http://localhost:9006"):
        self.base_url = base_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def close(self):
        await self.client.aclose()
    
    async def list_tools(self):
        """List all available tools."""
        try:
            # Try different possible endpoints
            endpoints = [
                f"{self.base_url}/mcp/list_tools",
                f"{self.base_url}/tools",
                f"{self.base_url}/list_tools",
            ]
            
            for endpoint in endpoints:
                try:
                    print(f"Trying endpoint: {endpoint}")
                    response = await self.client.get(endpoint)
                    if response.status_code == 200:
                        print(f"✅ Success with endpoint: {endpoint}")
                        return response.json()
                except Exception as e:
                    print(f"  Failed: {e}")
                    continue
            
            print("❌ Could not find tools endpoint")
            return None
        except Exception as e:
            print(f"Error listing tools: {e}")
            return None
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]):
        """Call a specific tool."""
        try:
            # Try different possible call endpoints
            endpoints = [
                f"{self.base_url}/mcp/call_tool",
                f"{self.base_url}/tools/call",
                f"{self.base_url}/call",
            ]
            
            payload = {
                "name": tool_name,
                "arguments": arguments
            }
            
            for endpoint in endpoints:
                try:
                    print(f"\nTrying to call {tool_name} at: {endpoint}")
                    response = await self.client.post(endpoint, json=payload)
                    if response.status_code == 200:
                        print(f"✅ Success!")
                        return response.json()
                    else:
                        print(f"  Status: {response.status_code}")
                        print(f"  Response: {response.text[:200]}")
                except Exception as e:
                    print(f"  Failed: {e}")
                    continue
            
            print(f"❌ Could not call tool {tool_name}")
            return None
        except Exception as e:
            print(f"Error calling tool: {e}")
            import traceback
            traceback.print_exc()
            return None


async def test_mcp_server():
    """Test the MCP server and its tools."""
    
    print("=" * 60)
    print("MCP Server Tool Test")
    print("=" * 60)
    print("\nMake sure the MCP server is running:")
    print("  python src/green_agent/mcp_server.py")
    print("=" * 60)
    print()
    
    client = SimpleMCPClient()
    
    try:
        # Test 1: Check if server is reachable
        print("\n[Test 1] Checking if MCP server is reachable...")
        try:
            response = await client.client.get(f"{client.base_url}/")
            print(f"✅ Server responded with status: {response.status_code}")
        except Exception as e:
            print(f"❌ Server not reachable: {e}")
            print("\nMake sure to start the MCP server first:")
            print("  python src/green_agent/mcp_server.py")
            return
        
        # Test 2: List available tools
        print("\n[Test 2] Listing available tools...")
        tools = await client.list_tools()
        if tools:
            print(f"Found tools: {json.dumps(tools, indent=2)}")
        
        # Test 3: Call initialize_smart_home
        print("\n[Test 3] Testing initialize_smart_home...")
        result = await client.call_tool("initialize_smart_home", {
            "environment_config": {
                "rooms": ["living_room", "bedroom", "kitchen"],
                "devices": {
                    "living_room_light": {"type": "light", "state": "off"},
                    "bedroom_thermostat": {"type": "thermostat", "temperature": 70}
                }
            }
        })
        if result:
            print(f"Result: {json.dumps(result, indent=2)}")
        
        # Test 4: Call assign_task
        print("\n[Test 4] Testing assign_task...")
        result = await client.call_tool("assign_task", {
            "task": {
                "task_id": "task_001",
                "instruction": "Turn on the living room light",
                "expected_actions": ["turn_on_light"]
            }
        })
        if result:
            print(f"Result: {json.dumps(result, indent=2)}")
        
        # Test 5: Call monitor_purple_agent_actions
        print("\n[Test 5] Testing monitor_purple_agent_actions...")
        result = await client.call_tool("monitor_purple_agent_actions", {
            "monitoring_config": {
                "task_id": "task_001",
                "timeout": 30
            }
        })
        if result:
            print(f"Result: {json.dumps(result, indent=2)}")
        
        # Test 6: Call evaluate_task_completion
        print("\n[Test 6] Testing evaluate_task_completion...")
        result = await client.call_tool("evaluate_task_completion", {
            "evaluation_input": {
                "task": {
                    "task_id": "task_001",
                    "instruction": "Turn on the living room light"
                },
                "actions": [
                    {"device": "living_room_light", "action": "turn_on"}
                ],
                "final_states": {
                    "living_room_light": {"state": "on"}
                }
            }
        })
        if result:
            print(f"Result: {json.dumps(result, indent=2)}")
        
        # Test 7: Call compute_accuracy_metrics
        print("\n[Test 7] Testing compute_accuracy_metrics...")
        result = await client.call_tool("compute_accuracy_metrics", {
            "predictions": ["action1", "action2", "action3"],
            "expected": ["action1", "action2", "action4"]
        })
        if result:
            print(f"Result: {json.dumps(result, indent=2)}")
        
        print("\n" + "=" * 60)
        print("Testing complete!")
        print("=" * 60)
        
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_mcp_server())

