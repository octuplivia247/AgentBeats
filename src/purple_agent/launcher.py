"""
Purple Agent Launcher for AgentBeats Platform

This launcher receives reset signals from agentbeats.org and manages
the purple agent lifecycle during battles.
"""

from fastapi import FastAPI
import uvicorn
import multiprocessing
import asyncio

from src.purple_agent.agent import start_purple_agent

launcher_app = FastAPI(title="Purple Agent Launcher")

# Global process reference
purple_agent_process = None


@launcher_app.post("/reset")
async def reset_agent():
    """
    Reset endpoint called by agentbeats.org before each battle.
    
    This should:
    1. Stop current agent instance
    2. Clear any state
    3. Start fresh agent instance
    """
    global purple_agent_process
    
    print("Launcher: Received reset signal from AgentBeats.org")
    
    # Stop current agent if running
    if purple_agent_process and purple_agent_process.is_alive():
        print("Stopping current purple agent...")
        purple_agent_process.terminate()
        purple_agent_process.join(timeout=5)
    
    # Start fresh agent
    print("Starting fresh purple agent...")
    purple_agent_process = multiprocessing.Process(
        target=start_purple_agent,
        args=("smarthome_purple_agent", "localhost", 9000, "http://localhost:9006")
    )
    purple_agent_process.start()
    
    # Wait for agent to be ready
    await asyncio.sleep(3)
    
    return {
        "status": "success",
        "message": "Purple agent reset and restarted",
        "agent_url": "http://localhost:9000"
    }


@launcher_app.get("/status")
async def get_status():
    """Get launcher and agent status."""
    global purple_agent_process
    
    agent_running = purple_agent_process is not None and purple_agent_process.is_alive()
    
    return {
        "launcher": "running",
        "agent_running": agent_running,
        "agent_url": "http://localhost:9000",
        "mcp_server": "http://localhost:9006"
    }


@launcher_app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "purple_agent_launcher"}


def start_purple_launcher(host="localhost", port=9010):
    """
    Start the purple agent launcher.
    
    This launcher is required for AgentBeats.org integration.
    
    Args:
        host: Host to bind
        port: Port to bind (default: 9010)
    """
    print("=" * 70)
    print("Starting Purple Agent Launcher for AgentBeats.org")
    print("=" * 70)
    print(f"Launcher URL: http://{host}:{port}")
    print(f"Agent URL: http://{host}:9000")
    print(f"MCP Server: http://localhost:9006")
    print("=" * 70)
    print("\nRegister at agentbeats.org:")
    print(f"  Agent URL:    http://YOUR_PUBLIC_IP:9000")
    print(f"  Launcher URL: http://YOUR_PUBLIC_IP:{port}")
    print("=" * 70)
    
    uvicorn.run(launcher_app, host=host, port=port)


if __name__ == "__main__":
    start_purple_launcher()

