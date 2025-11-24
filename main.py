"""CLI entry point for BenchPress agent!!"""

import typer
import asyncio

from src.green_agent import start_green_agent
from src.white_agent import start_white_agent
from src.launcher import launch_evaluation

app = typer.Typer(help="Agentified BenchPress - A2A Smart Home Assessment")


@app.command()
def green():
    """Start the green agent (assessment manager)."""
    start_green_agent()


@app.command()
def white():
    """Start the white agent (target being tested)."""
    start_white_agent()


@app.command()
def launch():
    """Launch the complete evaluation workflow."""
    asyncio.run(launch_evaluation())


@app.command()
def run(
    config_file: str = typer.Argument(..., help="Path to the agent configuration TOML file"),
    mcp: str = typer.Option(None, help="MCP server URL (e.g., http://localhost:9006)"),
    host: str = typer.Option("localhost", help="Host to bind the agent server"),
    port: int = typer.Option(9001, help="Port to bind the agent server")
):
    """Run an agent with a specific configuration file and optional MCP server."""
    # Extract agent name from config file path
    # e.g., "src/green_agent/SmartHome_green_agent_mcp.toml" -> "SmartHome_green_agent_mcp"
    import os
    agent_name = os.path.basename(config_file).replace('.toml', '')
    
    print(f"Running agent with config: {config_file}")
    print(f"Agent name: {agent_name}")
    if mcp:
        # Remove /sse suffix if present, as we'll use direct HTTP calls
        mcp_url = mcp.rstrip('/').replace('/sse', '')
        print(f"Using MCP server: {mcp_url}")
    else:
        mcp_url = None
    
    # Start the green agent with the specified configuration
    start_green_agent(agent_name=agent_name, host=host, port=port, mcp_url=mcp_url)


if __name__ == "__main__":
    app()