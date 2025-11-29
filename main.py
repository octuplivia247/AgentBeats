import typer
import asyncio

from src.green_agent import start_green_agent
from src.white_agent import start_white_agent
from src.launcher import launch_evaluation_refactored, quick_test

app = typer.Typer(help="Agentified A2A Smart Home Assessment")


@app.command()
def green():
    """Start the green agent (assessment manager)."""
    start_green_agent()


@app.command()
def white():
    """Start the white agent (target being tested)."""
    start_white_agent()



@app.command()
def launch_refactored():
    """Launch the refactored evaluation workflow (Purple Agent + MCP)."""
    asyncio.run(launch_evaluation_refactored())


@app.command()
def purple(
    host: str = typer.Option("localhost", help="Host to bind the agent server"),
    port: int = typer.Option(9000, help="Port to bind the agent server"),
    mcp_url: str = typer.Option("http://localhost:9006", help="MCP server URL")
):
    """Start the purple agent (task executor with MCP tools)."""
    from src.purple_agent import start_purple_agent
    start_purple_agent(host=host, port=port, mcp_url=mcp_url)


@app.command()
def purple_launcher(
    port: int = typer.Option(9010, help="Launcher port for AgentBeats.org integration")
):
    """Start the purple agent launcher for AgentBeats.org integration."""
    from src.purple_agent.launcher import start_purple_launcher
    start_purple_launcher(port=port)


@app.command()
def mcp():
    """Start the MCP server (smart home tools)."""
    from src.green_agent.mcp.mcp_server import http_app
    import uvicorn
    print("Starting MCP Server on port 9006...")
    uvicorn.run(http_app, host="localhost", port=9006)


@app.command()
def test_purple():
    """Quick test of purple agent with MCP tools."""
    asyncio.run(quick_test())


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


@app.command()
def run_scenario(
    scenario: str = typer.Argument("scenarios/smarthome/scenario.toml", help="Path to scenario TOML file"),
    show_logs: bool = typer.Option(False, "--show-logs", help="Show agent outputs during assessment"),
    serve_only: bool = typer.Option(False, "--serve-only", help="Start agents without running assessment")
):
    """Run a scenario following official AgentBeats patterns."""
    print(f"Running scenario: {scenario}")
    
    if serve_only:
        print("Serve-only mode: Starting agents...")
    
    if show_logs:
        import os
        os.environ["SHOW_AGENT_LOGS"] = "true"
    
    # Use our launcher which follows the same pattern
    asyncio.run(launch_evaluation_refactored())


if __name__ == "__main__":
    app()