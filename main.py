import typer
import asyncio

app = typer.Typer(help="Agentified A2A Smart Home Assessment")


@app.command()
def green(
    host: str = typer.Option("localhost", help="Host to bind the agent server"),
    port: int = typer.Option(9001, help="Port to bind the agent server"),
    mcp_url: str = typer.Option("http://localhost:9006", help="MCP server URL")
):
    """Start the green agent (evaluation orchestrator)."""
    from src.green_agent import start_green_agent
    start_green_agent(host=host, port=port, mcp_url=mcp_url)


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
def mcp(
    host: str = typer.Option("localhost", help="Host to bind the server"),
    port: int = typer.Option(9006, help="Port to bind the server")
):
    """Start the MCP server (smart home tools)."""
    from src.green_agent.mcp.mcp_server import http_app
    import uvicorn
    print(f"Starting MCP Server on http://{host}:{port}...")
    uvicorn.run(http_app, host=host, port=port)


@app.command()
def launch():
    """Launch all components (MCP + Purple + Green) and run evaluation."""
    from src.launcher import launch_evaluation_refactored
    asyncio.run(launch_evaluation_refactored())


@app.command()
def test_purple():
    """Quick test of purple agent with MCP tools."""
    from src.launcher import quick_test
    asyncio.run(quick_test())


@app.command()
def run(
    config_file: str = typer.Argument(..., help="Path to the agent configuration TOML file"),
    mcp: str = typer.Option(None, help="MCP server URL (e.g., http://localhost:9006)"),
    host: str = typer.Option("localhost", help="Host to bind the agent server"),
    port: int = typer.Option(9001, help="Port to bind the agent server")
):
    """Run an agent with a specific configuration file and optional MCP server."""
    from src.green_agent import start_green_agent
    import os
    
    agent_name = os.path.basename(config_file).replace('.toml', '')
    
    print(f"Running agent with config: {config_file}")
    print(f"Agent name: {agent_name}")
    if mcp:
        mcp_url = mcp.rstrip('/').replace('/sse', '')
        print(f"Using MCP server: {mcp_url}")
    else:
        mcp_url = "http://localhost:9006"
    
    start_green_agent(agent_name=agent_name, host=host, port=port, mcp_url=mcp_url)


@app.command()
def run_scenario(
    scenario: str = typer.Argument("scenarios/smarthome/scenario.toml", help="Path to scenario TOML file"),
    show_logs: bool = typer.Option(False, "--show-logs", help="Show agent outputs during assessment"),
    serve_only: bool = typer.Option(False, "--serve-only", help="Start agents without running assessment")
):
    """Run a scenario following official AgentBeats patterns."""
    from src.launcher import launch_evaluation_refactored
    
    print(f"Running scenario: {scenario}")
    
    if serve_only:
        print("Serve-only mode: Starting agents...")
    
    if show_logs:
        import os
        os.environ["SHOW_AGENT_LOGS"] = "true"
    
    asyncio.run(launch_evaluation_refactored())


if __name__ == "__main__":
    app()
