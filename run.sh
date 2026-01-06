#!/bin/bash
# AgentBeats Controller - Agent Startup Script
# This script is called by the AgentBeats Controller to start your agent

set -e  # Exit on error

echo "================================================"
echo "Starting Purple Agent for AgentBeats Platform"
echo "================================================"

# Load environment variables
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "✓ Environment variables loaded from .env"
fi

# Check for required environment variables
if [ -z "$OPENAI_API_KEY" ]; then
    echo "❌ Error: OPENAI_API_KEY not set!"
    echo "Please set it in .env file or export it:"
    echo "  export OPENAI_API_KEY=your-key-here"
    exit 1
fi

# Ensure MCP server is running
echo "Checking MCP server..."
MCP_RUNNING=$(curl -s http://localhost:9006/health 2>/dev/null || echo "")
if [ -z "$MCP_RUNNING" ]; then
    echo "Starting MCP server in background..."
    nohup python -m src.green_agent.mcp.mcp_server > logs/mcp_server.log 2>&1 &
    MCP_PID=$!
    echo $MCP_PID > .mcp_server.pid
    sleep 3
    echo "✓ MCP server started (PID: $MCP_PID)"
else
    echo "✓ MCP server already running"
fi

# Start Purple Agent
echo "Starting Purple Agent on port 9000..."
echo "MCP Server: http://localhost:9006"
echo "Agent URL: http://localhost:9000"
echo "================================================"

# Run the agent
python main.py purple --host 0.0.0.0 --port 9000

# Note: Controller will manage this process
# If agent crashes, controller will detect and restart using this script

