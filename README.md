# AgentBeats Hackathon - Smart Home Evaluation

Multi-agent evaluation framework for testing LLM-based smart home agents using A2A and MCP protocols.

## ⚡ Quick Demo (30 Seconds)

```bash
# 1. Setup
cp sample.env .env && echo "OPENAI_API_KEY=sk-proj-your-key" >> .env

# 2. Install
pip install -e ".[dev]"

# 3. Demo
agentbeats launch
```

## Architecture

```
┌─────────────┐
│    User     │  Sends evaluation requests
└──────┬──────┘
       │ A2A Protocol
┌──────▼───────────────────┐
│   Green Agent            │  Evaluation Orchestrator
│   Port: 9001             │  - Assigns tasks via A2A
│                          │  - Monitors via MCP
└──────┬──────────┬────────┘
       │ A2A      │ HTTP (monitoring)
┌──────▼──────────▼───────┐
│   Purple Agent           │  Task Executor
│   Port: 9000             │  - Uses MCP tools
│                          │  - LLM-powered reasoning
└──────┬─────────────────┘
       │ HTTP (MCP tool calls)
┌──────▼─────────────────┐
│   MCP Server            │  Smart Home Tools
│   Port: 9006            │  - Device control
│                         │  - Monitoring & metrics
└─────────────────────────┘
```

## 🚀 Quick Start (Official AgentBeats Way)

### Prerequisites
- Python(3.13+ recommended for Earthshaker)
- OpenAI API key 

### Installation & Demo

```bash
# 1. Setup environment
cp sample.env .env
# Edit .env and add your OPENAI_API_KEY

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Run demo
agentbeats launch
```

**That's it!** Watch the system:
- ✅ Start 3 agents (Green, Purple, MCP)
- ✅ Purple Agent calls MCP device control tools
- ✅ Green Agent orchestrates via A2A protocol
- ✅ Quantitative metrics computed (~30 seconds)


### Run System

```bash
python main.py launch

# Option 2: Start components individually
# Terminal 1
python main.py mcp

# Terminal 2  
python main.py purple

# Terminal 3
python main.py green --mcp http://localhost:9006
```

### Test

```bash
# Run integration tests
pytest test_refactored_architecture.py -v

# Quick smoke test
python main.py test-purple
```

## CLI Commands

```bash
# Main commands (use 'agentbeats' CLI)
agentbeats launch      
agentbeats mcp                   
agentbeats purple                 
agentbeats green                  
agentbeats purple-launcher        
agentbeats run-scenario scenarios/smarthome/scenario.toml  # Official pattern

# Development commands
make test                         # Run tests
make format                       # Format code
make lint                         # Run linters
make install-dev                  # Install dev dependencies
```

## Project Structure

```
AgentBeats2/
├── src/
│   ├── purple_agent/           # Task executor with MCP
│   │   ├── agent.py
│   │   └── mcp_client.py
│   ├── green_agent/            # Evaluation orchestrator
│   │   ├── agent.py
│   │   ├── core/               # Evaluation logic
│   │   └── mcp/                # MCP server & tools
│   ├── white_agent/            # Legacy (deprecated)
│   └── utils/                  # Shared utilities
├── data/                       # Test datasets
├── main.py                     # CLI entry point
└── test_refactored_architecture.py
```

## Key Components

### Green Agent (Evaluator)
- Receives evaluation requests via A2A
- Initializes smart home environment
- Assigns tasks to Purple Agent
- Monitors actions and computes metrics
- **Does NOT execute device control** (only monitors)

### Purple Agent (Executor)  
- Receives tasks via A2A
- Uses LLM (GPT-4) for reasoning
- **Calls MCP tools to control devices**
- Returns execution results

### MCP Server
- Provides device control tools
- Provides evaluation tools
- Logs all actions for monitoring
- Manages environment state

## MCP Tools

### Device Control (for Purple Agent)
- `control_device` - Control smart home devices
- `get_device_state` - Query device status
- `list_devices` - List available devices

### Evaluation (for Green Agent)
- `initialize_smart_home` - Set up environment
- `monitor_purple_agent_actions` - Track actions
- `evaluate_task_completion` - Assess results
- `compute_accuracy_metrics` - Calculate metrics

## Configuration

Agent configurations are in TOML files:
- `src/purple_agent/smarthome_purple_agent.toml`
- `src/green_agent/SmartHome_green_agent_mcp.toml`


## Architecture Decisions

### Why Purple Agent Uses MCP?
Following A2A protocol best practices:
- **Separation of concerns**: Purple executes, Green orchestrates
- **Scalability**: Can add more purple agents
- **Testability**: Each component isolated
- **Maintainability**: Clear responsibilities

### Why Not Green Agent?
Green Agent is the evaluator - it should **observe**, not **execute**. Using MCP for monitoring only maintains proper separation.

## Troubleshooting

### Port Already in Use
```bash
lsof -ti:9006 | xargs kill -9  # MCP Server
lsof -ti:9000 | xargs kill -9  # Purple Agent
lsof -ti:9001 | xargs kill -9  # Green Agent
```

### Import Errors
```bash
pip install --upgrade -r requirements.txt
```

### OpenAI API Key Not Set
```bash
export OPENAI_API_KEY=your-key-here
# or create .env file
echo "OPENAI_API_KEY=your-key-here" > .env
```

## Testing

```bash
# Run tests
pytest test_refactored_architecture.py -v

# With coverage
pytest --cov=src test_refactored_architecture.py

# Or using make
make test
```
