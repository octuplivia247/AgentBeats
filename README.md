# AgentBeats Hackathon - Smart Home Evaluation

Multi-agent evaluation framework for testing LLM-based smart home agents using A2A and MCP protocols.

## Quickstart

### Running Locally
```bash
./run_local.sh
```

### Running in Docker
```bash
./run_docker.sh
```

Replace `your-registry/agentbeats-green:v1` with your published Docker image.

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

## Scoring & Metrics

The green agent evaluates purple agents on smart home tasks using the following metrics:

- **Exact Match**: Binary score for perfect operation sequence matching
- **Precision**: Ratio of correct operations to total predicted operations
- **Recall**: Ratio of correct operations to total expected operations
- **F1 Score**: Harmonic mean of precision and recall

Scoring output is JSON printed to stdout:

```json
{
  "task_id": "smarthome_task_1",
  "exact_match": 0.0,
  "precision": 0.8,
  "recall": 0.9,
  "f1": 0.85,
  "total_steps": 5
}
```

## Baseline Purple Agent(s)

The project includes a baseline purple agent implemented in `src/purple_agent/`. This agent uses MCP tools to interact with the smart home environment.

To run the baseline agent:

```bash
python main.py purple --host localhost --port 9000 --mcp-url http://localhost:9006
```

## Troubleshooting

### Common Issues

- **Port conflicts**: Ensure ports 8080, 9000, 9001, 9006 are available
- **Missing API keys**: Set OPENAI_API_KEY in .env file
- **Docker build fails**: Ensure Docker has access to the context

### Error Messages

- `Connection refused`: Check if all services are running
- `Invalid operations`: Purple agent may be sending malformed commands

## Reproducibility

Results are deterministic for the same inputs. The baseline purple agent uses OpenAI GPT models; to ensure reproducibility:

- Set a fixed random seed in the LLM calls (if supported)
- Use the same API key and model version
- Run evaluations in sequence, not parallel

Example runs are provided in `examples/` with expected outputs.

## Contact

For questions or issues, please open an issue on GitHub.
