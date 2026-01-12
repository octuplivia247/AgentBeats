# HomeBench Green Agent

An evaluator agent for the [HomeBench](https://github.com/octuplivia247/AgentBeats) smart home benchmark. This agent orchestrates evaluation of "purple" agents (the agents being tested) by sending them smart home tasks and scoring their responses using standardized metrics.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [MCP Tools](#mcp-tools)
- [API Reference](#api-reference)
- [Testing](#testing)
- [Docker](#docker)
- [CI/CD](#cicd)
- [Project Structure](#project-structure)

---

## Overview

The Green Agent is an **evaluator** in the HomeBench benchmark system. It:

1. Receives evaluation requests specifying a purple agent URL and task configuration
2. Loads smart home tasks from JSONL datasets
3. Sends each task to the purple agent via the [A2A protocol](https://github.com/a2aproject/a2a-protocol)
4. Parses the purple agent's responses to extract device operations
5. Computes accuracy metrics (Exact Match, Precision, Recall, F1)
6. Returns a detailed evaluation report

**Important**: The Green Agent does NOT use any LLM directly. It evaluates agents that do.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Docker Container                             │
│                                                                  │
│  ┌─────────────────────────┐    ┌─────────────────────────────┐ │
│  │     A2A Server          │    │       MCP Server            │ │
│  │     (Port 9009)         │    │       (Port 9006)           │ │
│  │                         │    │                             │ │
│  │  • Receives eval reqs   │    │  • compute_accuracy_metrics │ │
│  │  • Talks to purple      │◄──►│  • parse_operations         │ │
│  │  • Returns reports      │    │  • compute_batch_metrics    │ │
│  └─────────────────────────┘    └─────────────────────────────┘ │
│              │                                                   │
└──────────────┼───────────────────────────────────────────────────┘
               │
               ▼
    ┌─────────────────────┐
    │   Purple Agent      │
    │   (Being Tested)    │
    │                     │
    │  Uses LLM to        │
    │  generate device    │
    │  operations         │
    └─────────────────────┘
```

### Components

| Component | File | Description |
|-----------|------|-------------|
| **Server** | `src/server.py` | Entry point. Starts both A2A and MCP servers |
| **Executor** | `src/executor.py` | A2A request handler and task lifecycle manager |
| **Agent** | `src/agent.py` | Core evaluation logic |
| **Messenger** | `src/messenger.py` | A2A client for talking to purple agents |
| **MCP Server** | `src/mcp_server.py` | HTTP API exposing evaluation tools |
| **MCP Client** | `src/mcp_client.py` | Client for calling MCP tools |
| **Metrics** | `src/metrics_calculator.py` | EM, Precision, Recall, F1 computation |
| **Models** | `src/models.py` | Pydantic models for MCP requests/responses |

---

## Quick Start

### Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker (optional, for containerized deployment)

### Local Development

```bash
# Clone and navigate
cd green-agent

# Install dependencies
uv sync

# Start the agent (both A2A and MCP servers)
uv run src/server.py

# The agent is now running:
# - A2A Server: http://localhost:9009
# - MCP Server: http://localhost:9006
```

### Verify It's Running

```bash
# Check A2A agent card
curl http://localhost:9009/.well-known/agent-card.json | jq .

# Check MCP health
curl http://localhost:9006/health

# List MCP tools
curl http://localhost:9006/tools | jq .
```

---

## Configuration

### Command Line Arguments

```bash
uv run src/server.py [OPTIONS]

Options:
  --host        A2A server host (default: 0.0.0.0)
  --port        A2A server port (default: 9009)
  --card-url    Override agent card URL
  --mcp-host    MCP server host (default: 0.0.0.0)
  --mcp-port    MCP server port (default: 9006)
  --no-mcp      Disable MCP server
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MCP_SERVER_URL` | `http://localhost:9006` | MCP server URL (auto-set when using `server.py`) |

### Evaluation Request Format

Send this JSON to the A2A server:

```json
{
  "participants": {
    "purple_agent": "http://localhost:8000"
  },
  "config": {
    "dataset_path": "data/tasks.jsonl",
    "timeout_seconds": 60,
    "task_ids": [0, 1, 2]
  }
}
```

| Field | Required | Description |
|-------|----------|-------------|
| `participants.purple_agent` | Yes | URL of the agent being evaluated |
| `config.dataset_path` | No | Path to JSONL task file |
| `config.timeout_seconds` | No | Per-task timeout (default: 60) |
| `config.task_ids` | No | Specific task indices to run |
| `config.tasks` | No | Inline task definitions |

### Task Format (JSONL)

Each line in the dataset file should be:

```json
{
  "task_id": "task_001",
  "instruction": "Turn on the living room light",
  "expected_operations": ["living_room.light.turn_on()"],
  "category": "valid_single",
  "home_id": 1
}
```

---

## MCP Tools

The MCP server exposes these tools via HTTP:

| Tool | Description | Arguments |
|------|-------------|-----------|
| `compute_accuracy_metrics` | Compute EM, P, R, F1 | `predictions: list[str]`, `expected: list[str]` |
| `parse_operations_from_response` | Extract operations from text | `response: str` |
| `compute_batch_metrics` | Aggregate metrics across tasks | `results: list[dict]` |
| `evaluate_task_completion` | Evaluate single task | `evaluation_input: dict` |
| `evaluate_homebench_task` | Parse HomeBench triple-quote format | `task_data: dict` |
| `log_device_action` | Log a device action | `device_name`, `action`, `parameters`, `success`, `error` |
| `get_action_logs` | Get all logged actions | (none) |
| `clear_action_logs` | Clear action logs | (none) |

### Example: Compute Metrics

```bash
curl -X POST http://localhost:9006/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "compute_accuracy_metrics",
    "arguments": {
      "predictions": ["living_room.light.turn_on()"],
      "expected": ["living_room.light.turn_on()"]
    }
  }'
```

Response:

```json
{
  "result": {
    "exact_match": 1.0,
    "precision": 1.0,
    "recall": 1.0,
    "f1": 1.0
  }
}
```

---

## API Reference

### A2A Endpoints (Port 9009)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/.well-known/agent-card.json` | GET | Agent card (capabilities, skills) |
| `/` | POST | Send evaluation request |

### MCP Endpoints (Port 9006)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Server info and tool list |
| `/health` | GET | Health check |
| `/tools` | GET | List all tools with descriptions |
| `/tools/call` | POST | Call a tool |

---

## Testing

### Run Tests Locally

The tests require a running agent:

```bash
# Terminal 1: Start the agent
uv run src/server.py

# Terminal 2: Run tests
cd green-agent
uv sync --extra test
uv run pytest -v --agent-url http://localhost:9009
```

### Run Tests Against Docker

```bash
# Build and start container
docker build -t homebench-green-agent .
docker run -d -p 9009:9009 -p 9006:9006 --name green-agent homebench-green-agent \
  --host 0.0.0.0 --port 9009 --mcp-host 0.0.0.0 --mcp-port 9006

# Run tests
uv run pytest -v --agent-url http://localhost:9009

# Cleanup
docker stop green-agent && docker rm green-agent
```

### Test Coverage

| Test | Description |
|------|-------------|
| `test_agent_card` | Validates agent card structure |
| `test_agent_card_has_evaluation_skill` | Checks for required skill |
| `test_message_format` | Validates A2A message format (streaming + non-streaming) |
| `test_invalid_request_rejected` | Invalid JSON is rejected |
| `test_missing_participant_rejected` | Missing purple_agent is rejected |

---

## Docker

### Build

```bash
# Requires BuildKit (for cache mounts)
docker buildx build -t homebench-green-agent .

# Or with DOCKER_BUILDKIT
DOCKER_BUILDKIT=1 docker build -t homebench-green-agent .
```

### Run

```bash
docker run -p 9009:9009 -p 9006:9006 homebench-green-agent \
  --host 0.0.0.0 --port 9009 --mcp-host 0.0.0.0 --mcp-port 9006
```

### Run in Background

```bash
docker run -d \
  -p 9009:9009 \
  -p 9006:9006 \
  --name green-agent \
  homebench-green-agent \
  --host 0.0.0.0 --port 9009 --mcp-host 0.0.0.0 --mcp-port 9006

# View logs
docker logs -f green-agent

# Stop
docker stop green-agent && docker rm green-agent
```

### Push to Docker Hub

```bash
docker login
docker tag homebench-green-agent YOUR_USERNAME/green-agent:latest
docker push YOUR_USERNAME/green-agent:latest
```

---

## CI/CD

The GitHub Actions workflow (`.github/workflows/test-and-publish.yml`) automatically:

1. **On every PR**: Builds, starts, and tests the agent
2. **On push to main**: Tests + publishes to GitHub Container Registry
3. **On version tags (`v*`)**: Tests + publishes with version tags

### Required Secrets

| Secret | Required | Description |
|--------|----------|-------------|
| `GITHUB_TOKEN` | Auto | Provided by GitHub Actions |
| `OPENAI_API_KEY` | Only if testing with real purple agents | Passed to container |

### Pull the Published Image

```bash
docker pull ghcr.io/octuplivia247/agentbeats:latest
```

---

## Project Structure

```
green-agent/
├── src/
│   ├── server.py           # Entry point - starts A2A + MCP servers
│   ├── executor.py         # A2A request handler
│   ├── agent.py            # Core evaluation logic
│   ├── messenger.py        # A2A client for purple agents
│   ├── mcp_server.py       # MCP HTTP server with tools
│   ├── mcp_client.py       # MCP client
│   ├── metrics_calculator.py # Metric computation
│   └── models.py           # Pydantic models
├── tests/
│   ├── conftest.py         # Pytest fixtures
│   └── test_agent.py       # A2A conformance tests
├── Dockerfile
├── pyproject.toml
├── uv.lock
└── README.md
```

---

## Metrics Explained

| Metric | Description |
|--------|-------------|
| **Exact Match (EM)** | 1.0 if predicted operations exactly match expected, else 0.0 |
| **Precision** | Fraction of predicted operations that are correct |
| **Recall** | Fraction of expected operations that were predicted |
| **F1** | Harmonic mean of precision and recall |

### Example

```
Expected:  ["light.on()", "thermostat.set(72)"]
Predicted: ["light.on()", "fan.off()"]

EM = 0.0 (not exact match)
Precision = 1/2 = 0.5 (1 correct out of 2 predicted)
Recall = 1/2 = 0.5 (1 correct out of 2 expected)
F1 = 0.5
```

---

## Troubleshooting

### "BuildKit is missing"

Install and configure buildx:

```bash
brew install docker-buildx
mkdir -p ~/.docker/cli-plugins
ln -sfn $(brew --prefix)/opt/docker-buildx/bin/docker-buildx ~/.docker/cli-plugins/docker-buildx
```

### "Agent not responding"

Check both servers are running:

```bash
curl http://localhost:9009/.well-known/agent-card.json
curl http://localhost:9006/health
```

### Container logs

```bash
docker logs green-agent
```

---

## License

See the main [AgentBeats](https://github.com/octuplivia247/AgentBeats) repository for license information.
