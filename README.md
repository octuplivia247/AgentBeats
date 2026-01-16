# BenchPress Green Agent: A2A Evaluation Framework for Smart Home Agents

An evaluator agent for the [HomeBench](https://github.com/octuplivia247/AgentBeats) smart home benchmark. 

## Overview of the BenchPress Benchmark
This Green Agent implements an evaluation framework for the HomeBench dataset, originally introduced by Li et al. (2025). HomeBench is designed to evaluate LLMs on smart home control tasks that include both valid and invalid instructions across single and multiple devices. Our implementation adapts this benchmark to the A2A protocol, allowing automated evaluation of "purple agents" (smart home assistants under test).

## HomeBench Dataset
The original HomeBench benchmark (arXiv:2505.19628) evaluates LLMs on five main task categories:

Valid Single (VS): Valid instructions for single device operation
Valid Multiple (VM): Valid instructions requiring multiple device operations
Invalid Single (IS): Invalid or ambiguous instructions for single devices that should be rejected
Invalid Multiple (IM): Invalid instructions involving multiple devices
Mix Multiple (MM): Mixed scenarios combining valid and invalid operations

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Scoring](#scoring)
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


# Quick Start

## Prerequisites

- Python 3.13+
- [uv](https://github.com/astral-sh/uv) package manager
- Docker (optional, for containerized deployment)

## Local Development

```bash
cd green-agent  # Clone and navigate
uv sync  # Install dependencies
uv run src/server.py  # Start the agent (both A2A and MCP servers)

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

# Scoring

| Metric | Description |
|--------|-------------|
| **Exact Match (EM)** | 1.0 if predicted operations exactly match expected, else 0.0 |
| **Precision** | Fraction of predicted operations that are correct |
| **Recall** | Fraction of expected operations that were predicted |
| **F1** | Harmonic mean of precision and recall |


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

### Example

```
Expected:  ["light.on()", "thermostat.set(72)"]
Predicted: ["light.on()", "fan.off()"]

EM = 0.0 (not exact match)
Precision = 1/2 = 0.5 (1 correct out of 2 predicted)
Recall = 1/2 = 0.5 (1 correct out of 2 expected)
F1 = 0.5
```

## Testing

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

Install and configure buildx:

```bash
brew install docker-buildx
mkdir -p ~/.docker/cli-plugins
ln -sfn $(brew --prefix)/opt/docker-buildx/bin/docker-buildx ~/.docker/cli-plugins/docker-buildx
```

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
