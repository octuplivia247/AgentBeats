# HomeBench Green Agent

Evaluator agent for the HomeBench benchmark with MCP tool integration.

## Features

- **A2A Protocol**: Agent-to-Agent communication
- **MCP Tools**: Structured evaluation via HTTP tool calls
- **Metrics**: Exact Match, Precision, Recall, F1

## Architecture

```
Docker Container
├── A2A Server (port 9009) ─── Agent evaluates purple agents
└── MCP Server (port 9006) ─── Metrics computation tools
```

## Usage

```bash
# Local
uv sync
uv run src/server.py

# Docker
docker build -t homebench-green-agent .
docker run -p 9009:9009 -p 9006:9006 homebench-green-agent
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `compute_accuracy_metrics` | Compute EM, P, R, F1 |
| `parse_operations_from_response` | Extract operations from text |
| `compute_batch_metrics` | Aggregate metrics |

```bash
curl -X POST http://localhost:9006/tools/call \
  -H "Content-Type: application/json" \
  -d '{"name": "compute_accuracy_metrics", "arguments": {"predictions": [...], "expected": [...]}}'
```

## Request Format

```json
{
  "participants": {"purple_agent": "http://localhost:8000"},
  "config": {"dataset_path": "data/tasks.jsonl", "timeout_seconds": 60}
}
```
