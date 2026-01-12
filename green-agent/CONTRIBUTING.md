# Contributing to HomeBench Green Agent

This guide is for team members taking over development while the primary maintainer is away. It covers the codebase architecture, development workflow, and common tasks.

---

## Quick Reference

| What | Command |
|------|---------|
| Install deps | `uv sync` |
| Start locally | `uv run src/server.py` |
| Run tests | `uv run pytest -v --agent-url http://localhost:9009` |
| Build Docker | `docker buildx build -t homebench-green-agent .` |
| Run Docker | `docker run -p 9009:9009 -p 9006:9006 homebench-green-agent --host 0.0.0.0 --port 9009 --mcp-host 0.0.0.0 --mcp-port 9006` |

---

## Codebase Overview

### What This Agent Does

The **Green Agent** is an evaluator. It does NOT generate smart home commands itself — it **tests other agents** (called "purple agents") that do.

```
Green Agent                    Purple Agent (being tested)
    │                                   │
    ├── Load tasks from JSONL ─────────►│
    │                                   │
    │◄── Receive predicted operations ──┤
    │                                   │
    ├── Compute accuracy metrics        │
    │                                   │
    └── Return evaluation report        │
```

### Key Files

| File | Purpose | When to Modify |
|------|---------|----------------|
| `src/agent.py` | Core evaluation logic | Adding new evaluation features |
| `src/server.py` | Entry point, starts both servers | Changing startup behavior |
| `src/mcp_server.py` | MCP tools (metrics, parsing) | Adding new tools |
| `src/messenger.py` | A2A client for purple agents | Changing how we talk to agents |
| `src/metrics_calculator.py` | EM, P, R, F1 computation | Changing metric logic |
| `src/executor.py` | A2A request handler | Changing request lifecycle |

### Data Flow

```
1. Request comes in via A2A (port 9009)
       │
       ▼
2. executor.py creates task, instantiates Agent
       │
       ▼
3. agent.py validates request, loads tasks
       │
       ▼
4. For each task:
   a. messenger.py sends task to purple agent
   b. Parse response (uses mcp_client → mcp_server)
   c. Compute score
       │
       ▼
5. Generate report, return via A2A
```

---

## Development Setup

### Prerequisites

```bash
# Install uv (Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install Docker buildx (for container builds)
brew install docker-buildx
mkdir -p ~/.docker/cli-plugins
ln -sfn $(brew --prefix)/opt/docker-buildx/bin/docker-buildx ~/.docker/cli-plugins/docker-buildx
```

### First Time Setup

```bash
cd green-agent
uv sync                    # Install dependencies
uv sync --extra test       # Install test dependencies
```

### Running Locally

```bash
# Start the agent
uv run src/server.py

# In another terminal, verify
curl http://localhost:9009/.well-known/agent-card.json | jq .
curl http://localhost:9006/health
```

---

## Testing

### Run All Tests

```bash
# Make sure agent is running first!
uv run src/server.py &

# Run tests
uv run pytest -v --agent-url http://localhost:9009
```

### Test Against Docker

```bash
# Build
docker buildx build -t homebench-green-agent .

# Run
docker run -d -p 9009:9009 -p 9006:9006 --name test-agent homebench-green-agent \
  --host 0.0.0.0 --port 9009 --mcp-host 0.0.0.0 --mcp-port 9006

# Wait for startup
sleep 5

# Test
uv run pytest -v --agent-url http://localhost:9009

# Cleanup
docker stop test-agent && docker rm test-agent
```

### What Tests Cover

| Test | What It Validates |
|------|-------------------|
| `test_agent_card` | Agent card has all required A2A fields |
| `test_agent_card_has_evaluation_skill` | Agent declares the `evaluate_smart_home_agent` skill |
| `test_message_format` | Responses are valid A2A format (streaming + non-streaming) |
| `test_invalid_request_rejected` | Invalid JSON is properly rejected |
| `test_missing_participant_rejected` | Missing `purple_agent` URL is rejected |

---

## Common Tasks

### Adding a New MCP Tool

1. Edit `src/mcp_server.py`
2. Add your function with the `@register_tool` decorator:

```python
@register_tool
def my_new_tool(arg1: str, arg2: int) -> dict[str, Any]:
    """Description of what this tool does."""
    # Your logic here
    return {"result": "value"}
```

3. (Optional) Add client method in `src/mcp_client.py`:

```python
async def my_new_tool(self, arg1: str, arg2: int) -> dict[str, Any]:
    return await self.call_tool("my_new_tool", {"arg1": arg1, "arg2": arg2})
```

### Changing Evaluation Logic

Edit `src/agent.py`:

- `_load_tasks()` — How tasks are loaded
- `_evaluate_task()` — Per-task evaluation
- `_parse_ops()` — Extract operations from responses
- `_compute_score()` — Scoring logic
- `_compute_results()` — Aggregate results
- `_generate_report()` — Report formatting

### Adding a New Test

Edit `tests/test_agent.py`:

```python
@pytest.mark.asyncio
async def test_my_new_feature(agent):
    """Test description."""
    events = await send_text_message('{"participants": ...}', agent)
    # Assertions here
```

### Updating Dependencies

```bash
# Add a new dependency
uv add some-package

# Update lockfile
uv lock

# Sync
uv sync
```

---

## CI/CD Pipeline

### How It Works

The workflow at `.github/workflows/test-and-publish.yml`:

1. **Build** Docker image
2. **Start** container with all secrets as env vars
3. **Wait** for A2A and MCP servers
4. **Smoke test** MCP tools
5. **Run** pytest suite
6. **Push** to ghcr.io (only on main/tags, not PRs)

### Triggering CI

| Event | What Happens |
|-------|--------------|
| Open PR | Build + Test (no publish) |
| Push to `main` | Build + Test + Publish as `latest` |
| Push tag `v1.2.3` | Build + Test + Publish as `1.2.3` and `1` |

### Adding Secrets

Go to: Repository → Settings → Secrets and variables → Actions → New repository secret

Current secrets needed:
- `GITHUB_TOKEN` — Automatic, no action needed
- Any secrets your agent needs — Automatically passed to container

---

## Debugging

### View Container Logs

```bash
docker logs green-agent
docker logs -f green-agent  # Follow
```

### Check Server Health

```bash
# A2A
curl http://localhost:9009/.well-known/agent-card.json

# MCP
curl http://localhost:9006/health
curl http://localhost:9006/tools | jq '.tools[].name'
```

### Test MCP Tool Manually

```bash
curl -X POST http://localhost:9006/tools/call \
  -H "Content-Type: application/json" \
  -d '{
    "name": "compute_accuracy_metrics",
    "arguments": {
      "predictions": ["light.on()"],
      "expected": ["light.on()"]
    }
  }' | jq .
```

### Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `BuildKit is missing` | Docker buildx not installed | See Prerequisites |
| `Agent not responding` | Server not started | Start with `uv run src/server.py` |
| `Missing roles: purple_agent` | Invalid request format | Include `purple_agent` URL |
| `Port already in use` | Previous instance running | `docker stop` or kill process |

---

## Architecture Decisions

### Why Two Servers?

- **A2A (9009)**: Standard agent protocol for the AgentBeats platform
- **MCP (9006)**: Tool-based API for metrics computation

This separation allows:
- MCP tools to be used independently
- Easier testing of metrics logic
- Future extensibility

### Why No LLM in Green Agent?

The Green Agent is an **evaluator**, not a participant. It:
- Doesn't need to understand natural language
- Uses regex to parse operations
- Computes metrics deterministically

The **Purple Agent** (being tested) uses an LLM.

### Why uv?

- Faster than pip/poetry
- Deterministic lockfile
- Works well in Docker with cache mounts

---

## Contacts

| Role | Contact |
|------|---------|
| Primary Maintainer | Gunish Matta (OOO until [date]) |
| Team | [Your team contact] |

---

## Files You Probably Don't Need to Touch

| File | Why |
|------|-----|
| `src/executor.py` | A2A boilerplate, rarely needs changes |
| `src/models.py` | Simple Pydantic models |
| `tests/conftest.py` | Pytest fixtures, stable |
| `Dockerfile` | Works, don't break it |
| `uv.lock` | Auto-generated, don't edit manually |

---

## Checklist for Changes

Before pushing:

- [ ] `uv sync` works
- [ ] `uv run src/server.py` starts without errors
- [ ] `uv run pytest -v --agent-url http://localhost:9009` passes
- [ ] Docker build works: `docker buildx build -t test .`
- [ ] Docker run works: `docker run -p 9009:9009 -p 9006:9006 test --host 0.0.0.0 --port 9009 --mcp-host 0.0.0.0 --mcp-port 9006`

---

Good luck! 🚀

