FROM ghcr.io/astral-sh/uv:python3.13-bookworm

RUN adduser agent
USER agent
WORKDIR /home/agent

# Copy green-agent application files
COPY green-agent/pyproject.toml green-agent/uv.lock green-agent/README.md ./
COPY green-agent/src src

# Copy data from parent directory
COPY data data

RUN \
    --mount=type=cache,target=/home/agent/.cache/uv,uid=1000 \
    uv sync --locked

# Environment variables for MCP integration
ENV MCP_SERVER_URL=http://127.0.0.1:9006
ENV PYTHONUNBUFFERED=1

# Expose both A2A server port (9009) and MCP server port (9006)
EXPOSE 9009
EXPOSE 9006

# Run the server which starts both A2A and MCP servers
ENTRYPOINT ["uv", "run", "src/server.py"]
CMD ["--host", "0.0.0.0", "--port", "9009", "--mcp-host", "0.0.0.0", "--mcp-port", "9006"]
