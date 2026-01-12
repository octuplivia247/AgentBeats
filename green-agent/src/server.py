"""Server entry point - starts both A2A and MCP servers."""

import argparse
import logging
import multiprocessing
import os
import time

import httpx
import uvicorn

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentSkill

from executor import Executor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_mcp(host: str, port: int):
    from mcp_server import run_server
    run_server(host, port)


def start_mcp(host: str, port: int) -> multiprocessing.Process | None:
    try:
        p = multiprocessing.Process(target=run_mcp, args=(host, port), daemon=True)
        p.start()
        logger.info(f"MCP server started (PID: {p.pid})")
        return p
    except Exception as e:
        logger.error(f"Failed to start MCP: {e}")
        return None


def wait_mcp(host: str, port: int, timeout: int = 10) -> bool:
    url = f"http://127.0.0.1:{port}/health"
    for _ in range(timeout * 2):
        try:
            with httpx.Client(timeout=2) as c:
                if c.get(url).status_code == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=9009)
    parser.add_argument("--card-url", type=str)
    parser.add_argument("--mcp-host", default="0.0.0.0")
    parser.add_argument("--mcp-port", type=int, default=9006)
    parser.add_argument("--no-mcp", action="store_true")
    args = parser.parse_args()

    mcp_proc = None
    if not args.no_mcp:
        mcp_proc = start_mcp(args.mcp_host, args.mcp_port)
        if mcp_proc and wait_mcp(args.mcp_host, args.mcp_port):
            os.environ["MCP_SERVER_URL"] = f"http://127.0.0.1:{args.mcp_port}"
            logger.info("MCP server ready")

    skill = AgentSkill(
        id="evaluate_smart_home_agent",
        name="Evaluate Smart Home Agent",
        description="Evaluate purple agents on HomeBench smart home tasks using MCP tools.",
        tags=["evaluation", "homebench", "smart-home", "mcp"],
        examples=['{"participants": {"purple_agent": "http://localhost:8000"}, "config": {"tasks": [...]}}']
    )

    card = AgentCard(
        name="HomeBench Green Agent",
        description="Evaluator agent for HomeBench benchmark with MCP integration.",
        url=args.card_url or f"http://{args.host}:{args.port}/",
        version="1.0.0",
        default_input_modes=["text"],
        default_output_modes=["text"],
        capabilities=AgentCapabilities(streaming=True),
        skills=[skill]
    )

    handler = DefaultRequestHandler(agent_executor=Executor(), task_store=InMemoryTaskStore())
    server = A2AStarletteApplication(agent_card=card, http_handler=handler)

    print(f"HomeBench Green Agent: {card.url}")
    if mcp_proc:
        print(f"MCP Server: http://{args.mcp_host}:{args.mcp_port}")

    try:
        uvicorn.run(server.build(), host=args.host, port=args.port)
    finally:
        if mcp_proc and mcp_proc.is_alive():
            mcp_proc.terminate()


if __name__ == "__main__":
    main()
