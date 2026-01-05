"""Purple Agent - Smart Home Task Execution Agent with MCP Integration.

The purple agent is the target agent being evaluated. It uses MCP tools to interact
with the smart home environment and execute tasks assigned by the green agent.
"""

from src.purple_agent.agent import PurpleAgentExecutor, start_purple_agent

__all__ = ["start_purple_agent", "PurpleAgentExecutor"]
