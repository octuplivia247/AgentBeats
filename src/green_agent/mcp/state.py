from typing import Optional, Dict, List

from src.green_agent.core.evaluator import HomeBenchEvaluator
from src.green_agent.core.models.task_result import TaskResult
from src.green_agent.core.smart_home_env_manager import SmartHomeEnvironment


class GlobalState:
    """Manages global state for the MCP server."""

    def __init__(self):
        self.environments: Dict[str, SmartHomeEnvironment] = {}
        self.evaluators: Dict[str, HomeBenchEvaluator] = {}
        self.current_env_id: Optional[str] = None
        self.task_results: Dict[str, List[TaskResult]] = {}

    def get_or_create_environment(self, env_id: str = "default") -> SmartHomeEnvironment:
        """Get or create an environment."""
        if env_id not in self.environments:
            self.environments[env_id] = SmartHomeEnvironment()
        return self.environments[env_id]

    def get_or_create_evaluator(self, agent_url: str, eval_id: str = "default") -> HomeBenchEvaluator:
        """Get or create an evaluator."""
        if eval_id not in self.evaluators:
            self.evaluators[eval_id] = HomeBenchEvaluator(agent_url)
        return self.evaluators[eval_id]

    def clear_all(self):
        """Clear all state."""
        self.environments.clear()
        self.evaluators.clear()
        self.task_results.clear()
        self.current_env_id = None
