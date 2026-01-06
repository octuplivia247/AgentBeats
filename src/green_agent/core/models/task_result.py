from dataclasses import dataclass
from typing import Any, List


@dataclass
class TaskResult:
    """
    Result of evaluating a single task.

    This class encapsulates all information about a task evaluation including
    the predicted and expected operations, success metrics, and execution details.

    Attributes:
        task_id: Unique identifier for the task
        instruction: The natural language instruction given to the agent
        expected_operations: List of operations that should have been performed
        predicted_operations: List of operations the agent actually performed
        actions_taken: Detailed log of actions executed in the environment
        success: Whether the task was completed successfully
        score: Numeric score (typically F1) representing performance
        feedback: Human-readable feedback about the evaluation
        category: HomeBench category (e.g., "normal_single", "mix_multi")
        execution_time: Time taken to complete the task in seconds
    """

    task_id: str
    instruction: str
    expected_operations: List[str]
    predicted_operations: List[str]
    actions_taken: List[Any]
    success: bool
    score: float
    feedback: str
    category: str = "normal_single"
    execution_time: float = 0.0
