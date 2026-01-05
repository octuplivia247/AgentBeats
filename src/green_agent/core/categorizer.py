from typing import Any, Dict, List


class TaskCategorizer:
    """
    Categorizes tasks into HomeBench evaluation categories.


    TODO: Implement this class
    - Map HomeBench type field to categories
    - Load and parse JSONL dataset files
    - Group examples by category
    """

    CATEGORIES = [
        "normal_single",
        "normal_multi",
        "mix_multi",
    ]

    @staticmethod
    def categorize_from_type(task_type: str) -> str:
        """Map HomeBench type to category."""
        raise NotImplementedError()

    @staticmethod
    def load_and_categorize_dataset(dataset_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Load dataset and group by category."""
        raise NotImplementedError()
