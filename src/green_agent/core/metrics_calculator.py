from typing import Dict, List, Tuple

from src.green_agent.core.models import TaskResult


class MetricsCalculator:
    """
    Calculates evaluation metrics for HomeBench tasks.

    This class provides static methods to compute:
    - Exact Match (EM): Whether predicted operations exactly match expected
    - Precision: Fraction of predicted operations that are correct
    - Recall: Fraction of expected operations that were predicted
    - F1 Score: Harmonic mean of precision and recall

    All methods should handle edge cases like empty lists.

    Usage:
        metrics = MetricsCalculator.compute_metrics(predicted, expected)
        # Returns: {"exact_match": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
    """

    @staticmethod
    def normalize_operation(op: str) -> str:
        """
        Normalize an operation string for consistent comparison.

        This should:
        1. Convert to lowercase
        2. Remove extra whitespace
        3. Standardize formatting

        Args:
            op: Operation string (e.g., "Living_Room.Light.Turn_On()")

        Returns:
            Normalized operation string (e.g., "living_room.light.turn_on()")

        TODO: Implement this method
        - Apply op.strip().lower().replace(" ", "")
        - Handle any special characters consistently
        """
        raise NotImplementedError()

    @staticmethod
    def compute_exact_match(predictions: List[str], expected: List[str]) -> float:
        """
        Compute exact match score.

        Exact match is 1.0 if the predicted operations exactly match the expected
        operations (as sets, order doesn't matter), otherwise 0.0.

        Args:
            predictions: List of predicted operation strings
            expected: List of expected operation strings

        Returns:
            1.0 if exact match, 0.0 otherwise

        TODO: Implement this method
        - Normalize both lists
        - Compare as sets (order-independent)
        - Return 1.0 if equal, 0.0 if not
        """
        raise NotImplementedError()

    @staticmethod
    def compute_precision_recall_f1(
        predictions: List[str], expected: List[str]
    ) -> Tuple[float, float, float]:
        """
        Compute precision, recall, and F1 score.

        Formulas:
        - Precision = |predicted ∩ expected| / |predicted|
        - Recall = |predicted ∩ expected| / |expected|
        - F1 = 2 * (precision * recall) / (precision + recall)

        Edge cases:
        - If both lists empty: return (1.0, 1.0, 1.0)
        - If predictions empty: return (0.0, 0.0, 0.0)
        - If expected empty: return (0.0, 0.0, 0.0)
        - If precision + recall = 0: F1 = 0.0

        Args:
            predictions: List of predicted operation strings
            expected: List of expected operation strings

        Returns:
            Tuple of (precision, recall, f1) each in range [0.0, 1.0]

        TODO: Implement this method
        - Normalize both lists
        - Convert to sets
        - Compute intersection (true positives)
        - Calculate precision, recall, F1 using formulas above
        - Handle edge cases
        """
        raise NotImplementedError()

    @staticmethod
    def compute_metrics(predictions: List[str], expected: List[str]) -> Dict[str, float]:
        """
        Compute all metrics at once.

        Convenience method that calls the other methods and returns all metrics.

        Args:
            predictions: List of predicted operation strings
            expected: List of expected operation strings

        Returns:
            Dictionary with all metrics:
                {
                    "exact_match": float,
                    "precision": float,
                    "recall": float,
                    "f1": float
                }

        TODO: Implement this method
        - Call compute_exact_match()
        - Call compute_precision_recall_f1()
        - Return combined dictionary
        """
        raise NotImplementedError()

    @staticmethod
    def compute_aggregate_metrics(results: List[TaskResult]) -> Dict[str, float]:
        """
        Compute aggregate metrics across multiple task results.

        This should collect all predictions and expected operations from all results,
        then compute metrics on the combined set.

        Args:
            results: List of TaskResult objects

        Returns:
            Dictionary with aggregate metrics

        TODO: Implement this method
        - Extract all predicted_operations from results
        - Extract all expected_operations from results
        - Call compute_metrics on combined lists
        """
        raise NotImplementedError()
