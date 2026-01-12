"""Metrics Calculator for HomeBench Evaluation."""

import re


class MetricsCalculator:
    """Calculates EM, Precision, Recall, F1 for agent task completion."""
    
    @staticmethod
    def normalize_operation(operation: str) -> str:
        """Normalize operation string for comparison."""
        if not operation:
            return ""
        normalized = operation.lower()
        normalized = re.sub(r'\s+', ' ', normalized)
        return normalized.strip()
    
    @staticmethod
    def normalize_operations_list(operations: list[str]) -> set[str]:
        """Normalize operations list to set."""
        return {
            MetricsCalculator.normalize_operation(op) 
            for op in operations 
            if op and op.strip()
        }
    
    def compute_metrics(self, predicted: list[str], expected: list[str]) -> dict[str, float]:
        """Compute EM, precision, recall, F1."""
        pred_set = self.normalize_operations_list(predicted)
        exp_set = self.normalize_operations_list(expected)
        
        exact_match = 1.0 if pred_set == exp_set else 0.0
        
        # Handle edge cases
        if len(pred_set) == 0 and len(exp_set) == 0:
            return {"exact_match": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0}
        if len(exp_set) == 0:
            return {"exact_match": 0.0, "precision": 0.0, "recall": 1.0, "f1": 0.0}
        if len(pred_set) == 0:
            return {"exact_match": 0.0, "precision": 1.0, "recall": 0.0, "f1": 0.0}
        
        true_positives = len(pred_set & exp_set)
        precision = true_positives / len(pred_set)
        recall = true_positives / len(exp_set)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {"exact_match": exact_match, "precision": precision, "recall": recall, "f1": f1}


class HomeBenchMetricsCalculator(MetricsCalculator):
    """Handles HomeBench triple-quote format parsing."""

    @staticmethod
    def parse_homebench_output(output: str) -> list[str]:
        """Parse HomeBench format: "'''op1,op2,'''" -> ["op1", "op2"]"""
        output = output.replace("'''", "").replace(" ", "").replace("\n", "")
        return [op for op in output.split(",") if op]
