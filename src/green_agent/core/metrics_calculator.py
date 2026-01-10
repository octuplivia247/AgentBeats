import re
from typing import List, Dict, Set, Any, Optional
import json
from dataclasses import dataclass
import os



@dataclass
class TaskResult:
    """
    Represents the result of a task execution.
    
    Attributes:
        task_id: Unique identifier for the task
        predicted_operations: List of operations predicted/executed by the agent
        expected_operations: List of operations that were expected
        additional_info: Optional dictionary for extra metadata
    """
    task_id: str
    predicted_operations: List[str]
    expected_operations: List[str]
    additional_info: Optional[Dict[str, Any]] = None


class MetricsCalculator:
    """
    Calculates evaluation metrics for agent task completion
    
    This calculator is designed for A2A protocol and supports:
    - Exact match: Binary metric for perfect prediction matching
    - Precision: Ratio of correct predictions to total predictions
    - Recall: Ratio of correct predictions to total expected operations
    - F1 Score: Harmonic mean of precision and recall
    
    The metrics follow the HomeBench evaluation methodology for smart home
    agent task completion assessment.
    
    Usage:
        calculator = MetricsCalculator()
        
        # Single task evaluation
        metrics = calculator.compute_metrics(
            predicted=["turn on light", "set temp 72"],
            expected=["turn on light", "set temp 72"]
        )
        
        # Multiple task evaluation
        results = [TaskResult(...), TaskResult(...)]
        aggregate = calculator.compute_aggregate_metrics(results)
    """
    
    def __init__(self):
        """Initialize the MetricsCalculator."""
        pass
    
    @staticmethod
    def normalize_operation(operation: str) -> str:
        """
        Normalize operation strings for consistent comparison.
        
        This method ensures that operation strings are compared consistently
        regardless of formatting differences. It:
        - Converts to lowercase for case-insensitive matching
        - Collapses multiple spaces into single spaces
        - Removes leading and trailing whitespace
        - Handles special characters consistently
        
        Args:
            operation: Raw operation string from agent or ground truth
            
        Returns:
            Normalized operation string ready for comparison
        """
        if not operation:
            return ""
        
        # Convert to lowercase for case-insensitive comparison
        normalized = operation.lower()
        
        # Replace multiple consecutive spaces with single space
        normalized = re.sub(r'\s+', ' ', normalized)
        
        # Remove leading and trailing whitespace
        normalized = normalized.strip()
        
        return normalized
    
    @staticmethod
    def normalize_operations_list(operations: List[str]) -> Set[str]:
        """
        Normalize a list of operations and return as a set for comparison.
        
        This helper method applies normalization to each operation in the list
        and returns a set for efficient comparison operations.
        
        Args:
            operations: List of operation strings to normalize
            
        Returns:
            Set of normalized operation strings (empty strings filtered out)
        """
        return {
            MetricsCalculator.normalize_operation(op) 
            for op in operations 
            if op and op.strip()
        }
    
    def compute_exact_match(
        self,
        predicted: List[str],
        expected: List[str]
    ) -> float:
        """
        Compute exact match score between predicted and expected operations.
        
        Returns 1.0 if the predicted and expected operations match exactly
        (as sets, so order doesn't matter), otherwise returns 0.0.
        This is a strict metric useful for evaluating whether an agent produced
        the exact set of required operations without any extras or omissions.
        
        Args:
            predicted: List of predicted/executed operations from the agent
            expected: List of expected/ground truth operations
            
        Returns:
            1.0 if exact match (same operations, regardless of order), 0.0 otherwise
            
        """
        predicted_set = self.normalize_operations_list(predicted)
        expected_set = self.normalize_operations_list(expected)
        
        return 1.0 if predicted_set == expected_set else 0.0
    
    def compute_precision_recall_f1(
        self,
        predicted: List[str],
        expected: List[str]
    ) -> tuple[float, float, float]:
        """
        Compute precision, recall, and F1 score.
        
        Precision measures what fraction of predicted operations were correct.
        Recall measures what fraction of expected operations were predicted.
        F1 is the harmonic mean of precision and recall.
        
        Edge case handling follows HomeBench conventions:
        - Both empty lists → (1.0, 1.0, 1.0) - perfect non-action
        - Expected empty, predicted not → (0.0, 1.0, 0.0) - false positives
        - Predicted empty, expected not → (1.0, 0.0, 0.0) - missed all operations
        - Division by zero handled gracefully
        
        Args:
            predicted: List of predicted/executed operations
            expected: List of expected/ground truth operations
            
        Returns:
            Tuple of (precision, recall, f1_score) where each is in [0.0, 1.0]
            
        Examples:
            >>> calc = MetricsCalculator()
            >>> calc.compute_precision_recall_f1([], [])
            (1.0, 1.0, 1.0)
            >>> calc.compute_precision_recall_f1(["a", "b"], ["a", "b", "c"])
            (1.0, 0.6666666666666666, 0.8)
            >>> calc.compute_precision_recall_f1(["a", "b", "c"], ["a", "b"])
            (0.6666666666666666, 1.0, 0.8)
            >>> calc.compute_precision_recall_f1(["x"], ["y"])
            (0.0, 0.0, 0.0)
        """
        predicted_set = self.normalize_operations_list(predicted)
        expected_set = self.normalize_operations_list(expected)
        
        # Both empty - perfect performance (no action needed, none taken)
        if len(predicted_set) == 0 and len(expected_set) == 0:
            return (1.0, 1.0, 1.0)
        
        # Expected is empty but predicted is not - false positives only
        if len(expected_set) == 0 and len(predicted_set) > 0:
            return (0.0, 1.0, 0.0)
        
        # Predicted is empty but expected is not - all missed
        if len(predicted_set) == 0 and len(expected_set) > 0:
            return (1.0, 0.0, 0.0)
        
        # Calculate true positives
        true_positives = len(predicted_set & expected_set)
        
        # Calculate precision: TP / (TP + FP) = TP / total_predicted
        precision = true_positives / len(predicted_set) if len(predicted_set) > 0 else 0.0
        
        # Calculate recall: TP / (TP + FN) = TP / total_expected
        recall = true_positives / len(expected_set) if len(expected_set) > 0 else 0.0
        
        # Calculate F1 score: 2 * (precision * recall) / (precision + recall)
        if precision + recall > 0:
            f1 = 2 * (precision * recall) / (precision + recall)
        else:
            f1 = 0.0
        
        return (precision, recall, f1)
    
    def compute_metrics(
        self,
        predicted: List[str],
        expected: List[str]
    ) -> Dict[str, float]:
        """
        Compute all metrics at once.
        
        Args:
            predicted: List of predicted/executed operations
            expected: List of expected/ground truth operations
            
        Returns:
            Dictionary containing all metrics:
            {
                'exact_match': Binary metric (0.0 or 1.0),
                'precision': Precision score [0.0, 1.0],
                'recall': Recall score [0.0, 1.0],
                'f1': F1 score [0.0, 1.0]
            }
        """
        exact_match = self.compute_exact_match(predicted, expected)
        precision, recall, f1 = self.compute_precision_recall_f1(predicted, expected)
        
        return {
            'exact_match': exact_match,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def compute_aggregate_metrics(
        self,
        task_results: List[TaskResult]
    ) -> Dict[str, float]:
        """
        Aggregate metrics across multiple TaskResult objects. Summary of agent performance
        
        Args:
            task_results: List of TaskResult objects to aggregate
            
        Returns:
            Dictionary containing aggregated metrics:
            {
                'avg_exact_match': Average exact match score,
                'avg_precision': Average precision,
                'avg_recall': Average recall,
                'avg_f1': Average F1 score,
                'total_tasks': Number of tasks evaluated,
                'perfect_tasks': Number of tasks with exact match = 1.0
            }
        """
        if not task_results:
            return {
                'avg_exact_match': 0.0,
                'avg_precision': 0.0,
                'avg_recall': 0.0,
                'avg_f1': 0.0,
                'total_tasks': 0,
                'perfect_tasks': 0
            }
        
        total_exact_match = 0.0
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0
        perfect_tasks = 0
        
        for task_result in task_results:
            metrics = self.compute_metrics(
                task_result.predicted_operations,
                task_result.expected_operations
            )
            
            total_exact_match += metrics['exact_match']
            total_precision += metrics['precision']
            total_recall += metrics['recall']
            total_f1 += metrics['f1']
            
            if metrics['exact_match'] == 1.0:
                perfect_tasks += 1
        
        num_tasks = len(task_results)
        
        return {
            'avg_exact_match': total_exact_match / num_tasks,
            'avg_precision': total_precision / num_tasks,
            'avg_recall': total_recall / num_tasks,
            'avg_f1': total_f1 / num_tasks,
            'total_tasks': num_tasks,
            'perfect_tasks': perfect_tasks
        }
    
    def compute_detailed_report(
        self,
        task_results: List[TaskResult]
    ) -> Dict[str, Any]:
        """
        Generate a detailed evaluation report with breakdown.
        This method provides both aggregate metrics and per-task details       
         
        Args:
            task_results: List of TaskResult objects to analyze
            
        Returns:
            Dictionary containing:
            - 'aggregate': Aggregate metrics across all tasks
            - 'per_task': List of per-task metrics with task IDs
            - 'summary': Additional summary statistics
        """
        aggregate = self.compute_aggregate_metrics(task_results)
        
        per_task = []
        for task_result in task_results:
            task_metrics = self.compute_metrics(
                task_result.predicted_operations,
                task_result.expected_operations
            )
            per_task.append({
                'task_id': task_result.task_id,
                'metrics': task_metrics,
                'predicted_count': len(task_result.predicted_operations),
                'expected_count': len(task_result.expected_operations)
            })
        
        return {
            'aggregate': aggregate,
            'per_task': per_task,
            'summary': {
                'total_tasks': len(task_results),
                'perfect_tasks': aggregate['perfect_tasks'],
                'success_rate': aggregate['avg_exact_match'],
                'avg_f1': aggregate['avg_f1']
            }
        }
    
class HomeBenchMetricsCalculator(MetricsCalculator):
    """
    Extends MetricsCalculator to handle HomeBench format data
    Adds parsing logic for the triple-quote format
    """
    
    @staticmethod
    def parse_homebench_output(output: Optional[str]) -> List[str]:
        """
        Parse HomeBench format output into list of operations.
        Handles None inputs safely.
        """
        if not output:
            return []
        
        # Remove triple quotes and strip whitespace
        output = output.replace("'''", "").strip()
        
        # Split by comma and strip spaces from each operation
        operations = [op.strip() for op in output.split(",") if op.strip()]
        
        return operations
    
    def evaluate_task(self, task_data: Dict[str, Any]) -> Dict[str, float]:
        """
        Evaluate a single task in HomeBench format
        
        Args:
            task_data: Dict with keys:
                - 'id': task identifier
                - 'expected_output': ground truth output
                - 'predicted_output': agent's output
                
        Returns:
            Dictionary with metrics:
                - 'exact_match': 1.0 if perfect match, 0.0 otherwise
                - 'precision': precision score
                - 'recall': recall score
                - 'f1': F1 score
        """
        expected = self.parse_homebench_output(task_data['expected_output'])
        predicted = self.parse_homebench_output(task_data['predicted_output'])
        
        return self.compute_metrics(predicted, expected)
    

    def evaluate_batch(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Evaluate multiple tasks and return aggregate metrics
        
        Args:
            tasks: List of task dicts with format:
                {
                    'id': task_id,
                    'agent_id': agent_id,
                    'unique_id': unique_id,
                    'expected_output': "...",
                    'predicted_output': "..."
                }
        
        Returns:
            Dictionary with:
                - 'aggregate': Overall metrics (EM, Precision, Recall, F1)
                - 'by_category': Metrics broken down by task type
                - 'per_task': Individual task results, with agent_id and unique_id first
        """
        # Convert to TaskResult format for aggregate computation
        task_results = []
        for task in tasks:
            expected = self.parse_homebench_output(task['expected_output'])
            predicted = self.parse_homebench_output(task['predicted_output'])
            
            task_results.append(TaskResult(
                task_id=task['id'],
                predicted_operations=predicted,
                expected_operations=expected,
                additional_info={
                    'type': task.get('type'),
                    'agent_id': task.get('agent_id', 'agent_dummy'),
                    'unique_id': task.get('unique_id', 247)
                }
            ))
        
        # Compute aggregate metrics
        aggregate = self.compute_aggregate_metrics(task_results)
        
        # Compute per-category metrics
        by_category = self._compute_category_metrics(task_results)
        
        # Compute per-task metrics, with agent_id and unique_id first
        per_task = []
        for task_result in task_results:
            metrics = self.compute_metrics(
                task_result.predicted_operations,
                task_result.expected_operations
            )
            per_task.append({
                'agent_id': task_result.additional_info.get('agent_id'),
                'unique_id': task_result.additional_info.get('unique_id'),
                **metrics,
                'task_id': task_result.task_id,
                'type': task_result.additional_info.get('type'),
                'predicted_count': len(task_result.predicted_operations),
                'expected_count': len(task_result.expected_operations)
            })
        
        return {
            'aggregate': aggregate,
            'by_category': by_category,
            'per_task': per_task
        }

    
    def _compute_category_metrics(
        self, 
        task_results: List[TaskResult]
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute metrics grouped by task category
        
        Categories: normal, unexist_device, unexist_attribute, 
                   multi_normal, multi_mix, multi_error
        """
        # Group tasks by category
        categories = {}
        for result in task_results:
            task_type = result.additional_info.get('type', 'unknown')
            if task_type not in categories:
                categories[task_type] = []
            categories[task_type].append(result)
        
        # Compute metrics for each category
        category_metrics = {}
        for cat_name, cat_tasks in categories.items():
            category_metrics[cat_name] = self.compute_aggregate_metrics(cat_tasks)
        
        return category_metrics


def evaluate_agent_output(tasks_with_predictions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Main function to call after purple agent outputs data
    
    Args:
        tasks_with_predictions: List of task dicts with both expected and predicted outputs
        
    Returns:
        Complete evaluation results with aggregate, per-category, and per-task metrics
    """
    calculator = HomeBenchMetricsCalculator()
    return calculator.evaluate_batch(tasks_with_predictions)


def final_evaluator_results(output_purple_agent) -> float:
    """
    Evaluate the agent output and return ONLY the overall accuracy (avg_exact_match).
    Tasks with missing 'predicted_output' are treated as failed.
    """
    prepared_tasks = []
    for t in output_purple_agent:
        task = dict(t)
        if "predicted_output" not in task or task["predicted_output"] is None:
            task["predicted_output"] = None  # explicitly mark as missing
        prepared_tasks.append(task)

    # Evaluate all tasks
    results = evaluate_agent_output(prepared_tasks)

    # Recalculate overall accuracy treating missing predicted_output as 0
    per_task = results.get("per_task", [])
    if not per_task:
        return 0.0

    total_exact_match = 0.0
    for task in per_task:
        # If predicted_output is None → task failed
        if task.get("predicted_count", 0) == 0:
            total_exact_match += 0.0
        else:
            total_exact_match += task.get("exact_match", 0.0)

    overall_accuracy = total_exact_match / len(per_task)
    return overall_accuracy




#################### TESTING #####################


# # import your evaluator
# # make sure this import path is correct
# # from evaluator import final_evaluator_results
# # or if same file, you can skip this import

# def load_jsonl(path: str) -> List[Dict[str, Any]]:
#     """Load a JSONL file into a list of dicts"""
#     data = []
#     with open(path, "r") as f:
#         for line_num, line in enumerate(f, 1):
#             line = line.strip()
#             if not line:
#                 continue
#             try:
#                 data.append(json.loads(line))
#             except json.JSONDecodeError as e:
#                 raise ValueError(f"JSON error on line {line_num}: {e}")
#     return data


# def prepare_tasks(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
#     """
#     Ensure each task has a predicted_output field.
#     If missing, set it to error_input (common HomeBench baseline).
#     """
#     prepared = []
#     for t in tasks:
#         task = dict(t)  # shallow copy

#         if "predicted_output" not in task:
#             # baseline / placeholder prediction
#             task["predicted_output"] = "'''error_input'''"

#         prepared.append(task)

#     return prepared

# #for testing purposes
# if __name__ == "__main__":

#     with open("subset_20_homebench_4-3-1-1-1.jsonl") as f:
#         tasks = [json.loads(line) for line in f if line.strip()]

#     for t in tasks:
#         if "predicted_output" not in t:
#             t["predicted_output"] = "'''error_input'''"

#     final_evaluator_results(tasks)