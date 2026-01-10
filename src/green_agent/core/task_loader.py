"""
Task loading utilities for HomeBench evaluation.

This module provides functions to load tasks and environment configurations
from HomeBench JSONL files.
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_homebench_config(
    method_path: str = "data/home_status_method.jsonl",
) -> Dict[str, Any]:
    """
    Load HomeBench configuration from JSONL file.

    Args:
        method_path: Path to home_status_method.jsonl

    Returns:
        Configuration dictionary with methods and home_status
    """
    with open(method_path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
        if not line:
            raise ValueError("Empty JSONL file")
        return json.loads(line)


def load_tasks_from_dataset(
    dataset_path: str = "data/home_status_data.jsonl",
    task_ids: Optional[List[int]] = None,
) -> List[Dict[str, Any]]:
    """
    Load tasks from HomeBench dataset.

    Args:
        dataset_path: Path to home_status_data.jsonl
        task_ids: Optional list of task indices to load (0-based)

    Returns:
        List of task dictionaries with:
        - task_id: Unique task identifier
        - instruction: Natural language instruction
        - expected_operations: List of expected operations
        - category: Task category (e.g., "normal_single", "multi8_mix")
        - home_id: Home environment ID
    """
    tasks = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if task_ids is not None and idx not in task_ids:
                continue
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                # Parse expected operations from output
                output = data.get("output", "")
                # Remove surrounding quotes and triple quotes
                output = output.strip("'\"")
                operations = [
                    op.strip()
                    for op in output.split(",")
                    if op.strip() and op.strip() != "error_input"
                ]

                tasks.append({
                    "task_id": data.get("id", f"task_{idx}"),
                    "instruction": data.get("input", ""),
                    "expected_operations": operations,
                    "category": data.get("type", "normal_single"),
                    "home_id": data.get("home_id", 0),
                })
            except json.JSONDecodeError:
                continue

    return tasks


def get_task_by_id(
    dataset_path: str,
    task_id: str,
) -> Optional[Dict[str, Any]]:
    """
    Get a specific task by ID.

    Args:
        dataset_path: Path to home_status_data.jsonl
        task_id: Task ID to find

    Returns:
        Task dictionary or None if not found
    """
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if data.get("id") == task_id:
                    output = data.get("output", "").strip("'\"")
                    operations = [
                        op.strip()
                        for op in output.split(",")
                        if op.strip() and op.strip() != "error_input"
                    ]
                    return {
                        "task_id": data.get("id"),
                        "instruction": data.get("input", ""),
                        "expected_operations": operations,
                        "category": data.get("type", "normal_single"),
                        "home_id": data.get("home_id", 0),
                    }
            except json.JSONDecodeError:
                continue
    return None

