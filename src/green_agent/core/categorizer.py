from typing import Any, Dict, List
import json
import logging
from pathlib import Path


logger = logging.getLogger(__name__)


class TaskCategorizer:
    """
    Categorizes tasks into HomeBench evaluation categories.

    This class provides two helpers:
    - `categorize_from_type` maps the dataset's `type` string to one of the
      HomeBench categories.
    - `load_and_categorize_dataset` reads a JSONL file and groups examples by
      category, handling malformed lines gracefully.
    """

    CATEGORIES = [
        "normal_single",
        "unexist_single",
        "unexist_device_single",
        "unexist_attribute_single",
        "normal_multi",
        "mix_multi",
        "error_multi",
    ]

    @staticmethod
    def categorize_from_type(task_type: str) -> str:
        """Map HomeBench `type` string to a known category.

        The dataset uses compact type labels (e.g. "normal", "unexist_device",
        "multi8_mix", "multi7_mix"). This function normalizes and maps those
        values into the canonical categories listed in `CATEGORIES`.

        Returns one of the categories in `CATEGORIES`, or the string
        `'unknown'` / `'malformed'` for unexpected inputs.
        """
        if not isinstance(task_type, str):
            return "malformed"

        t = task_type.strip().lower()
        # Multi examples: prefer explicit mix / error detection
        if "multi" in t:
            if "mix" in t:
                return "mix_multi"
            if "error" in t:
                return "error_multi"
            return "normal_multi"

        # Single examples
        if "unexist_device" in t or t == "unexistdevice":
            return "unexist_device_single"
        if "unexist_attribute" in t or t == "unexistattribute":
            return "unexist_attribute_single"
        if t.startswith("unexist") or "unexist_" in t:
            return "unexist_single"
        if "normal" in t:
            return "normal_single"
        if "error" in t:
            return "error_multi"

        return "unknown"

    @staticmethod
    def load_and_categorize_dataset(dataset_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """Load a JSONL dataset and group examples by category.

        Args:
            dataset_path: Path to a JSONL file where each line is a JSON object
                representing a task and containing a `type` field.

        Returns:
            A dict mapping category name -> list of example dicts. Always
            includes keys for all values in `CATEGORIES` plus `unknown` and
            `malformed` (for lines that couldn't be parsed or lacked `type`).
        """
        path = Path(dataset_path)
        categorized: Dict[str, List[Dict[str, Any]]] = {c: [] for c in TaskCategorizer.CATEGORIES}
        categorized.setdefault("unknown", [])
        categorized.setdefault("malformed", [])

        try:
            with path.open("r", encoding="utf-8") as fh:
                for lineno, raw in enumerate(fh, start=1):
                    line = raw.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        logger.warning("Malformed JSON at %s:%d", dataset_path, lineno)
                        categorized["malformed"].append({"line": lineno, "raw": line})
                        continue

                    if not isinstance(obj, dict):
                        categorized["malformed"].append({"line": lineno, "value": obj})
                        continue

                    # Add predicted_output key as a placeholder if not present
                    obj.setdefault("predicted_output", None)

                    type_field = obj.get("type")
                    if type_field is None:
                        categorized["malformed"].append({"line": lineno, "task": obj})
                        continue

                    cat = TaskCategorizer.categorize_from_type(type_field)
                    if cat == "malformed":
                        categorized["malformed"].append({"line": lineno, "task": obj})
                    elif cat == "unknown":
                        categorized["unknown"].append(obj)
                    else:
                        categorized.setdefault(cat, []).append(obj)

        except FileNotFoundError:
            logger.error("Dataset file not found: %s", dataset_path)
            raise

        return categorized
