# AgentBeats Dataset

This directory contains the evaluation datasets for the HomeBench smart home benchmark.

## Files

### test_data.jsonl
- **Tasks**: 998 evaluation tasks (reduced from 17,366 for faster evaluation)
- **Categories**: 26 task types with proportional distribution maintained
- **Format**: JSONL (one JSON object per line)
- **Fields**:
  - `id`: Unique task identifier
  - `input`: Natural language instruction
  - `output`: Expected operations in HomeBench format (`'''op1,op2'''`)
  - `home_id`: Home configuration ID (0-99)
  - `type`: Task category

### test_data.full.jsonl (backup)
- **Tasks**: 17,366 complete dataset
- **Note**: Ignored by git, created automatically as backup
- To restore full dataset: `cp data/test_data.full.jsonl data/test_data.jsonl`

### home_status_method_all.jsonl
- **Homes**: 100 home configurations
- **Content**: Device capabilities and current states for each home
- **Format**: JSONL with `home_id`, `method` (available operations), and `home_status` (current states)

## Task Categories Distribution (test_data.jsonl)

| Category | Count | % | Description |
|----------|-------|---|-------------|
| normal | 356 | 35.7% | Valid single-device instructions |
| unexist_device | 241 | 24.1% | Instructions referencing non-existent devices |
| unexist_attribute | 148 | 14.8% | Instructions with invalid device attributes |
| multi*_mix | 235 | 23.5% | Multi-device mixed valid/invalid instructions |
| multi*_normal | 12 | 1.2% | Multi-device valid instructions |
| multi*_unexist_* | 6 | 0.6% | Multi-device invalid instructions |

**Total**: 998 tasks across 26 categories

## Dataset Sampling

The current dataset is a stratified sample that maintains the original distribution:
- Proportional sampling from each category
- Minimum 1 task per category (even rare ones)
- Random seed: 42 (reproducible)

### To Create Custom Sample

```python
import json
import random
from collections import defaultdict

# Load full dataset
with open('data/test_data.full.jsonl') as f:
    tasks = [json.loads(line) for line in f if line.strip()]

# Group by category
by_category = defaultdict(list)
for task in tasks:
    by_category[task.get('type', 'unknown')].append(task)

# Sample proportionally
target = 1000  # Desired number of tasks (adjust as needed)
sampled = []
for cat, cat_tasks in by_category.items():
    proportion = len(cat_tasks) / len(tasks)
    sample_size = max(1, int(target * proportion))
    sampled.extend(random.sample(cat_tasks, min(sample_size, len(cat_tasks))))

# Write
with open('data/test_data.jsonl', 'w') as f:
    for task in sampled:
        f.write(json.dumps(task) + '\n')
```

## Usage in Evaluation

The dataset is automatically loaded by the Green Agent:

```json
{
  "participants": {"purple_agent": "http://..."},
  "config": {
    "dataset_path": "data/test_data.jsonl",
    "home_data_path": "data/home_status_method_all.jsonl"
  }
}
```

### Load Subset for Testing

```json
{
  "config": {
    "reduced_tests": true,
    "max_tasks": 50
  }
}
```

### Use Full Dataset

```bash
# Restore full dataset (17,366 tasks)
cp data/test_data.full.jsonl data/test_data.jsonl

# Or specify in config
{
  "config": {
    "dataset_path": "data/test_data.full.jsonl"
  }
}
```

## File Sizes

- `test_data.jsonl`: ~279 KB (998 tasks)
- `test_data.full.jsonl`: ~4.8 MB (17,366 tasks)
- `home_status_method_all.jsonl`: ~2.1 MB (100 homes)

## Notes

- The reduced dataset provides ~5.7% of the original, sufficient for evaluation
- Distribution is maintained for representative testing
- Full dataset available in backup for comprehensive evaluation
- Evaluation time: ~3-5 minutes for 1K tasks vs ~45-60 minutes for 17K tasks
