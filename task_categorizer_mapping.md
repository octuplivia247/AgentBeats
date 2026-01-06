# Task Categorizer Input-Output Mapping

This file shows how the TaskCategorizer maps input `type` fields from the dataset to output categories.

## Mapping Rules:
- "normal" → normal_single
- "unexist_device" → unexist_device_single
- "unexist_attribute" → unexist_attribute_single
- "multiX_normal" → normal_multi
- "multiX_mix" → mix_multi
- "multiX_error" → error_multi
- "multiX_unexist_device" → normal_multi (no explicit error/mix)
- "multiX_unexist_attribute" → normal_multi (no explicit error/mix)

## Examples from subset_20_homebench_4-3-1-1-1.jsonl:

Input Type | Output Category | Count
-----------|-----------------|------
multi10_mix | mix_multi | 1
unexist_device | unexist_device_single | 5
normal | normal_single | 8
multi2_normal | normal_multi | 2
unexist_attribute | unexist_attribute_single | 1
multi3_unexist_device | normal_multi | 1
multi5_mix | mix_multi | 1
multi3_unexist_attribute | normal_multi | 1

## Downstream Usage:
The categorized data is returned as a dict where keys are category names and values are lists of task dicts.
Each task dict contains: home_id, input, type, id, output.

This allows downstream processing to filter tasks by category for evaluation.