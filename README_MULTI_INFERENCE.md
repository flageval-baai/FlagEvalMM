# Multiple Inference Feature

Due to variation of reasoning model's output, we need to perform multiple inferences for each question when temperature >= 0, and evaluate the results by taking the average accuracy.

This document describes the multiple inference feature that allows models to perform multiple inferences for each question when temperature >= 0, and evaluates the results by taking the average accuracy.

## Overview

When `temperature >= 0` and `num_infers > 1`, the model will perform multiple inferences for each question and calculate the average accuracy score across all inferences.

The `MultiInferenceEvaluator` expands multiple inference predictions into individual evaluations, then aggregates the results back to provide comprehensive evaluation metrics.

## How to Use

### Basic Configuration

Use `MultiInferenceEvaluator` as the evaluator in your task configuration:

```python
# Example: blink.py
task_name = "vqa"

config = dict(
    dataset_path="BLINK-Benchmark/BLINK",
    split="val",
    processed_dataset_path="BLINK",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt="The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options.",
    ),
    name="blink_val",
)

evaluator = dict(type="MultiInferenceEvaluator", detailed_keys=["sub_task"])
```

### Advanced Configuration with Aggregation Fields

For tasks that require additional evaluation metrics beyond accuracy, you can specify `aggregation_fields` to preserve extra evaluation data during the multi-inference aggregation process:

```python
# Example: acrostic task with additional evaluation metrics
evaluator = dict(
    type="MultiInferenceEvaluator", 
    eval_func="evaluate.py", 
    aggregation_fields=["detail", "judge_score", "judge_response"]
)
```

#### aggregation_fields Parameter

The `aggregation_fields` parameter specifies which additional fields from the evaluation process should be preserved when aggregating multiple inference results. This is particularly useful for:

- **Complex evaluation metrics**: When your evaluation function adds custom fields to prediction objects
- **Detailed analysis**: Preserving intermediate evaluation results for further analysis
- **Multi-faceted evaluation**: Tasks that involve multiple evaluation dimensions (e.g., accuracy + quality scores)

**Example usage scenarios:**
- `detail`: Stores detailed validation results
- `judge_score`: Numerical scores from LLM-based evaluation

**How it works:**
1. During evaluation, your `eval_func` adds these fields to each prediction object
2. When aggregating multiple inferences, the evaluator preserves these fields alongside the standard accuracy metrics
3. The final output contains both aggregated accuracy scores and the specified additional fields

### Command Line Usage

When starting a task, add the `--num-infers` and `--temperature` arguments:

#### GPT-4o-mini:
```bash
flagevalmm --tasks tasks/blink/blink_val.py \
    --exec model_zoo/vlm/api_model/model_adapter.py \
    --model gpt-4o-mini \
    --num-workers 8 \
    --output-dir ./results_temperature/gpt-4o-mini \
    --url openai_url \
    --api-key openai_api_key \
    --num-infers 5 \
    --temperature 0.6 \
    --try-run \
    --use-cache
```

#### Qwen2.5-VL-7B-Instruct:
```bash
flagevalmm --tasks tasks/blink/blink_val.py \
    --cfg model_configs/open/Qwen2.5-VL-7B-Instruct.json \
    --quiet \
    --output-dir ./results/Qwen2.5-VL-7B-Instruct \
    --exec model_zoo/vlm/api_model/model_adapter.py \
    --backend vllm \
    --num-infers 3 \
    --try-run \
    --temperature 0.6
```


### With Aggregation Fields
When using `aggregation_fields`, additional evaluation data is preserved:
```json
{
    "question_id": "q1",
    "question": "Create an acrostic for 'CAT'",
    "answer": {
        "inference_0": "Cute animals today...",
        "inference_1": "Cats are terrific...",
        "inference_2": "Clever and talented..."
    },
    "correct": 0.67,
    "inference_scores": [1, 1, 0],
    "num_inferences": 3,
    "judge_score": [8.5, 8.5, 8.5],
    "judge_response": [
        "The answer is ....",
        "The answer is ....",
        "The answer is ...."
    ]
}
```

