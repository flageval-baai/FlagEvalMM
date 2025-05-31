# Multiple Inference Feature

Due to variation of reasoning model's output, we need to perform multiple inferences for each question when temperature >= 0, and evaluate the results by taking the average accuracy.

This document describes the multiple inference feature that allows models to perform multiple inferences for each question when temperature >= 0, and evaluates the results by taking the average accuracy.

## Overview

When `temperature >= 0` and `num_infers > 1`, the model will perform multiple inferences for each question and calculate the average accuracy score across all inferences.

## How to use

Use `MultiInferenceEvaluator` as the evaluator:

```python
## blink.py
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

When start a task, add "nums-infer" argument and add `temperature` in --extra-args, such as: 

```bash
flagevalmm --tasks tasks/blink/blink_val.py  --exec model_zoo/vlm/api_model/model_adapter.py --model gpt-4o-mini --num-workers 8 --output-dir ./results_temperature/gpt-4o-mini --url openai_url --api-key openai_api_key --num-infers 3 --try-run --use-cache --extra-args "temperature=0.6"
```

## Output Format

For each question with multiple inferences, the result includes:

```json
{
    "question_id": "q1",
    "question": "What is this?",
    "answer": {
        "inference_0": "extracted_result1",
        "inference_1": "extracted_result2", 
        "inference_2": "extracted_result3",
        "inference_3": "extracted_result4",
        "inference_4": "extracted_result5"
    },
    "multiple_answers": ["result1", "result2", "result3", "result4", "result5"],
    "correct": 0.8,
    "inference_scores": [1, 1, 0, 1, 1],
    "num_inferences": 5
}
```