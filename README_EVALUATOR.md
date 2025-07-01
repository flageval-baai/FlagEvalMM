# How to use BaseEvaluator

Due to variation of reasoning model's output, we need to perform multiple inferences for each question when temperature >= 0, and evaluate the results by taking the average accuracy.

This document describes the multiple inference feature that allows models to perform multiple inferences for each question when temperature >= 0, and evaluates the results by taking the average accuracy.

## Overview

When `temperature >= 0` and `num_infers > 1`, the model will perform multiple inferences for each question and calculate the average accuracy score across all inferences.

The `BaseEvaluator` automatically detects and expands multiple inference predictions into individual evaluations, then aggregates the results back to provide comprehensive evaluation metrics.

## How to Use

### Basic Configuration

Use `BaseEvaluator` as the evaluator in your task configuration (multi-inference support is automatic):

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

evaluator = dict(type="BaseEvaluator", detailed_keys=["sub_task"])
# evaluator = dict(
#     type="BaseEvaluator",
#     eval_model_name="gpt-4.1-mini-2025-04-14",
#     use_llm_evaluator=True,
#     use_cache=True,
#     base_url=base_url,
#     api_key=api_key,
# )
```

### Parameter `question_type`

The `BaseEvaluator` uses a `question_type` parameter in your data.json file to determine the appropriate evaluation method for different types of questions. The `BaseEvaluator` natively supports the following question types:
- **multiple-choice**: For multiple-choice questions where the answer is a letter (A, B, C, D, etc.)
- **fill-blank**: For fill-in-the-blank questions that require string similarity matching
- **llm-judge**: For open-ended questions that require LLM-based evaluation (requires `use_llm_evaluator=True`)

Example:
```json
[
  {
    "question": "<image 1> Baxter Company has a relevant range of production between 15,000 and 30,000 units. The following cost data represents average variable costs per unit for 25,000 units of production. If 30,000 units are produced, what are the per unit manufacturing overhead costs incurred?Options: A. $6, B. $7, C. $8, D. $9",
    "answer": "B",
    "question_id": "validation_Accounting_1",
    "option": [
      "$6",
      "$7",
      "$8",
      "$9"
    ],
    "question_type": "multiple-choice",
    "img_path": [
      "img/validation_Accounting_1_1.png"
    ]
  },
  {
    "question": "Maxwell Software, Inc., has the following mutually exclusive projects.Suppose the company uses the NPV rule to rank these two projects.<image 1> Which project should be chosen if the appropriate discount rate is 15 percent?Options: A. Project A, B. Project B. Answer with the choice only.",
    "answer": "B",
    "question_id": "validation_Accounting_3",
    "option": [
      "Project A",
      "Project B"
    ],
    "question_type": "open-ended",
    "img_path": [
      "img/validation_Accounting_3_1.png"
    ]
  },
  {
    "question": "Each situation below relates to an independent company's Owners' Equity. <image 1> Calculate the missing values of company 2.Options: A. $1,620, B. $12,000, C. $51,180, D. $0",
    "answer": "D",
    "question_id": "validation_Accounting_4",
    "option": [
      "$1,620",
      "$12,000",
      "$51,180",
      "$0"
    ],
    "question_type": "llm-judge",
    "img_path": [
      "img/validation_Accounting_4_1.png"
    ]
  }
]
```

### Advanced Configuration with Aggregation Fields

For tasks that require additional evaluation metrics beyond accuracy, you can specify `aggregation_fields` to preserve extra evaluation data during the multi-inference aggregation process:

```python
# Example: acrostic task with additional evaluation metrics
evaluator = dict(
    type="BaseEvaluator", 
    eval_func="evaluate.py", 
    aggregation_fields=["detail", "judge_score", "judge_response"]
)
```

The `aggregation_fields` parameter specifies which additional fields from the evaluation process should be preserved when aggregating multiple inference results. 

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

### Multi-inference Command Line Usage

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
