---
name: flagevalmm-add-dataset
description: >
  Integrate new evaluation datasets into FlagEvalMM as benchmark tasks.
  Use when adding a dataset from HuggingFace or other sources to FlagEvalMM,
  creating task configs, writing data processors, building custom evaluators,
  setting up prompt templates, or running evaluation benchmarks on VLMs.
  Trigger on: "add dataset to FlagEvalMM", "create a new task", "integrate benchmark",
  "evaluate model on [dataset]", "write process.py", "write evaluator", or any
  request involving the tasks/ directory of FlagEvalMM.
---

# FlagEvalMM Dataset Integration

Add new evaluation datasets to FlagEvalMM as benchmark tasks. This skill covers the
full workflow: data processing, task configuration, evaluation logic, prompt design,
and verification.

## Task Directory Structure

Every task lives under `tasks/<task_name>/` with these files:

```
tasks/<task_name>/
├── <task_name>.py              # Task config (one per split/variant) [REQUIRED]
├── process.py                  # Data processor                      [REQUIRED for HuggingFace data]
├── evaluate.py                 # Custom eval function                [OPTIONAL]
├── post_prompt.py              # Shared prompt config via _base_     [OPTIONAL]
└── <name>_evaluator.py         # Custom evaluator class              [OPTIONAL]
```

## Workflow

Follow these steps in order. Each step has a template — adapt it to the specific dataset.

### Step 1: Understand the Dataset

Before writing any code, answer these questions:

1. **Source**: HuggingFace dataset ID? Local path? Custom download?
2. **Splits**: What splits exist (train, val, test, custom)?
3. **Modality**: Images, video, text-only, multi-image?
4. **Question format**: Multiple-choice, open-ended, fill-blank, yes-no, bbox, custom?
5. **Answer format**: Letter, number, free text, structured (intervals, coordinates)?
6. **Evaluation method**: Exact match? Fuzzy? Interval? LLM judge? Custom metric?
7. **Categories**: Are there sub-tasks or categories for per-group breakdowns?

Inspect the dataset before coding:

```python
from datasets import load_dataset
ds = load_dataset("org/DatasetName", split="test")
print(ds.column_names, len(ds))
for k, v in ds[0].items():
    if k != "image":
        print(f"  {k}: {repr(v)[:200]}")
```

### Step 2: Write process.py

The processor converts raw data into FlagEvalMM's standard `data.json` format.

**Function signature**: `def process(cfg)` — cfg has attributes: `dataset_path`, `split`,
`processed_dataset_path`, `processor`, and optionally `dataset_name`, `anno_file`.

**Output location**: `{cfg.processed_dataset_path}/{cfg.split}/data.json` plus media files.

#### data.json Schema

Each item is a dict. Only `question_id` is strictly required; include others as needed:

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question_id` | str/int | Yes | Unique per item across the entire dataset |
| `question` | str | Yes* | The question text |
| `question_type` | str | No | One of: `multiple-choice`, `fill-in-the-blank`, `yes-no`, `multiple-response`, `cloze`, `short-answer`, `bbox`, `open` |
| `answer` | str/list/dict | No | Ground truth answer |
| `options` | list[str] | No | For multiple-choice: list of option texts |
| `img_path` | str/list[str] | No | Relative path(s) to image(s) |
| `video_path` | str/list[str] | No | Relative path(s) to video(s) |
| `sub_task` | str | No | Category name for per-group breakdowns |

Any extra fields (e.g., `evaluator`, `evaluator_kwargs`, `design`, `image_type`) are
preserved in the annotation dict and available to evaluators.

#### Template: HuggingFace Image QA

```python
import os
import os.path as osp
import json
from datasets import load_dataset


def process(cfg):
    dataset = load_dataset(cfg.dataset_path, split=cfg.split)
    output_dir = osp.join(cfg.processed_dataset_path, cfg.split)
    img_dir = osp.join(output_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    content = []
    for item in dataset:
        entry = {
            "question_id": item["id"],
            "question": item["question"],
            "question_type": "multiple-choice",
            "answer": item["answer"],
            "options": item["options"],
            "img_path": osp.join("img", f"{item['id']}.png"),
        }
        item["image"].save(osp.join(output_dir, entry["img_path"]))
        content.append(entry)

    with open(osp.join(output_dir, "data.json"), "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)
```

#### Multi-Image Handling

When items have multiple images, store paths as a list and add `<image N>` tokens:

```python
entry["img_path"] = []
for i in range(num_images):
    path = f"img/{qid}_{i + 1}.jpg"
    entry["img_path"].append(path)
    item[f"image_{i + 1}"].save(osp.join(output_dir, path))
img_prefix = "".join(f"<image {i + 1}> " for i in range(len(entry["img_path"])))
entry["question"] = img_prefix + entry["question"]
```

#### Video Handling

Set `video_path` instead of `img_path`. Use `VideoDataset` type in the task config.

#### Skipping process.py

If data is already in the right format on disk, skip process.py and use direct paths:

```python
dataset = dict(
    type="VqaBaseDataset",
    data_root="/path/to/data",
    anno_file="data.json",       # or a list of json files
    name="my_task",
)
```

### Step 3: Write the Task Config

The task config is a Python file declaring three things: dataset, evaluator, and optionally
prompt template.

#### Minimal Config (HuggingFace + Standard Evaluator)

```python
config = dict(
    dataset_path="org/DatasetName",
    split="test",
    processed_dataset_path="DatasetName",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(type="PromptTemplate"),
    name="my_task_test",
)

evaluator = dict(type="BaseEvaluator")
```

#### With Custom Evaluator Class

Register external evaluator files from the same directory:

```python
register_evaluator = {"my_evaluator.py": "MyEvaluator"}

dataset = dict(...)
evaluator = dict(type="MyEvaluator", custom_param="value")
```

#### With Config Inheritance (_base_)

Share prompt config across multiple splits by creating a `post_prompt.py`:

```python
# post_prompt.py
def post_prompt_func(annotation: dict, **kwargs):
    return "Answer with the option letter."

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate", post_prompt=post_prompt_func),
)
```

Then inherit in each split config:

```python
# my_task_test.py
_base_ = "post_prompt.py"
config = dict(...)
dataset = dict(type="VqaBaseDataset", config=config, name="my_task_test")
evaluator = dict(...)
```

The child config merges with and overrides the parent. This avoids repeating prompt logic.

#### Per-Category Breakdowns

Add `sub_task` field in data.json and use `detailed_keys`:

```python
evaluator = dict(type="BaseEvaluator", detailed_keys=["sub_task"])
```

### Step 4: Choose Dataset Type

| Type | When to Use |
|------|-------------|
| `VqaBaseDataset` | Image QA, text QA, any standard question-answer (default choice) |
| `VideoDataset` | Video understanding tasks (extends VqaBaseDataset) |
| `Text2ImageBaseDataset` | Text-to-image generation evaluation |
| `Text2VideoBaseDataset` | Text-to-video generation evaluation |
| `RetrievalBaseDataset` | Image/text retrieval tasks |
| Custom class | Complex loading beyond standard patterns (register via `register_dataset`) |

When in doubt, start with `VqaBaseDataset` — it handles images, multi-image, and even
video through the `video_path` field.

### Step 5: Choose Evaluator

| Evaluator | When to Use |
|-----------|-------------|
| `BaseEvaluator` | Multiple-choice, fill-blank, yes-no — auto-detected from `question_type` |
| `BaseEvaluator` + `eval_func` | Custom scoring in a separate `evaluate.py` file |
| `ExtractEvaluator` | Extract answer via LLM, then compare |
| `OpenEvaluator` | LLM-based semantic evaluation for open-ended answers |
| `MmmuEvaluator` | MMMU-specific evaluation |
| `CocoEvaluator` | COCO caption/detection metrics |
| Custom class | Complex evaluation needing stateful tracking or special aggregation |

#### Custom eval_func Template

```python
# evaluate.py
def get_result(annotations: dict, predictions: list) -> dict:
    """
    annotations: dict keyed by question_id (string)
    predictions: list of dicts with "question_id", "answer", etc.
    Returns: dict of metric_name -> value
    """
    correct = 0
    for pred in predictions:
        qid = str(pred["question_id"])
        gt = annotations[qid]
        if pred["answer"].strip() == gt["answer"].strip():
            correct += 1
    return {"accuracy": round(correct / len(predictions) * 100, 2)}
```

#### Custom Evaluator Class Template

Use when you need to override `cal_accuracy` for complex metrics (e.g., interval matching,
IoU, multi-metric tracking):

```python
from flagevalmm.evaluator import BaseEvaluator
from flagevalmm.registry import EVALUATORS

@EVALUATORS.register_module()
class MyEvaluator(BaseEvaluator):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def cal_accuracy(self, annotations, predictions, *args, **kwargs):
        results = {}
        # ... compute metrics from annotations and predictions ...
        return results
```

### Step 6: Design the Prompt Template

`PromptTemplate` supports four parameters, each a string or callable:

| Parameter | Purpose |
|-----------|---------|
| `pre_prompt` | Text prepended before the question |
| `post_prompt` | Text appended after the question (most commonly customized) |
| `examples` | Few-shot examples inserted before the question |
| `prompt_func` | Completely replaces the default prompt assembly |

**Default behavior** (when post_prompt is None): auto-generates instructions based on
`question_type`, with Chinese/English detection.

**Callable pattern** — the most flexible approach:

```python
def post_prompt_func(annotation: dict, **kwargs):
    qtype = annotation.get("question_type", "")
    if qtype == "multiple-choice":
        return "Answer with the option letter only."
    else:
        return "Provide a brief answer."

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate", post_prompt=post_prompt_func),
)
```

The callable receives the full annotation dict, so it can branch on any field.

### Step 7: Test and Verify

#### Quick Validation (32 samples)

```bash
flagevalmm --tasks tasks/<task_name>/<task_name>.py \
           --cfg <model_config>.json \
           --try-run --debug
```

This processes only the first 32 samples — fast enough to catch config errors, data loading
issues, and evaluator bugs before committing to a full run.

#### Full Run

```bash
flagevalmm --tasks tasks/<task_name>/<task_name>.py \
           --cfg <model_config>.json
```

#### Multiple Splits in One Run

```bash
flagevalmm --tasks tasks/<task_name>/split_a.py tasks/<task_name>/split_b.py \
           --cfg <model_config>.json
```

#### Results Location

Results are saved to `{output_dir}/{task_name}/`:

| File | Contents |
|------|----------|
| `{task_name}.json` | Per-sample predictions with model answers |
| `{task_name}_result.json` | Aggregate metrics |
| `{task_name}_evaluated.json` | Predictions annotated with eval results |
| `{task_name}_config.json` | Task configuration snapshot |
| `items/` | Incremental results during inference |

#### Model Config (OpenRouter Example)

```json
{
    "model_name": "openai/gpt-5-mini",
    "api_key": "<your-api-key>",
    "url": "https://openrouter.ai/api/v1/chat/completions",
    "use_cache": false,
    "num_workers": 3,
    "max_image_size": 4718592,
    "max_tokens": 28000,
    "max_long_side": 1000,
    "output_dir": "results/openai/gpt-5-mini"
}
```

#### Evaluation-Only Rerun

If you already have predictions and want to re-run just the evaluator:

```bash
flagevalmm --tasks tasks/<task_name>/<task_name>.py \
           --without-infer \
           --output-dir ./previous_results
```

## Decision Tree

```
Is data on HuggingFace?
├── Yes → config dict + process.py
└── No  → data_root + anno_file (skip process.py)

Question format?
├── Multiple-choice → BaseEvaluator (automatic)
├── Open-ended      → ExtractEvaluator or OpenEvaluator (LLM judge)
├── Custom metric   → eval_func="evaluate.py" (simple) or custom class (complex)
└── Standard NLP    → CocoEvaluator, MmmuEvaluator, etc.

Multiple splits or variants?
├── Yes → _base_ = "post_prompt.py" for shared config
└── No  → inline prompt_template in task config

Modality?
├── Single image → VqaBaseDataset, img_path as string
├── Multi-image  → VqaBaseDataset, img_path as list, add <image N> tokens
├── Video        → VideoDataset, video_path field
└── Text-only    → VqaBaseDataset, no img_path
```
