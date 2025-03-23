# FlagEvalMM Tools

This directory contains utility tools for working with FlagEvalMM.

## Batch Model Execution

The `run_models.py` script allows you to run multiple models in parallel on a multi-GPU system with automatic GPU management. This is particularly useful for evaluating multiple models on the same benchmark tasks.

### Features

- Dynamic GPU allocation based on model requirements
- Parallel execution of models to maximize resource utilization
- Automatic logging of model outputs
- Graceful handling of interrupted runs
- Support for specifying a common model directory (required)

### Usage

The script requires specifying both a configuration file and the model directory:

```bash
python tools/run_models.py --config tools/configs/example_batch.py --models-base-dir /share/projset/models/vlm
```

You can also specify a custom model configuration directory (defaults to model_configs/open):

```bash
python tools/run_models.py --config tools/configs/example_batch.py --models-base-dir /share/projset/models/vlm --cfg-dir model_configs/open
```

### Configuration Format

The configuration file should be in Python format with the following structure:

```python
# List of models and their backends
model_info = [
    ["Model1-Name", "backend_type"],
    ["Model2-Name", "custom/adapter.py"],
    # ...
]

# List of tasks to evaluate
tasks = [
    "tasks/task1.py",
    "tasks/task2.py",
    # ...
]

# Optional output directory
output_dir = "./results/batch_eval"
```

Each model also needs a JSON configuration file in the specified `cfg-dir` directory matching the model name. For example:

```json
{
    "backend": "vllm",
    "extra_args": "--tensor-parallel-size 4 --max-model-len 32768"
}
```

The script will automatically add the model path using the specified `--model-dir`. For example, if you run with `--model-dir /path/to/models`, the actual model path will be set to `/path/to/models/Model-Name`.

Note: HuggingFace model references (like "Qwen/Qwen2-VL-72B-Instruct") will be preserved as-is.

### GPU Requirements

The script automatically allocates GPUs based on predefined requirements in the `GPU_REQUIREMENTS` dictionary. Models not listed there will use 1 GPU by default.

### Example

Run multiple models on MMMU and MMVET tasks:

```bash
# Run with models located in a common directory
python tools/run_models.py --config tools/configs/example_batch.py --model-dir models/vlm
```

This will evaluate all models specified in the config on all the listed tasks with appropriate GPU allocation, using the model files from the specified model directory.

