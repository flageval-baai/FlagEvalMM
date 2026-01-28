Usage Guide
===========

This guide covers the various ways to use FlagEvalMM for multimodal model evaluation.

Command Line Interface
----------------------

FlagEvalMM provides a command-line interface through the ``flagevalmm`` command.

Basic Syntax
~~~~~~~~~~~~

.. code-block:: bash

   flagevalmm [OPTIONS] --tasks TASK_FILES --exec MODEL_ADAPTER --model MODEL_NAME

Required Arguments
~~~~~~~~~~~~~~~~~~

* ``--tasks``: Path(s) to task configuration files
* ``--exec``: Path to model adapter script
* ``--model``: Model name or path

Optional Arguments
~~~~~~~~~~~~~~~~~~

* ``--num-workers``: Number of parallel workers (default: 1)
* ``--output-dir``: Output directory for results (default: ./results)
* ``--backend``: Inference backend (vllm, transformers, sglang, lmdeploy)
* ``--extra-args``: Additional arguments for the backend
* ``--cfg``: Configuration file path
* ``--api-key``: API key for API-based models
* ``--url``: API endpoint URL
* ``--use-cache``: Enable response caching
* ``--without-infer``: Skip inference and only run evaluation
* ``--try-run``: Run in debug mode with limited samples

Configuration Files
-------------------

JSON Configuration
~~~~~~~~~~~~~~~~~~

You can use JSON configuration files to simplify complex commands:

.. code-block:: json

   {
       "model_name": "Qwen/Qwen2-VL-7B-Instruct",
       "api_key": "EMPTY",
       "output_dir": "./results/qwen2-vl-7b",
       "num_workers": 8,
       "backend": "vllm",
       "extra_args": "--limit-mm-per-prompt image=10 --max-model-len 32768"
   }

Task Configuration
~~~~~~~~~~~~~~~~~~

Task configuration files define the dataset and evaluation settings:

.. code-block:: python

   # Example task configuration
   dataset = dict(
       type='MMBenchDataset',
       data_file='path/to/dataset.json',
       name='mmbench_dev_en',
       debug=False
   )

   evaluator = dict(
       type='MultipleChoiceEvaluator'
   )

Evaluation Modes
----------------

Single Task Evaluation
~~~~~~~~~~~~~~~~~~~~~~~

Evaluate a single task:

.. code-block:: bash

   flagevalmm --tasks tasks/mmmu/mmmu_val.py \
           --exec model_zoo/vlm/api_model/model_adapter.py \
           --model llava-hf/llava-onevision-qwen2-7b-ov-chat-hf \
           --output-dir ./results/single-task

Multi-Task Evaluation
~~~~~~~~~~~~~~~~~~~~~~

Evaluate multiple tasks in one run:

.. code-block:: bash

   flagevalmm --tasks tasks/mmmu/mmmu_val.py tasks/mmvet/mmvet_v2.py \
           --exec model_zoo/vlm/api_model/model_adapter.py \
           --model llava-hf/llava-onevision-qwen2-7b-ov-chat-hf \
           --output-dir ./results/multi-task

Batch Model Evaluation
~~~~~~~~~~~~~~~~~~~~~~~

Use the batch evaluation tool for multiple models:

.. code-block:: bash

   python tools/run_models.py --config tools/configs/batch_config.py --models-base-dir /path/to/models

Backend-Specific Usage
----------------------

VLLM Backend
~~~~~~~~~~~~

.. code-block:: bash

   flagevalmm --tasks tasks/mmmu/mmmu_val.py \
           --exec model_zoo/vlm/api_model/model_adapter.py \
           --model llava-hf/llava-onevision-qwen2-7b-ov-chat-hf \
           --backend vllm \
           --extra-args "--limit-mm-per-prompt image=10 --max-model-len 32768"

Multi-GPU with VLLM:

.. code-block:: bash

   flagevalmm --tasks tasks/mmmu/mmmu_val.py \
           --exec model_zoo/vlm/api_model/model_adapter.py \
           --model Qwen/Qwen2-VL-72B-Instruct \
           --backend vllm \
           --extra-args "--tensor-parallel-size 4 --max-model-len 32768"

Transformers Backend
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   flagevalmm --tasks tasks/mmmu/mmmu_val.py \
           --exec model_zoo/vlm/llama-vision/model_adapter.py \
           --model meta-llama/Llama-3.2-11B-Vision-Instruct \
           --output-dir ./results/llama-vision

SGLang Backend
~~~~~~~~~~~~~~

.. code-block:: bash

   flagevalmm --tasks tasks/mmmu/mmmu_val.py \
           --exec model_zoo/vlm/api_model/model_adapter.py \
           --model llava-hf/llava-onevision-qwen2-7b-ov-chat-hf \
           --backend sglang \
           --extra-args "--mem-fraction-static 0.8"

API-Based Models
~~~~~~~~~~~~~~~~

OpenAI GPT models:

.. code-block:: bash

   flagevalmm --tasks tasks/mmmu/mmmu_val.py \
           --exec model_zoo/vlm/api_model/model_adapter.py \
           --model gpt-4o-mini \
           --url https://api.openai.com/v1/chat/completions \
           --api-key $OPENAI_API_KEY \
           --use-cache

Output and Results
------------------

Result Structure
~~~~~~~~~~~~~~~~

After evaluation, results are organized as follows:

.. code-block:: text

   output_dir/
   ├── model_name/
   │   ├── task_name/
   │   │   ├── results.json          # Main results
   │   │   ├── detailed_results.json # Per-sample results
   │   │   ├── predictions.json      # Model predictions
   │   │   └── logs/                 # Evaluation logs
   │   └── summary.json              # Cross-task summary

Result Formats
~~~~~~~~~~~~~~

The main results file contains:

.. code-block:: json

   {
       "accuracy": 85.2,
       "total_samples": 1000,
       "correct_samples": 852,
       "subject_scores": {
           "math": 78.5,
           "science": 89.3,
           "history": 87.1
       },
       "metadata": {
           "model": "llava-hf/llava-onevision-qwen2-7b-ov-chat-hf",
           "task": "mmmu_val",
           "timestamp": "2024-01-01T12:00:00"
       }
   }

Advanced Usage
--------------

Custom Model Adapters
~~~~~~~~~~~~~~~~~~~~~~

Create custom model adapters for new models by extending the base adapter:

.. code-block:: python

   from model_zoo.base_adapter import BaseAdapter

   class CustomModelAdapter(BaseAdapter):
       def __init__(self, model_path):
           super().__init__(model_path)
           # Custom initialization

       def predict(self, inputs):
           # Custom prediction logic
           return predictions

Custom Evaluation Metrics
~~~~~~~~~~~~~~~~~~~~~~~~~~

Define custom evaluators for specific tasks:

.. code-block:: python

   from flagevalmm.evaluator import BaseEvaluator
   from flagevalmm.registry import EVALUATORS

   @EVALUATORS.register_module()
   class CustomEvaluator(BaseEvaluator):
       def evaluate(self, predictions, annotations):
           # Custom evaluation logic
           return results

Performance Optimization
------------------------

Memory Management
~~~~~~~~~~~~~~~~~

* Use ``--num-workers`` to control parallel processing
* Adjust batch sizes in model adapters
* Use gradient checkpointing for large models

GPU Utilization
~~~~~~~~~~~~~~~

* Use ``--tensor-parallel-size`` for multi-GPU inference
* Monitor GPU memory usage
* Consider model quantization for memory efficiency

Caching
~~~~~~~

* Enable ``--use-cache`` to avoid re-computation
* Cache is stored in ``~/.cache/flagevalmm`` by default
* Clear cache periodically to save disk space

Troubleshooting
---------------

Common Issues
~~~~~~~~~~~~~

**Out of Memory**: Reduce batch size or use model sharding
**Slow Inference**: Check GPU utilization and consider using VLLM backend
**Model Loading Issues**: Verify model path and access permissions
**Task Configuration Errors**: Check task file syntax and required fields

Debug Mode
~~~~~~~~~~

Use ``--try-run`` for quick debugging with limited samples:

.. code-block:: bash

   flagevalmm --tasks tasks/mmmu/mmmu_val.py \
           --exec model_zoo/vlm/api_model/model_adapter.py \
           --model llava-hf/llava-onevision-qwen2-7b-ov-chat-hf \
           --try-run 