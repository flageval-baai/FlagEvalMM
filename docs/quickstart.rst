Quick Start
===========

This guide will get you up and running with FlagEvalMM in minutes.

Your First Evaluation
----------------------

After installation, you can run your first evaluation with a simple command:

.. code-block:: bash

   flagevalmm --tasks tasks/mmmu/mmmu_val.py \
           --exec model_zoo/vlm/api_model/model_adapter.py \
           --model llava-hf/llava-onevision-qwen2-7b-ov-chat-hf \
           --num-workers 8 \
           --output-dir ./results/llava-onevision-qwen2-7b-ov-chat-hf \
           --backend vllm \
           --extra-args "--limit-mm-per-prompt image=10 --max-model-len 32768"

Command Breakdown
-----------------

Let's break down what each parameter does:

* ``--tasks``: Path to the task configuration file you want to evaluate
* ``--exec``: Script to adapt the model for evaluation
* ``--model``: Model name from Hugging Face or local path
* ``--num-workers``: Number of parallel workers for evaluation
* ``--output-dir``: Directory to save evaluation results
* ``--backend``: Inference backend (vllm, transformers, sglang, etc.)
* ``--extra-args``: Additional arguments for the backend

Basic Examples
--------------

Example 1: Evaluating with VLLM Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   flagevalmm --tasks tasks/mmmu/mmmu_val.py \
           --exec model_zoo/vlm/api_model/model_adapter.py \
           --model llava-hf/llava-onevision-qwen2-7b-ov-chat-hf \
           --num-workers 8 \
           --output-dir ./results/llava \
           --backend vllm

Example 2: Evaluating with Transformers Backend
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   flagevalmm --tasks tasks/mmmu/mmmu_val.py \
           --exec model_zoo/vlm/llama-vision/model_adapter.py \
           --model meta-llama/Llama-3.2-11B-Vision-Instruct \
           --output-dir ./results/llama-vision

Example 3: Evaluating API-based Models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   flagevalmm --tasks tasks/mmmu/mmmu_val.py \
           --exec model_zoo/vlm/api_model/model_adapter.py \
           --model gpt-4o-mini \
           --num-workers 4 \
           --url https://api.openai.com/v1/chat/completions \
           --api-key $OPENAI_API_KEY \
           --output-dir ./results/gpt-4o-mini \
           --use-cache

Using Configuration Files
--------------------------

For complex evaluations, you can use JSON configuration files:

Create a config file ``config.json``:

.. code-block:: json

   {
       "model_name": "Qwen/Qwen2-VL-72B-Instruct",
       "api_key": "EMPTY",
       "output_dir": "./results/Qwen2-VL-72B-Instruct",
       "min_short_side": 28,
       "num_workers": 8,
       "backend": "vllm",
       "extra_args": "--limit-mm-per-prompt image=18 --tensor-parallel-size 4 --max-model-len 32768 --trust-remote-code --mm-processor-kwargs '{\"max_dynamic_patch\":4}'"
   }

Then run:

.. code-block:: bash

   flagevalmm --tasks tasks/mmmu_pro/mmmu_pro_standard_test.py \
           --exec model_zoo/vlm/api_model/model_adapter.py \
           --cfg config.json

Multi-GPU Setup
---------------

For large models that require multiple GPUs:

.. code-block:: bash

   flagevalmm --tasks tasks/mmmu_pro/mmmu_pro_standard_test.py \
           --exec model_zoo/vlm/api_model/model_adapter.py \
           --model Qwen/Qwen2-VL-72B-Instruct \
           --num-workers 8 \
           --output-dir ./results/Qwen2-VL-72B-Instruct \
           --backend vllm \
           --extra-args "--tensor-parallel-size 4 --max-model-len 32768"

Multiple Tasks
--------------

You can evaluate multiple tasks in a single run:

.. code-block:: bash

   flagevalmm --tasks tasks/mmmu/mmmu_val.py tasks/mmvet/mmvet_v2.py tasks/ocrbench/ocrbench_test.py \
           --exec model_zoo/vlm/api_model/model_adapter.py \
           --model llava-hf/llava-onevision-qwen2-7b-ov-chat-hf \
           --output-dir ./results/multi-task \
           --backend vllm

Understanding Results
---------------------

After evaluation completes, you'll find results in the specified output directory:

.. code-block:: text

   results/
   ├── model_name/
   │   ├── task_name/
   │   │   ├── results.json          # Main results file
   │   │   ├── detailed_results.json # Detailed per-sample results
   │   │   └── logs/                 # Evaluation logs
   │   └── summary.json              # Overall summary

The main metrics and scores will be in ``results.json``.

Next Steps
----------

Now that you've run your first evaluation, you might want to:

* Learn more about :doc:`configuration` options
* Explore available :doc:`tasks` and benchmarks
* Check out the :doc:`usage` guide for more complex scenarios

.. tip::
   Start with smaller tasks and models to familiarize yourself with the framework before running large-scale evaluations. 