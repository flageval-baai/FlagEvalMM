FlagEvalMM Documentation
========================

.. image:: ../assets/logo.png
   :alt: FlagEvalMM Logo
   :align: center
   :width: 400px

.. centered:: **A Flexible Framework for Comprehensive Multimodal Model Evaluation**

FlagEvalMM is an open-source evaluation framework designed to comprehensively assess multimodal models. 
It provides a standardized way to evaluate models that work with multiple modalities (text, images, video) 
across various tasks and metrics.

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   usage

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   configuration
   tasks
   tools

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/flagevalmm
   api/models
   api/evaluator
   api/dataset
   api/common

.. toctree::
   :maxdepth: 1
   :caption: Additional Information

   GitHub Repository <https://github.com/flageval-baai/FlagEvalMM>

Key Features
------------

* **Flexible Architecture**: Support for multiple multimodal models and evaluation tasks, including VQA, image retrieval, text-to-image, etc.
* **Comprehensive Benchmarks and Metrics**: Support for new and commonly used benchmarks and metrics.
* **Extensive Model Support**: The model_zoo provides inference support for a wide range of popular multimodal models including QwenVL and LLaVA. Additionally, it offers seamless integration with API-based models such as GPT, Claude, and HuanYuan.
* **Extensible Design**: Easily extendable to incorporate new models, benchmarks, and evaluation metrics.

Quick Start
-----------

Install FlagEvalMM:

.. code-block:: bash

   git clone https://github.com/flageval-baai/FlagEvalMM.git
   cd FlagEvalMM
   pip install -e .

Run a basic evaluation:

.. code-block:: bash

   flagevalmm --tasks tasks/mmmu/mmmu_val.py \
           --exec model_zoo/vlm/api_model/model_adapter.py \
           --model llava-hf/llava-onevision-qwen2-7b-ov-chat-hf \
           --num-workers 8 \
           --output-dir ./results/llava-onevision-qwen2-7b-ov-chat-hf \
           --backend vllm \
           --extra-args "--limit-mm-per-prompt image=10 --max-model-len 32768"

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search` 