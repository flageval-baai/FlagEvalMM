# FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation

![FlagEvalMM Logo](assets/logo.png)

## Overview

FlagEvalMM is an open-source evaluation framework designed to comprehensively assess multimodal models. It provides a standardized way to evaluate models that work with multiple modalities (text, images, video) across various tasks and metrics.

For Chinese documentation, please refer to [README_ZH.md](README_ZH.md).

## Key Features

- **Flexible Architecture**: Support for multiple multimodal models and evaluation tasks, including: VQA, image retrieval, text-to-image, etc.
- **Comprehensive Benchmarks and Metrics**: Support new and commonly used benchmarks and metrics.
- **Extensive Model Support**: The model_zoo provides inference support for a wide range of popular multimodal models including QWenVL and LLaVA. Additionally, it offers seamless integration with API-based models such as GPT, Claude, and HuanYuan.
- **Extensible Design**: Easily extendable to incorporate new models, benchmarks, and evaluation metrics.

## Installation

### Basic Installation

```bash
git clone https://github.com/flageval-baai/FlagEvalMM.git
cd FlagEvalMM
pip install -e .
```

### Optional Dependencies

FlagEvalMM supports multiple backend engines for inference. Install the ones you plan to use:

#### VLLM Backend

Currently(November 30, 2024), we recommend using vllm==0.6.3.post1 and torch==2.4.0 for optimal inference performance and stability.

```bash
pip install vllm==0.6.3.post1
```

#### SGLang Backend

```bash
pip install --upgrade pip
pip install "sglang[all]"
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

For detailed installation instructions, please refer to the [official SGLang documentation](https://sgl-project.github.io/start/install.html).

#### LMDeploy Backend

```bash
pip install lmdeploy
```

For detailed installation instructions, please refer to the [official LMDeploy documentation](https://lmdeploy.readthedocs.io/en/latest/).


#### FlagScale Backend

```bash
git clone https://github.com/FlagOpen/FlagScale.git
cd FlagScale/install
./install-requirements.sh --env inference
cd vllm
pip install .
```

For detailed installation instructions, please refer to the [official FlagScale documentation](https://lmdeploy.readthedocs.io/en/latest/).

#### Transformers

For optimal performance for transformers, we recommend installing flash-attention

```bash
pip install flash-attn --no-build-isolation
```

### About API keys

If you want to evaluate some tasks by GPT (like charxiv, math_verse, etc.), you need to set the following environment variables:

```bash
export FLAGEVAL_API_KEY=$YOUR_OPENAI_API_KEY
export FLAGEVAL_BASE_URL="https://api.openai.com/v1"
```

## Usage

FlagevalMM supports one-key evaluation:

An example of llava with vllm as backend:

```bash
flagevalmm --tasks tasks/mmmu/mmmu_val.py \
        --exec model_zoo/vlm/api_model/model_adapter.py \
        --model llava-hf/llava-onevision-qwen2-7b-ov-chat-hf \
        --num-workers 8 \
        --output-dir ./results/llava-onevision-qwen2-7b-ov-chat-hf \
        --backend vllm \
        --extra-args "--limit-mm-per-prompt image=10 --max-model-len 32768"
```

`--tasks` is the path of the task you want to evaluate, witch supports multiple tasks.

`--exec` is the script to adapt the model.

`--model` can be the model name in huggingface or your own model path. It is recommended to download the model from huggingface in advance.

`--extra-args` are the parameters for the vllm server.

For large models like Qwen2-VL-72B that use vllm, you can enable multi-GPU inference with the `--tensor-parallel-size` parameter:

```bash
flagevalmm --tasks tasks/mmmu_pro/mmmu_pro_standard_test.py tasks/ocrbench/ocrbench_test.py \
        --exec model_zoo/vlm/api_model/model_adapter.py \
        --model Qwen/Qwen2-VL-72B-Instruct \
        --num-workers 8 \
        --output-dir ./results/Qwen2-VL-72B-Instruct \
        --backend vllm \
        --extra-args "--limit-mm-per-prompt image=18 --tensor-parallel-size 4 --max-model-len 32768 --trust-remote-code --mm-processor-kwargs '{\"max_dynamic_patch\":4}'"
```

Since the parameters can be quite complex, it's recommended to use a JSON config file instead. Here's an example:

Create a config file named `qwen2_vl_72b_instruct.json`:

```json
{
    "model_name": "Qwen/Qwen2-VL-72B-Instruct",
    "api_key": "EMPTY",
    "output_dir": "./results/Qwen2-VL-72B-Instruct",
    "min_short_side": 28,
    "num_workers": 8,
    "backend": "vllm",
    "extra_args": "--limit-mm-per-prompt image=18 --tensor-parallel-size 4 --max-model-len 32768 --trust-remote-code --mm-processor-kwargs '{\"max_dynamic_patch\":4}'"
}
```

This simplifies your evaluation command to:

```bash
flagevalmm --tasks tasks/mmmu_pro/mmmu_pro_standard_test.py tasks/ocrbench/ocrbench_test.py \
        --exec model_zoo/vlm/api_model/model_adapter.py \
        --cfg qwen2_vl_72b_instruct.json
```

## Multi-task and Multi-model Evaluation

For evaluating multiple models on the same benchmarks, FlagEvalMM provides a batch execution tool with automatic GPU management:

```bash
python tools/run_models.py --config tools/configs/example_batch.py --models-base-dir /path/to/models
```

The batch configuration file format:

```python
# List of models and their backends
model_info = [
    ["Qwen2-VL-7B-Instruct", "api_model"],
    ["InternVL2-8B", "api_model"],
    ["Phi-3.5-vision-instruct", "Phi_3.5_v"],
]

# List of tasks to evaluate
tasks = [
    "tasks/mmmu/mmmu_val.py",
    "tasks/mmvet/mmvet_v2.py",
]

# Optional: Default output directory for results
output_dir = "./results/batch_eval" 
```

The tool automatically allocates GPUs based on model requirements, runs models in parallel, and logs outputs to separate files. For more details, see `tools/README.md`.

Example of evaluating models without vllm (using transformers instead):

```bash
flagevalmm --tasks tasks/mmmu/mmmu_val.py \
        --exec model_zoo/vlm/llama-vision/model_adapter.py \
        --model meta-llama/Llama-3.2-11B-Vision-Instruct \
        --output-dir ./results/Meta-Llama-3.2-11B-Vision-Instruct
```

For models using transformers directly, the `--backend` and `--extra-args` parameters are not required. Additional model examples can be found in the `model_zoo/vlm/` directory.

Example of evaluating gpt-style models:

```bash
flagevalmm --tasks tasks/mmmu/mmmu_val.py \
        --exec model_zoo/vlm/api_model/model_adapter.py \
        --model gpt-4o-mini \
        --num-workers 4 \
        --url https://api.openai.com/v1/chat/completions \
        --api-key $OPENAI_API_KEY \
        --output-dir ./results/gpt-4o-mini \
        --use-cache
```

`--use-cache` is optional, it will cache the model outputs, the same question with the same model setting will get results from cache.

## Start data server and evaluate seperately

Above is one-key evaluation, you can also start data server seperately and evaluate seperately.Example of evaluating qwen-vl-2 model:

```bash
# Start data server
python flagevalmm/server/run_server.py --tasks tasks/charxiv/charxiv_val.py --output-dir ./results/qwenvl2-7b --port 11823 
```

### Evaluate seperately

This will start a server on port 11823, and the data server will be running until you stop it.

```bash
python flagevalmm/eval.py --output-dir ./results/qwenvl2-7b --tasks tasks/charxiv/charxiv_val.py --model your_model_path/Qwen2-VL-7B-Instruct/ --exec model_zoo/vlm/qwen_vl/model_adapter.py --server-port 11823
```

This will evaluate the model on the data server.
If you have generated results from the data server, you can directly evaluate the results by:

```bash
python flagevalmm/eval.py --output-dir ./results/qwenvl2-7b --tasks tasks/charxiv/charxiv_val.py --without-infer
```

## About Data

In the task configuration file, we download datasets from HuggingFace by default. If you need to use your own dataset, please set the `dataset_path` to your dataset path in the configuration file.

FlagEvalMM will preprocess data from various sources, and the processed data will be stored in the `~/.cache/flagevalmm` directory by default. You can change the data storage path by modifying the `FLAGEVALMM_CACHE` environment variable.

## Citation

```bibtex
@misc{flagevalmm,
    author = {Zheqi He, Yesheng Liu, Jingshu Zheng, Bowen Qin, Jinge Yao, Richen Xuan and Xi Yang},
    title = {FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation},
    year = {2024},
    publisher = {Zenodo},
    version = {0.3.4},
    url = {https://github.com/flageval-baai/FlagEvalMM}
}
```
