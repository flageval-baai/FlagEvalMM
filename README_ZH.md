# FlagEvalMM: 一个灵活的多模态模型综合评测框架

![FlagEvalMM Logo](assets/logo.png)

## 概述

FlagEvalMM 是一个开源的评测框架，旨在对多模态模型进行全面评估。它为评估处理多种模态（文本、图像、视频）的模型提供了标准化的方法，可以跨多种任务和指标进行评测。

## 主要特点

- **灵活的架构**：支持多种多模态模型和评估任务，包括：视觉问答(VQA)、图像检索、文本生成图像等。
- **全面的基准和指标**：支持新的和常用的基准测试和评估指标。
- **广泛的模型支持**：model_zoo 提供了对众多流行多模态模型的推理支持，包括 QwenVL 和 LLaVA。此外，还提供与基于 API 的模型（如 GPT、Claude 和 HuanYuan）的无缝集成。
- **可扩展设计**：易于扩展以整合新的模型、基准和评估指标。

## 安装

### 基础安装

```bash
git clone https://github.com/flageval-baai/FlagEvalMM.git
cd FlagEvalMM
pip install -e .
```

### 可选依赖

FlagEvalMM 支持多个推理后端引擎。根据需要安装：

#### VLLM 后端

目前（2025年7月28日），我们推荐使用 vllm==0.8.5.post1 和 torch==2.6.0 以获得最佳的推理性能和稳定性。

```bash
pip install vllm==0.8.5.post1
```

#### SGLang 后端

```bash
pip install --upgrade pip
pip install "sglang[all]"
pip install flashinfer -i https://flashinfer.ai/whl/cu121/torch2.4/
```

详细安装说明请参考 [SGLang 官方文档](https://sgl-project.github.io/start/install.html)。

#### LMDeploy 后端

```bash
pip install lmdeploy
```

详细安装说明请参考 [LMDeploy 官方文档](https://lmdeploy.readthedocs.io/en/latest/)。


#### FlagScale 后端

```bash
git clone https://github.com/FlagOpen/FlagScale.git
cd FlagScale/install
./install-requirements.sh --env inference
cd vllm
pip install .
```

详细安装说明请参考 [FlagScale 官方文档](https://github.com/FlagOpen/FlagScale)。


#### Transformers

为获得最佳性能，我们建议安装 flash-attention

```bash
pip install flash-attn --no-build-isolation
```

### 关于 API 密钥

如果你想使用 GPT 评估某些任务（如 charxiv、math_verse 等），需要设置以下环境变量：

```bash
export FLAGEVAL_API_KEY=$YOUR_OPENAI_API_KEY
export FLAGEVAL_BASE_URL="https://api.openai.com/v1"
```

## 使用方法

FlagevalMM 支持一键评测：

使用 vllm 作为后端的 llava 示例：

```bash
flagevalmm --tasks tasks/mmmu/mmmu_val.py \
        --exec model_zoo/vlm/api_model/model_adapter.py \
        --model llava-hf/llava-onevision-qwen2-7b-ov-chat-hf \
        --num-workers 8 \
        --output-dir ./results/llava-onevision-qwen2-7b-ov-chat-hf \
        --backend vllm \
        --extra-args "--limit-mm-per-prompt image=10 --max-model-len 32768"
```

`--tasks` 是要评测的任务路径，支持多个任务。

`--exec` 是执行模型推理的代码。

`--model` 模型路径，可以是 huggingface 上的模型名称或你自己的模型路径。建议提前从 huggingface 下载模型。

`--extra-args` 是 vllm 服务器的参数。

对于像 Qwen2-VL-72B 这样使用 vllm 的大型模型，可以通过 `--tensor-parallel-size` 参数启用多 GPU 推理：

```bash
flagevalmm --tasks tasks/mmmu_pro/mmmu_pro_standard_test.py tasks/ocrbench/ocrbench_test.py \
        --exec model_zoo/vlm/api_model/model_adapter.py \
        --model Qwen/Qwen2-VL-72B-Instruct \
        --num-workers 8 \
        --output-dir ./results/Qwen2-VL-72B-Instruct \
        --backend vllm \
        --extra-args "--limit-mm-per-prompt image=18 --tensor-parallel-size 4 --max-model-len 32768 --trust-remote-code \
        --mm-processor-kwargs '{\"max_dynamic_patch\":4}'"
```

由于参数可能相当复杂，建议使用 JSON 配置文件。示例如下：

创建一个名为 `qwen2_vl_72b_instruct.json` 的配置文件：

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

这样可以简化你的评测命令为：

```bash
flagevalmm --tasks tasks/mmmu_pro/mmmu_pro_standard_test.py tasks/ocrbench/ocrbench_test.py \
        --exec model_zoo/vlm/api_model/model_adapter.py \
        --cfg qwen2_vl_72b_instruct.json
```

不使用 vllm 的模型评测示例（使用 transformers）：

```bash
flagevalmm --tasks tasks/mmmu/mmmu_val.py \
        --exec model_zoo/vlm/llama-vision/model_adapter.py \
        --model meta-llama/Llama-3.2-11B-Vision-Instruct \
        --output-dir ./results/Meta-Llama-3.2-11B-Vision-Instruct
```

对于直接使用 transformers 的模型，不需要 `--backend` 和 `--extra-args` 参数。更多模型示例可以在 `model_zoo/vlm/` 目录中找到。

评测 gpt 风格模型的示例：

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

`--use-cache` 是可选的，它会缓存模型输出，相同设置下的相同问题将从缓存中获取结果。

## 分别启动数据服务器和评测

上面是一键评测，你也可以分别启动数据服务器和进行评测。以评测 qwen-vl-2 模型为例：

```bash
# 启动数据服务器
python flagevalmm/server/run_server.py --tasks tasks/charxiv/charxiv_val.py --output-dir ./results/qwenvl2-7b --port 11823 
```

### 分别评测

这将在端口 11823 上启动服务器，数据服务器将一直运行直到你停止它。

```bash
python flagevalmm/eval.py --output-dir ./results/qwenvl2-7b --tasks tasks/charxiv/charxiv_val.py --model your_model_path/Qwen2-VL-7B-Instruct/ --exec model_zoo/vlm/qwen_vl/model_adapter.py --server-port 11823
```

这将在数据服务器上评测模型。
如果你已经从数据服务器生成了结果，可以直接评测结果：

```bash
python flagevalmm/eval.py --output-dir ./results/qwenvl2-7b --exec model_zoo/vlm/qwen_vl/model_adapter.py --tasks tasks/charxiv/charxiv_val.py --without-infer
```

## 添加你自己的任务

如果你想添加自己的数据集，请参照该文档：[添加你自己的任务](./tasks/README.md)

## 关于数据

在任务配置文件中，我们默认从 HuggingFace 下载数据集。如果你需要使用自己的数据集，请在配置文件中将 `dataset_path` 设置为你的数据集路径。

FlagEvalMM 会预处理来自各种来源的数据，处理后的数据默认存储在 `~/.cache/flagevalmm` 目录中。你可以通过修改 `FLAGEVALMM_CACHE` 环境变量来更改数据存储路径。

## 引用

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