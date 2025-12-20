# Flagen: A Flexible Multimodal Evaluation Framework For MLLMs and Unified models

![FlagEvalMM Logo](assets/logo.png)

[Documentation](https://flagevalmm.readthedocs.io/en/latest/) | [中文文档](https://github.com/flageval-baai/FlagEvalMM/blob/main/README_ZH.md)

# How to use
We provide a example config file for Bagel model. You can use it to evaluate the Bagel model.

```bash
flagevalmm --cfg model_configs/uni/Bagel.yaml
```

Config file example:
```yaml
tasks: # task configuration
  files: # support multiple tasks
    - file: tasks/t2i/wise/wise.py # task file
      data_root: /share/project/mmdataset/t2i/wise/ # data root
    # - file: tasks/t2i/longtext_bench/longtext_bench.py
    #   data_root: /share/project/mmdataset/t2i/LongText-Bench/
  debug: false # debug mode, only run 32 samples for testing
  try_run: false # try run mode, only run 32 samples for testing
  output_dir: outputs/Bagel_Gen/ # output directory
  num_workers: 8 # number of workers for inference
  skip: false # skip mode, skip the task if it has been run

server: # server configuration
  local_mode: true
  quiet: false

model: # model configuration
  model_name: Bagel
  model_path: /share/project/models/vlm/BAGEL-7B-MoT
  exec: model_zoo/uni/bagel/model_adapter.py

extra_args: # optional arguments for the model
  save_items: true
  num_images: 1
  batch_size: 1
  cfg_scale: 4.0
  resolution: 1024
  num_timesteps: 50
  cfg_interval: [0.4, 1.0]
  cfg_renorm_min: 0.0
  timestep_shift: 3.0
  think: false
  think_simple: false
```


## Overview

Flagen is an open-source evaluation framework designed to comprehensively assess MLLMs and Unified models. It provides a standardized way to evaluate models that work with multiple modalities (text, images, video) across various tasks and metrics.

## Key Features

- **Flexible Architecture**: Support for multiple multimodal models and evaluation tasks, including: VQA, text-to-image, etc.
- **Comprehensive Benchmarks and Metrics**: Support new and commonly used benchmarks and metrics.
- **Extensive Model Support**: The model_zoo provides inference support for a wide range of popular multimodal models including QWenVL and LLaVA. Additionally, it offers seamless integration with API-based models such as GPT, Claude, and HuanYuan.
- **Extensible Design**: Easily extendable to incorporate new models, benchmarks, and evaluation metrics.

## Getting Started

[Basic Installation](https://flagevalmm.readthedocs.io/en/latest/installation.html)

[Quick Start](https://flagevalmm.readthedocs.io/en/latest/quickstart.html)

[Usage Guide](https://flagevalmm.readthedocs.io/en/latest/usage.html)

## Citation

```bibtex
@inproceedings{he-etal-2025-flagevalmm,
    title = "FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation",
    author = "He, Zheqi  and
      Liu, Yesheng  and
      Zheng, Jing-Shu  and
      Li, Xuejing  and
      Yao, Jin-Ge  and
      Qin, Bowen  and
      Xuan, Richeng  and
      Yang, Xi",
    editor = "Mishra, Pushkar  and
      Muresan, Smaranda  and
      Yu, Tao",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-demo.6/",
    pages = "51--61",
    ISBN = "979-8-89176-253-4"
}
```
