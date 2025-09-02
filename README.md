# FlagEvalMM: A Flexible Framework for Comprehensive Multimodal Model Evaluation

![FlagEvalMM Logo](assets/logo.png)

[Documentation](https://flagevalmm.readthedocs.io/en/latest/) | [中文文档](https://github.com/flageval-baai/FlagEvalMM/blob/main/README_ZH.md)

## Overview

FlagEvalMM is an open-source evaluation framework designed to comprehensively assess multimodal models. It provides a standardized way to evaluate models that work with multiple modalities (text, images, video) across various tasks and metrics.

## News

- [2025-09-02] Supported by FlagEvalMM, [LRM-Eval](https://github.com/flageval-baai/LRM-Eval) is released. We include the evaluation code of [ROME](https://huggingface.co/datasets/BAAI/ROME) in the tasks/rome directory.

## Key Features

- **Flexible Architecture**: Support for multiple multimodal models and evaluation tasks, including: VQA, image retrieval, text-to-image, etc.
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
