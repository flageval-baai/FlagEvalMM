# Video-SafetyBench: A Benchmark for Safety Evaluation of Video LVLMs

This task is for inference and evaluation of [Video-SafetyBench](https://arxiv.org/abs/2505.11842).

We use the [Qwen2.5-72B-Instruct](https://huggingface.co/Qwen/Qwen2.5-72B-Instruct) as the judge model, make sure you have enough GPU memory to load the model.

## Usage

1.Model inference

```bash
flagevalmm --tasks  tasks/video_safety_bench/video_safety_harmful.py tasks/video_safety_bench/video_safety_benign.py --output-dir output  --cfg model_configs/video_safe/Qwen2.5-VL-7B-Instruct.json --backend vllm
```

2.Evaluation

```bash
flagevalmm --tasks  tasks/video_safety_bench/video_safety_harmful.py tasks/video_safety_bench/video_safety_benign.py --output-dir output  --cfg model_configs/video_safe/Qwen2.5-VL-7B-Instruct.json --backend vllm --wi
```
