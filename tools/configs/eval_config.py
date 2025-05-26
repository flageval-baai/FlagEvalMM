# 模型列表和后端
model_info = [
    ["/share/project/emllm_mnt.1d/sfs/baaiei/vlm_ckpt/qwen2.5vl/qwen2.5vl_object_ref_20k/checkpoint-500 ", "vllm"]
]

# 要评测的任务
tasks = [
    "tasks/blink/blink_val.py",
    "tasks/where2place/where2place.py"
]

# 输出目录
output_dir = "/root/.cache/flagevalmm/results/multi_gpu_eval"