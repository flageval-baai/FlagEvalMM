#!/bin/bash

tasks=tasks/vsi_bench/vsi_bench_tiny.py
model_dir=/share/project/caomingyu/vlm_ckpt
model_names=(
/share/project/caomingyu/vlm_ckpt/qwen2.5vl_7b_Robobrain_final_1757K_stage2/checkpoint-1000
/share/project/caomingyu/vlm_ckpt/qwen2.5vl_7b_Robobrain_final_1757K_stage2/checkpoint-2000
/share/project/caomingyu/vlm_ckpt/qwen2.5vl_7b_Robobrain_final_1757K_stage2/checkpoint-3000
/share/project/caomingyu/vlm_ckpt/qwen2.5vl_7b_Robobrain_final_1757K_stage2/checkpoint-4000
)

export PYTHONPATH=/share/project/tanhuajie/FlagEvalMM

for model_name in "${model_names[@]}"; do
    echo "Evaluating model: $model_name"
    
    python flagevalmm/eval.py --tasks $tasks \
        --exec model_zoo/vlm/qwen_vl/model_adapter.py \
        --model $model_dir/$model_name \
        --num-workers 20 \
        --output-dir /share/project/tanhuajie/FlagEvalMM/results/$model_name \
        --backend vllm \
        --extra-args "--limit-mm-per-prompt image=256 --max-model-len 32768 --gpu_memory_utilization 0.7 --trust-remote-code --mm-processor-kwargs '{\"max_dynamic_patch\":2}'"
    
    echo "Finished evaluating $model_name"
    echo "----------------------------------"
done