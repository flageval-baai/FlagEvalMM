  # tasks/mmmu/mmmu_val.py \
# tasks/ocrbench/ocrbench_test.py \
# tasks/text_vqa/text_vqa_val.py
# tasks/vg/vg_task.py
# tasks/vsi_bench/vsi_bench_tiny/test.py
# tasks/refcoco/refcoco_val.py


# model_zoo/vlm/qwen_vl/model_adapter.py
# tasks/robovqa/robovqa_test.py
# tasks/openeqa/openeqa_test.py

tasks=$1
model_dir=$2
result_path=$3

flagevalmm --tasks $tasks \
        --exec model_zoo/vlm/qwen_vl/model_adapter.py \
        --model $model_dir \
        --num-workers 8 \
        --output-dir /root/.cache/flagevalmm/results/$result_path \
        --backend vllm \
        --extra-args "--limit-mm-per-prompt image=8  --max-model-len 8192 --trust-remote-code --mm-processor-kwargs '{\"max_dynamic_patch\":2}'"
        #--extra-args "--limit-mm-per-prompt image=256 --max-model-len 32768 --gpu_memory_utilization 0.7 --trust-remote-code --mm-processor-kwargs '{\"max_dynamic_patch\":2}'"