  # tasks/mmmu/mmmu_val.py \
# tasks/ocrbench/ocrbench_test.py \
# tasks/text_vqa/text_vqa_val.py
# tasks/vg/vg_task.py
# tasks/vsi_bench/vsi_bench_tiny/test.py
# tasks/refcoco/refcoco_val.py


# model_zoo/vlm/qwen_vl/model_adapter.py
# tasks/robovqa/robovqa_test.py
# tasks/openeqa/openeqa_test.py


model_dir=$1
result_path=$2

flagevalmm --tasks   tasks/openeqa/openeqa_test.py  \
        --exec model_zoo/vlm/qwen_vl/model_adapter.py \
        --model $model_dir \
        --num-workers 8 \
        --output-dir /root/.cache/flagevalmm/results/$result_path \
        --backend vllm  \
        --extra-args "--limit-mm-per-prompt image=18 --tensor-parallel-size 8 --max-model-len 8192 --trust-remote-code --mm-processor-kwargs '{\"max_dynamic_patch\":8}'"
