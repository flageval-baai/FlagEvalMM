#!/bin/bash

# 定义checkpoint路径和对应的名称
checkpoints=(
# "/share/project/caomingyu/vlm_ckpt/qwen2.5vl_7b_Robobrain_final_4056K/checkpoint-1000"
"/share/project/caomingyu/vlm_ckpt/qwen2.5vl_7b_Robobrain_final_4056K/checkpoint-2000"
"/share/project/caomingyu/vlm_ckpt/qwen2.5vl_7b_Robobrain_final_4056K/checkpoint-3000"
"/share/project/caomingyu/vlm_ckpt/qwen2.5vl_7b_Robobrain_final_4056K/checkpoint-4000"
"/share/project/caomingyu/vlm_ckpt/qwen2.5vl_7b_Robobrain_final_4056K/checkpoint-5000"
"/share/project/caomingyu/vlm_ckpt/qwen2.5vl_7b_Robobrain_final_4056K/checkpoint-6000"
"/share/project/caomingyu/vlm_ckpt/qwen2.5vl_7b_Robobrain_final_4056K/checkpoint-7000"
"/share/project/caomingyu/vlm_ckpt/qwen2.5vl_7b_Robobrain_final_4056K/checkpoint-8000"
"/share/project/caomingyu/vlm_ckpt/qwen2.5vl_7b_Robobrain_final_4056K/checkpoint-9000"
# "/share/project/caomingyu/vlm_ckpt/qwen2.5vl_7b_Robobrain_final_1757K_stage2/checkpoint-1000"
# "/share/project/caomingyu/vlm_ckpt/qwen2.5vl_7b_Robobrain_final_1757K_stage2/checkpoint-2000"
# "/share/project/caomingyu/vlm_ckpt/qwen2.5vl_7b_Robobrain_final_1757K_stage2/checkpoint-3000"
# "/share/project/caomingyu/vlm_ckpt/qwen2.5vl_7b_Robobrain_final_1757K_stage2/checkpoint-4000"
)

# 基础任务定义
task_script=("tasks/refcoco_z/refcoco_task.py")
#tasks/refcoco_z/refcoco_task.py
#tasks/erqa/erqa.py
# 提取任务名称（不包含路径和扩展名）
task_name=$(basename "$task_script" .py)

# 用于分配GPU的计数器
gpu_id_counter=0

# 日志文件保存目录
log_dir="eval_logs"
mkdir -p "$log_dir"

# 存储任务信息的数组
declare -a task_pids
declare -a task_names
declare -a task_logs

# 遍历所有checkpoint
for ckpt_path in "${checkpoints[@]}"; do
    parent_dir_name=$(basename "$(dirname "$ckpt_path")")
    ckpt_dir_name=$(basename "$ckpt_path")
    
    # 路径处理逻辑（不变）
    if [[ "$parent_dir_name" == "robobrain_stage1" ]]; then
        result_name_suffix="robobrain_stage1_${ckpt_dir_name}"
    elif [[ "$ckpt_path" == *"qwen2.5vl"* ]]; then
        if [[ "$parent_dir_name" == "qwen2.5vl" ]]; then
            sub_path=$(basename "$(dirname "$ckpt_path")")
            model_name=$(basename "$ckpt_path")
            result_name_suffix="qwen2.5vl_${sub_path}_${model_name}"
        else
            result_name_suffix="${parent_dir_name}_${ckpt_dir_name}"
        fi
    else
        result_name_suffix="${parent_dir_name}_${ckpt_dir_name}"
    fi
    
    if [[ ! "$ckpt_dir_name" == checkpoint-* ]]; then
        result_name_suffix="${ckpt_dir_name}"
    fi

    result_name_suffix=$(echo "$result_name_suffix" | tr '/' '_')
    
    # 修改日志文件名，末尾添加任务名
    log_file="${log_dir}/complete_log_${result_name_suffix}_${task_name}.txt"
    
    # 修改GPU分配逻辑，只使用GPU 1-7
    assigned_gpu=$(( (gpu_id_counter % 8)))
    
    echo "正在启动评估任务:"
    echo "  Checkpoint: $ckpt_path"
    echo "  结果名称后缀: $result_name_suffix" 
    echo "  指定GPU: $assigned_gpu"
    echo "  日志文件: $log_file"
    echo "----------------------------------------------------"
    
    # 在后台执行评估命令，并捕获PID
    CUDA_VISIBLE_DEVICES=$assigned_gpu bash evaluating.sh "$task_script" "$ckpt_path" "$result_name_suffix" > "$log_file" 2>&1 &
    task_pid=$!
    
    # 存储任务信息
    task_pids+=($task_pid)
    task_names+=("$result_name_suffix")
    task_logs+=("$log_file")
    
    # 更新GPU计数器
    gpu_id_counter=$((gpu_id_counter + 1))
    
    # 每隔几秒启动一个任务
    sleep 5 
done

echo "所有评估任务已启动。"
echo "您可以使用 'jobs' 命令查看后台任务状态，或直接查看 ${log_dir}/ 中的日志文件。"
echo "等待所有后台任务完成..."

# 检查每个任务的完成状态
failed_tasks=0
for i in "${!task_pids[@]}"; do
    wait ${task_pids[$i]}
    exit_status=$?
    
    if [ $exit_status -ne 0 ]; then
        failed_tasks=$((failed_tasks + 1))
        echo "警告: 任务 ${task_names[$i]} 失败 (PID: ${task_pids[$i]}, 退出状态: $exit_status)"
        echo "       日志文件: ${task_logs[$i]}"
        # 输出日志文件的最后几行，帮助诊断问题
        echo "       错误信息 (最后10行):"
        tail -10 "${task_logs[$i]}" | sed 's/^/       /'
        echo "----------------------------------------------------"
    fi
done

if [ $failed_tasks -eq 0 ]; then
    echo "所有任务已成功完成！"
else
    echo "警告: $failed_tasks 个任务失败。请检查上面的错误信息或相应的日志文件。"
fi