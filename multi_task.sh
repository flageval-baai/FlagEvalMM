#!/bin/bash

echo "======================================================================================="
echo "                多任务评估脚本 - 多模型顺序评估，单模型内多GPU并行评估                "
echo "======================================================================================="
echo "此脚本将按顺序评估多个模型检查点。对于每个检查点，"
echo "将在多个GPU上并行评估该模型在不同任务上的性能。"
echo "每个任务会分配到一个独立的GPU上运行。"
echo ""

# 定义多个checkpoint路径列表
# 你可以在这里添加或修改你的检查点路径
checkpoints_list=(
"/share/project/caomingyu/vlm_ckpt/qwen2.5vl_7b_Robobrain_final_4056K/checkpoint-1000"
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
echo "📂 待评估的检查点列表:"
for ckpt_path in "${checkpoints_list[@]}"; do
    echo "  - $ckpt_path"
done
echo ""

# 定义多个任务
task_scripts=(
"tasks/refcoco_z/refcoco_task.py"
)
echo "📋 待评估任务数量 (每个检查点): ${#task_scripts[@]}"
if [ ${#task_scripts[@]} -gt 0 ]; then
    echo "任务列表详情:"
    for ((i=0; i<${#task_scripts[@]}; i++)); do
        echo "      [$((i+1))] ${task_scripts[$i]}"
    done
fi
echo ""

# 设定可用的GPU数量
available_gpus=8
echo "💻 可用GPU数量: $available_gpus"

# 日志文件保存目录
log_dir_base="eval_logs_multi_ckpt"
mkdir -p "$log_dir_base"
echo "📁 基础日志保存目录: $log_dir_base"
echo ""

# 外层循环，遍历所有检查点
for checkpoint in "${checkpoints_list[@]}"
do
    echo "======================================================================================="
    echo "🏁 开始评估检查点: $checkpoint"
    echo "======================================================================================="

    # 为当前检查点创建特定的日志子目录
    current_ckpt_name=$(basename "$checkpoint")
    current_log_dir="${log_dir_base}/${current_ckpt_name}"
    mkdir -p "$current_log_dir"
    echo "📁 当前检查点的日志保存目录: $current_log_dir"

    # 存储当前检查点任务信息的数组
    declare -a task_pids
    declare -a task_names
    declare -a task_logs

    echo "🔄 处理模型路径并生成结果名称后缀 (针对 $current_ckpt_name)..."
    # 处理checkpoint路径生成结果名称后缀
    parent_dir_name=$(basename "$(dirname "$checkpoint")")
    ckpt_dir_name=$(basename "$checkpoint")

    if [[ "$parent_dir_name" == "robobrain_stage1" ]]; then
        result_name_suffix="robobrain_stage1_${ckpt_dir_name}"
    elif [[ "$checkpoint" == *"qwen2.5vl"* ]]; then # 更通用的匹配qwen2.5vl
        # 尝试提取更有意义的名称，如果父目录就是qwen2.5vl，则使用更上一级目录名
        grandparent_dir_name=$(basename "$(dirname "$(dirname "$checkpoint")")")
        if [[ "$parent_dir_name" == "qwen2.5vl" && "$grandparent_dir_name" != "." && "$grandparent_dir_name" != "/" ]]; then
             result_name_suffix="qwen2.5vl_${grandparent_dir_name}_${ckpt_dir_name}"
        else
            result_name_suffix="qwen2.5vl_${parent_dir_name}_${ckpt_dir_name}"
        fi
    else
        result_name_suffix="${parent_dir_name}_${ckpt_dir_name}"
    fi

    # 如果检查点本身不是以 checkpoint- 开头（比如它是一个目录名），直接使用目录名
    if [[ ! "$ckpt_dir_name" == checkpoint-* ]]; then
        result_name_suffix="${ckpt_dir_name}"
    fi

    result_name_suffix=$(echo "$result_name_suffix" | tr '/' '_') # 替换路径中的斜杠
    echo "✅ 生成的结果名称后缀: $result_name_suffix"
    echo ""

    echo "🚀 开始为检查点 [$current_ckpt_name] 启动评估任务..."
    echo "---------------------------------------------------------------------------------------"
    # 遍历所有任务，每个分配到一个GPU上 (针对当前检查点)
    for i in "${!task_scripts[@]}"; do
        task_script="${task_scripts[$i]}"
        task_name=$(basename "$task_script" .py)
        assigned_gpu=$((i % available_gpus))
        log_file="${current_log_dir}/complete_log_${result_name_suffix}_${task_name}.txt"
        
        echo "  【任务 $((i+1))/${#task_scripts[@]}】对 [$current_ckpt_name]"
        echo "    🔹 任务脚本: $task_script"
        echo "    🔹 任务名称: $task_name"
        echo "    🔹 使用GPU: $assigned_gpu"
        echo "    🔹 日志文件: $log_file"
        
        CUDA_VISIBLE_DEVICES=$assigned_gpu bash evaluating.sh "$task_script" "$checkpoint" "$result_name_suffix" > "$log_file" 2>&1 &
        task_pid=$!
        echo "    🔹 进程ID: $task_pid"
        echo "    ✅ 任务已启动"
        echo "  -----------------------------------------------------------------------------------"
        
        task_pids+=($task_pid)
        task_names+=("${task_name}")
        task_logs+=("$log_file")
        
        # 每隔几秒启动一个任务 (如果任务很多，避免瞬间过载)
        # 根据实际情况调整 sleep 时间，如果GPU资源充足且任务启动快，可以缩短或移除
        if [ $(( (i+1) % available_gpus )) -eq 0 ]; then # 每分配完一轮GPU后稍作停顿
            echo "    💤 短暂休眠10秒，等待当前批次任务稳定启动..."
            sleep 10 
        elif [ ${#task_scripts[@]} -gt $available_gpus ]; then # 如果任务总数大于GPU数，则每个任务启动后都停一下
            sleep 5
        fi
    done

    echo ""
    echo "🎉 检查点 [$current_ckpt_name] 的所有评估任务已成功启动！共 ${#task_scripts[@]} 个任务"
    echo "您可以使用以下方法监控任务进度:"
    echo "  - 查看后台任务: jobs"
    echo "  - 查看日志目录: ls -lh $current_log_dir"
    echo "  - 查看特定任务日志: tail -f <日志文件>"
    echo ""
    echo "⏳ 正在等待检查点 [$current_ckpt_name] 的所有任务完成..."
    echo "---------------------------------------------------------------------------------------"

    success_tasks_current_ckpt=0
    failed_tasks_current_ckpt=0
    for i in "${!task_pids[@]}"; do
        echo -n "等待任务 ${task_names[$i]} (PID: ${task_pids[$i]}) for checkpoint [$current_ckpt_name] 完成... "
        wait ${task_pids[$i]}
        exit_status=$?
        
        if [ $exit_status -eq 0 ]; then
            success_tasks_current_ckpt=$((success_tasks_current_ckpt + 1))
            echo "✅ 成功"
        else
            failed_tasks_current_ckpt=$((failed_tasks_current_ckpt + 1))
            echo "❌ 失败 (退出状态: $exit_status)"
            echo "  ⚠️  警告: 任务 ${task_names[$i]} for checkpoint [$current_ckpt_name] 失败 (PID: ${task_pids[$i]}, 退出状态: $exit_status)"
            echo "      📝 日志文件: ${task_logs[$i]}"
            echo "      🔍 错误信息 (最后10行):"
            tail -10 "${task_logs[$i]}" | sed 's/^/        /'
            echo "  -----------------------------------------------------------------------------------"
        fi
    done

    echo ""
    echo "---------------------------------------------------------------------------------------"
    echo "📊 检查点 [$current_ckpt_name] 评估任务执行结果统计"
    echo "---------------------------------------------------------------------------------------"
    echo "  ✅ 成功任务数: $success_tasks_current_ckpt"
    echo "  ❌ 失败任务数: $failed_tasks_current_ckpt"
    echo "  🔢 总任务数: $((success_tasks_current_ckpt + failed_tasks_current_ckpt))"
    if [ $((success_tasks_current_ckpt + failed_tasks_current_ckpt)) -ne 0 ]; then
        success_percentage=$((success_tasks_current_ckpt * 100 / (success_tasks_current_ckpt + failed_tasks_current_ckpt)))
        echo "  📈 成功率: ${success_percentage}%"
    else
        echo "  📈 成功率: 0%"
    fi
    echo ""

    if [ $failed_tasks_current_ckpt -eq 0 ]; then
        echo "🎉 检查点 [$current_ckpt_name] 的所有任务已成功完成！"
        echo "结果文件保存在 (大致位置，具体看evaluating.sh的逻辑): /root/.cache/flagevalmm/results/${result_name_suffix}/"
    else
        echo "⚠️  警告: $failed_tasks_current_ckpt 个任务 for checkpoint [$current_ckpt_name] 失败。请检查上面的错误信息或相应的日志文件。"
        echo "以下是失败的任务列表 for [$current_ckpt_name]:"
        for i in "${!task_pids[@]}"; do
            # 再次检查确保只列出失败的
            if ! wait ${task_pids[$i]} 2>/dev/null; then
                 echo "    ❌ ${task_names[$i]} (PID: ${task_pids[$i]}) 日志: ${task_logs[$i]}"
            fi
        done
    fi
    echo "======================================================================================="
    echo "✅ 完成检查点: $checkpoint 的评估。"
    # 如果不是最后一个检查点，则在开始下一个之前等待一段时间，给系统一点喘息时间
    if [[ "$checkpoint" != "${checkpoints_list[-1]}" ]]; then
        echo "⏸️ 准备评估下一个检查点，暂停30秒..."
        sleep 30
    fi
done

echo "======================================================================================="
echo "所有检查点的评估流程已全部完成！"
echo "======================================================================================="