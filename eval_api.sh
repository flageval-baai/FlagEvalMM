#!/bin/bash

# 在这里直接定义任务列表
TASK_FILES=(
#   "tasks/mmmu/mmmu_val.py" # 示例任务，你可以根据需要修改或添加更多任务
  "tasks/blink/blink_val.py"
#   "tasks/embspatial_bench/embspatial_bench.py"
#   "tasks/cv_bench/cv_bench_test.py"
#   "tasks/erqa/erqa.py"
#   "tasks/sat/sat.py"
#   "tasks/vsi_bench/vsi_bench_tiny.py"
)

# 从这里开始，脚本的其余部分与之前类似，但直接使用上面定义的 TASK_FILES 数组

MODEL_EXEC="model_zoo/vlm/api_model/model_adapter.py"
MODEL_NAME="gemini-2.5-pro-preview-05-06"  # 更新模型名称
NUM_WORKERS=4
API_URL="http://104.243.40.194:3000/v1" # 更新API URL 为 OpenAI 的基础 URL
# API_KEY="sk-BQYAnntifGZ9gOm62b97Df65Da474d37903eAc6f010eC36e" # 旧的API Key，注释掉
# 重要：请确保OPENAI_API_KEY环境变量已正确设置，例如：export OPENAI_API_KEY="your_actual_api_key"
# 如果未设置，脚本将无法使用正确的API密钥。
API_KEY="sk-dlmwezf4dYHM9J3HZvV5WyZbFF1b84ySgJySUp0YN2qSK1zL" # 直接使用这里设置的API Key
BASE_OUTPUT_DIR="./results" # 基础输出目录
USE_CACHE_FLAG="--use-cache" # 是否使用缓存

# 检查 TASK_FILES 数组是否为空
if [ ${#TASK_FILES[@]} -eq 0 ]; then
  echo "错误：任务列表为空。请在脚本中定义至少一个任务，或取消注释示例任务。"
  exit 1
fi

# 不再检查 OPENAI_API_KEY 环境变量，因为API_KEY已在脚本中直接设置
# if [ -z "$OPENAI_API_KEY" ]; then
#   echo "错误：环境变量 OPENAI_API_KEY 未设置。"
#   echo "请通过 export OPENAI_API_KEY=\"your_actual_api_key\" 设置您的OpenAI API密钥。"
#   exit 1
# fi

echo "将要执行以下任务:"
for TASK_FILE in "${TASK_FILES[@]}"
do
  echo "  - $TASK_FILE"
done
echo ""

# Loop through each task and run the evaluation
for TASK_FILE in "${TASK_FILES[@]}"
do
  # Extract the task name from the task file path for a unique output directory
  # 也将模型名称包含在输出目录中，以区分不同模型的结果
  TASK_NAME_ONLY=$(basename "$TASK_FILE" .py)
  OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL_NAME}_${TASK_NAME_ONLY}"

  echo "Running evaluation for task: $TASK_FILE with model: $MODEL_NAME"
  echo "Output will be saved to: $OUTPUT_DIR"

  flagevalmm --tasks "$TASK_FILE" \
    --exec "$MODEL_EXEC" \
    --model "$MODEL_NAME" \
    --num-workers "$NUM_WORKERS" \
    --url "$API_URL" \
    --api-key "$API_KEY" \
    --output-dir "$OUTPUT_DIR" \
    $USE_CACHE_FLAG

  echo "Finished evaluation for task: $TASK_FILE"
  echo "----------------------------------------"
done

echo "All tasks completed."