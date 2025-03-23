# This is an example configuration file for batch model evaluation
# Format: [model_name, backend]
# backend can be:
# - "api_model" for models using the API adapter with vllm backend
# - "<backend_folder>/model_adapter.py" for models with a specific adapter in model_zoo/vlm/
# - "<backend_folder>/<custom_adapter>.py" for models with custom adapter files

model_info = [
    # Small models
    ["Qwen2-VL-7B-Instruct", "api_model"],
    ["InternVL2-8B", "api_model"],
    ["llava-onevision-qwen2-7b-ov-chat-hf", "api_model"],
    ["Phi-3.5-vision-instruct", "Phi_3.5_v"],
    
    # Large models (requiring multiple GPUs)
    ["Qwen2-VL-72B-Instruct", "api_model"],
    ["InternVL2-Llama3-76B", "api_model"],
    ["Meta-Llama-3.2-11B-Vision-Instruct", "api_model"],
    
    # Custom adapter example
    ["InternVL2_5-26B", "intern_vl/model_adapter_v2_5.py"],
]

# List of tasks to evaluate
tasks = [
    "tasks/mmmu/mmmu_val.py",
    "tasks/mmvet/mmvet_v2.py",
    "tasks/cmmmu/cmmmu_val.py",
    "tasks/blink/blink_val.py",
]

# Optional: Default output directory for results
output_dir = "./results/batch_eval" 