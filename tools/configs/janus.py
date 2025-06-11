# This is an example configuration file for batch model evaluation
# Format: [model_name, backend]
# backend can be:
# - "api_model" for models using the API adapter with vllm backend
# - "<backend_folder>/model_adapter.py" for models with a specific adapter in model_zoo/vlm/
# - "<backend_folder>/<custom_adapter>.py" for models with custom adapter files

model_info = [
    ["Janus-Pro-7B", "model_zoo/uni/janus/model_adapter.py"],
]

# List of tasks to evaluate
tasks = ["tasks/mmmu/mmmu_val.py", "tasks/t2i/coco/coco_1k_test.py"]

# Optional: Default output directory for results
output_dir = "./results/janus"
