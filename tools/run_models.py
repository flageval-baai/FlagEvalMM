import subprocess
import time
from typing import List, Dict, Set, Optional
import os
import argparse
from mmengine import Config
import json


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Config file with model information and tasks",
    )
    parser.add_argument(
        "--cfg-dir",
        type=str,
        default="model_configs/open",
        help="Directory containing model configs",
    )
    parser.add_argument(
        "--output-dir", type=str, help="Optional output directory override"
    )
    parser.add_argument(
        "--models-base-dir",
        type=str,
        required=True,
        help="Base directory for model files",
    )
    return parser.parse_args()


# Define GPU requirements for each model
GPU_REQUIREMENTS = {
    "Qwen2-VL-72B-Instruct": 4,
    "Qwen2.5-VL-72B-Instruct": 4,
    "InternVL2-Llama3-76B": 4,
    "InternVL2-26B": 4,
    "InternVL2_5-26B": 4,
    "InternVL2_5-78B": 8,
    "llava-onevision-qwen2-72b-ov-chat-hf": 4,
    "Molmo-72B-0924": 4,
    "NVLM-D-72B": 4,
    "Meta-Llama-3.2-11B-Vision-Instruct": 2,
    "Meta-Llama-3.2-90B-Vision-Instruct": 8,
    "Pixtral-Large-Instruct-2411": 8,
    "deepseek-vl2": 4,
}


class GPUManager:
    def __init__(self, total_gpus: int = 8):
        self.total_gpus = total_gpus
        self.available_gpus = set(range(total_gpus))
        self.running_processes: Dict[subprocess.Popen, Set[int]] = {}

    def get_gpu_requirement(self, model_name: str) -> int:
        return GPU_REQUIREMENTS.get(model_name, 1)

    def can_allocate_gpus(self, required_gpus: int) -> bool:
        return len(self.available_gpus) >= required_gpus

    def allocate_gpus(self, required_gpus: int) -> Set[int]:
        if not self.can_allocate_gpus(required_gpus):
            return set()
        allocated = set(list(self.available_gpus)[:required_gpus])
        self.available_gpus -= allocated
        return allocated

    def release_gpus(self, gpus: Set[int]):
        self.available_gpus.update(gpus)

    def run_command(self, cmd: str, model_name: str):
        required_gpus = self.get_gpu_requirement(model_name)
        allocated_gpus = self.allocate_gpus(required_gpus)
        if not allocated_gpus:
            return False

        gpu_list = ",".join(map(str, allocated_gpus))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_list

        # Create logs directory if it doesn't exist
        os.makedirs("logs", exist_ok=True)

        # Create log file name based on model name
        log_file = f"logs/{model_name}.log"
        with open(log_file, "w") as f:
            process = subprocess.Popen(
                cmd, shell=True, env=env, stdout=f, stderr=subprocess.STDOUT
            )

        self.running_processes[process] = allocated_gpus
        print(f"Started {model_name} on GPUs: {gpu_list} (logging to {log_file})")
        return True

    def check_processes(self):
        completed = []
        for process in self.running_processes:
            if process.poll() is not None:  # Process has finished
                gpus = self.running_processes[process]
                self.release_gpus(gpus)
                completed.append(process)

        for process in completed:
            del self.running_processes[process]


def run_models(model_info: List, cmds: List[str]):
    gpu_manager = GPUManager(total_gpus=8)
    cmd_index = 0

    try:
        while cmd_index < len(cmds) or gpu_manager.running_processes:
            # Check and clean up finished processes
            gpu_manager.check_processes()

            # Try to launch new commands if available
            while cmd_index < len(cmds):
                model_name = model_info[cmd_index][0]
                required_gpus = gpu_manager.get_gpu_requirement(model_name)

                if gpu_manager.can_allocate_gpus(required_gpus):
                    if gpu_manager.run_command(cmds[cmd_index], model_name):
                        cmd_index += 1
                else:
                    break  # Not enough GPUs available, wait for running processes to finish

            time.sleep(10)  # Wait before checking again
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C. Terminating all running processes...")
        for process in list(gpu_manager.running_processes.keys()):
            print(f"Killing process {process.pid}...")
            process.kill()
            process.wait()  # Wait for the process to actually terminate
            gpus = gpu_manager.running_processes[process]
            gpu_manager.release_gpus(gpus)
        print("All processes terminated.")
        raise  # Re-raise the KeyboardInterrupt


def update_model_config(
    model_config_path: str, model_name: str, models_base_dir: str
) -> Optional[Dict]:
    """Update model config with standardized paths"""
    if not os.path.exists(model_config_path):
        config_data = {}
    else:
        with open(model_config_path, "r") as f:
            config_data = json.load(f)

    try:
        # Update model_name path with standardized path
        new_model_path = os.path.join(models_base_dir, model_name)
        config_data["model_name"] = new_model_path

        # Write the updated config back to the file
        with open(model_config_path, "w") as f:
            json.dump(config_data, f, indent=4)

        return config_data
    except Exception as e:
        print(f"Error updating config file {model_config_path}: {e}")
        return None


if __name__ == "__main__":
    args = parse_args()
    config = Config.fromfile(args.config)

    # Use provided output directory or one from config
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = config.get("output_dir", None)

    # Create models_base_dir if it doesn't exist
    os.makedirs(args.models_base_dir, exist_ok=True)

    if args.models_base_dir:
        # Update model configs with specified paths
        for model_info_tuple in config.model_info:
            model_name = model_info_tuple[0]
            model_config_path = f"{args.cfg_dir}/{model_name}.json"
            update_model_config(model_config_path, model_name, args.models_base_dir)

    cmds = []
    for model_name, backend in config.model_info:
        cmd = f"flagevalmm --tasks {' '.join(config.tasks)} --cfg {args.cfg_dir}/{model_name}.json --quiet --skip"

        if output_dir:
            cmd += f" --output-dir {output_dir}/{model_name}"

        # Handle different backend types
        if backend == "api_model":
            cmd += " --exec model_zoo/vlm/api_model/model_adapter.py --backend vllm"
        else:
            if backend.endswith(".py"):
                cmd += f" --exec model_zoo/vlm/{backend}"
            else:
                cmd += f" --exec model_zoo/vlm/{backend}/model_adapter.py"

        cmds.append(cmd)
        print(cmd)

    # Run the models
    run_models(config.model_info, cmds)
