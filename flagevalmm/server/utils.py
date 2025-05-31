import requests
import argparse
import re
import random
import socket
from PIL import Image
import numpy as np
from mmengine.config import Config
from typing import Any, List, Optional, Tuple
import importlib.util
from flagevalmm.registry import DATASETS, EVALUATORS
import os.path as osp

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)


@retry(wait=wait_random_exponential(min=2, max=10), stop=stop_after_attempt(3))
def get_meta(task_name: str, server_ip: str, server_port: int, timeout: int = 1000):
    url = f"{server_ip}:{server_port}/meta_info?task={task_name}"
    meta_info = requests.get(url, timeout=timeout).json()
    return meta_info


@retry(wait=wait_random_exponential(min=2, max=10), stop=stop_after_attempt(3))
def get_task_info(server_ip: str, server_port: int, timeout: int = 1000):
    url = f"{server_ip}:{server_port}/task_info"
    task_info = requests.get(url, timeout=timeout).json()
    return task_info


@retry(wait=wait_random_exponential(min=2, max=10), stop=stop_after_attempt(3))
def submit(
    task_name: str,
    model_name: str,
    server_ip: str,
    server_port: int,
    timeout: int = 1000,
    output_dir: str = "",
) -> Any:
    url = f"{server_ip}:{server_port}/evaluate?task={task_name}&model_name={model_name}"
    if output_dir:
        url += f"&output_dir={output_dir}"
    response = requests.get(url, timeout=timeout)
    return response.json()


@retry(wait=wait_random_exponential(min=2, max=10), stop=stop_after_attempt(3))
def get_data(
    index: int, task_name: str, server_ip: str, server_port: int, timeout: int = 1000
) -> Any:
    url = f"{server_ip}:{server_port}/get_data?index={index}&task={task_name}"
    response = requests.get(url, timeout=timeout)
    if response.status_code != 200:
        raise Exception(
            f"Failed to get data from server: {response.status_code} reason: {response.text}"
        )
    return response.json()


def get_retrieval_data(
    index: int,
    task_name: str,
    data_type: str,
    server_ip: str,
    server_port: int,
    timeout: int = 1000,
):
    url = f"{server_ip}:{server_port}/get_retrieval_data?index={index}&type={data_type}&task={task_name}"
    response = requests.get(url, timeout=timeout).json()
    return response


def parse_args():
    parser = argparse.ArgumentParser(description="Infer a model")
    parser.add_argument("--tasks", nargs="+", help="tasks to run")
    parser.add_argument("--exec", type=str, help="model path in examples")
    parser.add_argument("--debug", action="store_true", help="debug mode")
    parser.add_argument(
        "--try-run",
        action="store_true",
        help="try run mode, only run the first 32 samples",
    )
    parser.add_argument("--output-dir", type=str, help="output dir")
    parser.add_argument("--data-root", type=str, help="data root")
    parser.add_argument("--model", type=str, help="model name or path")
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["http", "claude", "gemini", "gpt", "hunyuan"],
        help="type of the model",
    )
    parser.add_argument("--cfg", "-c", type=str, help="config file")
    parser.add_argument("--num-workers", "--num_workers", type=int)
    parser.add_argument("--backend", type=str)
    parser.add_argument(
        "--no-local-mode",
        action="store_false",
        dest="local_mode",
        help="disable local mode (use evaluation server)",
    )
    parser.add_argument("--disable-evaluation-server", "-ds", action="store_true")
    parser.add_argument("--skip", action="store_true", help="skip finished tasks")
    parser.add_argument(
        "--server-port",
        "--server_port",
        type=int,
        help="port of evaluation server",
        default=5000,
    )
    parser.add_argument(
        "--server-ip",
        "--server_ip",
        type=str,
        default="http://localhost",
        help="ip of evaluation server",
    )
    parser.add_argument("--timeout", type=int, default=1000)
    parser.add_argument("--quiet", "-q", action="store_true", help="quiet mode")
    parser.add_argument(
        "--without-infer", "-wi", action="store_true", help="without inference"
    )
    parser.add_argument("--url", type=str, help="url of api model")
    parser.add_argument("--api-key", type=str, help="api key of api model")
    parser.add_argument(
        "--use-cache", action="store_true", help="use cache of api model"
    )
    parser.add_argument(
        "--extra-args", type=str, help="extra args of local server model"
    )
    parser.add_argument(
        "--num-infers",
        type=int,
        default=1,
        help="number of inferences to perform for each question (when temperature >= 0)",
    )
    args = parser.parse_args()
    return args


def process_images_symbol(
    text: str, dst_pattern: Optional[str] = None
) -> Tuple[str, List[int]]:
    pattern = r"<image (\d+)>"

    matches = [int(num) - 1 for num in re.findall(pattern, text)]
    if dst_pattern is not None:
        text = re.sub(pattern, dst_pattern, text)
    return text, matches


def load_pil_image(
    img_paths: List[str],
    img_idx: List[int],
    reduplicate: bool = False,
    reqiures_img: bool = False,
) -> Tuple[List[Image.Image], List[int]]:
    image_list = []
    for img_path in img_paths:
        img = Image.open(img_path).convert("RGB")
        image_list.append(img)
    if reduplicate:
        img_idx = list(set(img_idx))
    image_list_processed = []
    for i in img_idx:
        if i < len(image_list):
            image_list_processed.append(image_list[i])
        else:
            print("[warning] image index out of range")
            image_list_processed.append(image_list[-1])
    if reqiures_img and len(image_list_processed) == 0:
        # Create a dummy image
        dummy_image = np.ones((256, 256, 3), dtype=np.uint8) * 128
        dummy_image_pil = Image.fromarray(dummy_image)
        image_list_processed.append(dummy_image_pil)
    return image_list_processed, img_idx


def default_collate_fn(batch: List[Tuple[Any, Any, Any]]) -> Tuple[Any, Any, Any]:
    question_ids = [item[0] for item in batch]
    questions = [item[1] for item in batch]
    images_list = [item[2] for item in batch]

    return question_ids, questions, images_list


def is_port_occupied(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(("localhost", port)) == 0


def get_random_port() -> int:
    while True:
        port = random.randint(3000, 30000)
        if not is_port_occupied(port):
            return port


def merge_args(cfg: Config, task_config_file: str, args: argparse.Namespace) -> Config:
    if args.debug:
        cfg.server.debug = True
    if args.data_root:
        cfg.dataset.data_root = args.data_root
    if args.try_run:
        cfg.dataset.debug = True
    base_dir = osp.abspath(osp.dirname(task_config_file))
    cfg.dataset.base_dir = base_dir
    if cfg.get("evaluator", None):
        cfg.evaluator.base_dir = base_dir

    return cfg


def maybe_register_class(cfg: Config, task_config_file: str) -> None:
    """Register custom dataset and evaluator classes from config file.
    Args:
        cfg: Config object containing registration info
        task_config_file: Path to task config file
    """

    def _import_module(base_dir: str, file_name: str, module_name: str) -> None:
        """Helper function to import a module from file.
        Args:
            base_dir: Base directory containing the module file
            file_name: Name of the file to import
            module_name: Name to give the imported module
        """
        file_path = osp.join(base_dir, file_name)
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Failed to load module from {file_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

    base_dir = osp.abspath(osp.dirname(task_config_file))

    # Register custom dataset classes
    if "register_dataset" in cfg:
        for file_name, class_name in cfg.register_dataset.items():
            if class_name not in DATASETS.module_dict:
                _import_module(base_dir, file_name, class_name)

    # Register custom evaluator classes
    if "register_evaluator" in cfg:
        for file_name, class_name in cfg.register_evaluator.items():
            if class_name not in EVALUATORS.module_dict:
                _import_module(base_dir, file_name, class_name)
