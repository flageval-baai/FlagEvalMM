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

from dataclasses import dataclass, field
from typing import Dict, Union

from omegaconf import DictConfig, OmegaConf


@dataclass
class ServerCfg:
    ip: str = "http://localhost"
    port: int = 5000
    timeout: int = 1000
    local_mode: bool = True
    quiet: bool = False
    disable_evaluation_server: bool = False


@dataclass
class TasksCfg:
    files: list[str] = field(default_factory=list)
    data_root: Optional[str] = None
    debug: bool = False
    try_run: bool = False
    output_dir: Optional[str] = None
    num_workers: Optional[int] = None
    skip: bool = True


@dataclass
class InferCfg:
    use_cache: bool = False
    num_infers: int = 1
    temperature: float = 0.0
    stream: bool = False
    max_tokens: Optional[int] = None
    reasoning: Optional[Dict[str, Any]] = None
    provider: Optional[Dict[str, Any]] = None
    retry_time: Optional[int] = None
    system_prompt: Optional[str] = None
    thinking: Optional[bool] = False


@dataclass
class RunCfg:
    """
    Structured runtime config (YAML-friendly).

    Notes:
    - We intentionally keep `model`/`infer` as free-form dicts to avoid breaking
      existing model configs that may include arbitrary keys.
    - Use `load_run_cfg()` to load user-provided config without injecting defaults.
    """

    server: ServerCfg = field(default_factory=ServerCfg)
    tasks: TasksCfg = field(default_factory=TasksCfg)
    model: Dict[str, Any] = field(default_factory=dict)
    infer: InferCfg = field(default_factory=InferCfg)
    # Free-form extra config for adapters.
    extra_config: Dict[str, Any] = field(default_factory=dict)


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
    parser = argparse.ArgumentParser(description="Infer a model with runtime config")
    parser.add_argument(
        "--cfg",
        "-c",
        type=str,
        required=True,
        help="runtime config file (yaml/json)",
    )
    # Optional: override adapter entrypoint without editing YAML.
    parser.add_argument(
        "--exec",
        type=str,
        default=None,
        help="model adapter entrypoint; overrides cfg.model.exec if provided",
    )
    parser.add_argument(
        "--without-infer",
        "-wi",
        dest="without_infer",
        action="store_const",
        const=True,
        default=False,
        help="only evaluate (skip model inference)",
    )
    return parser.parse_args()


def load_run_cfg(path_or_str_or_dict: Optional[str | DictConfig]) -> Any:
    """
    Load config from:
    - None -> {}
    - DictConfig -> resolved dict
    - str:
      - existing file: yaml/yml/json
      - otherwise: json string
    """
    if path_or_str_or_dict is None or path_or_str_or_dict == {}:
        return {}
    if isinstance(path_or_str_or_dict, DictConfig):
        return OmegaConf.to_container(path_or_str_or_dict, resolve=True)  # type: ignore[return-value]

    s = str(path_or_str_or_dict)
    if osp.exists(s):
        ext = osp.splitext(s)[1].lower()
        if ext in {".yaml", ".yml"}:
            cfg = OmegaConf.load(s)
            return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
        else:
            raise ValueError(f"Invalid config file: {path_or_str_or_dict}")
    else:
        raise ValueError(f"Config file not found: {path_or_str_or_dict}")


def load_run_cfg_with_defaults(
    path_or_str_or_dict: Optional[Union[str, Dict[str, Any], DictConfig]]
) -> Any:
    """
    Load runtime config, normalize to nested schema, and inject structured defaults.

    Returns a plain dict containing all defaults from RunCfg unless overridden by user config.
    """
    user_cfg = load_run_cfg(path_or_str_or_dict)
    defaults = OmegaConf.structured(RunCfg())
    merged = OmegaConf.merge(defaults, OmegaConf.create(user_cfg or {}))
    return OmegaConf.to_container(merged, resolve=True)  # type: ignore[return-value]


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
    if getattr(args, "debug", None) is True:
        cfg.server.debug = True
    if getattr(args, "data_root", None):
        cfg.dataset.data_root = args.data_root
    if getattr(args, "try_run", None) is True:
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
