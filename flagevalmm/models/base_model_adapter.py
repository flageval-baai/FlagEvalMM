import json
import copy
from typing import List, Dict, Any, Callable, Optional
import os.path as osp
from accelerate import Accelerator
from torch.utils.data import DataLoader

import os

from flagevalmm.server.utils import get_meta, get_task_info, submit, get_data
from flagevalmm.common.logger import get_logger
from flagevalmm.server.evaluation_server import EvaluationServer
from mmengine.config import Config
from flagevalmm.server.utils import maybe_register_class

os.environ["no_proxy"] = "127.0.0.1,localhost"
logger = get_logger(__name__)


def merge_args(
    cfg: Config,
    task_config_file: str,
    data_root: Optional[str] = None,
    debug: bool = False,
) -> Config:
    if data_root:
        cfg.dataset.data_root = data_root
    if debug:
        cfg.dataset.debug = True
    base_dir = osp.abspath(osp.dirname(task_config_file))
    cfg.dataset.base_dir = base_dir
    if cfg.get("evaluator", None):
        cfg.evaluator.base_dir = base_dir
    return cfg


def load_tasks(
    task_config_files: List[str], data_root: Optional[str] = None, debug: bool = False
) -> Dict[str, Config]:
    config_dict = {}
    for task_config_file in task_config_files:
        cfg = Config.fromfile(task_config_file, lazy_import=False)
        task_name = cfg.dataset.name
        cfg = merge_args(cfg, task_config_file, data_root, debug)
        maybe_register_class(cfg, task_config_file)
        config_dict[task_name] = cfg
    logger.info(f"Loaded {len(config_dict)} tasks: {config_dict.keys()}")
    return config_dict


class TaskManager:
    def __init__(
        self,
        server_ip: str,
        server_port: int,
        timeout: int = 1000,
        local_mode: bool = False,
        task_names: List[str] = None,
        **kwargs,
    ):
        self.server_ip = server_ip
        self.server_port = server_port
        self.timeout = timeout
        self.local_mode = local_mode
        if local_mode:
            assert task_names is not None, "task_names must be provided in local mode"
        self.task_names = task_names
        if local_mode:
            config_dict = load_tasks(
                task_names,
                debug=kwargs.get("debug", False),
                data_root=kwargs.get("data_root", None),
            )

            self.evaluation_server = EvaluationServer(
                config_dict,
                output_dir=kwargs.get("output_dir", None),
                model_path=kwargs.get("model_path", None),
                debug=kwargs.get("debug", False),
                quiet=kwargs.get("quiet", False),
                local_mode=True,
            )

    def get_task_info(self):
        if not self.local_mode:
            return get_task_info(self.server_ip, self.server_port)

        task_info = {
            "task_names": list(self.evaluation_server.config_dict.keys()),
            "model_path": self.evaluation_server.model_path,
        }
        return task_info

    def get_meta_info(self, task_name: str):
        if not self.local_mode:
            return get_meta(task_name, self.server_ip, self.server_port)
        return self.evaluation_server.get_task_meta_info(task_name)

    def submit(self, task_name: str, model_name: str, output_dir: str):
        if not self.local_mode:
            submit(
                task_name,
                model_name,
                self.server_ip,
                self.server_port,
                self.timeout,
                output_dir,
            )
        else:
            self.evaluation_server.evaluate_task(task_name, model_name, output_dir)

    def get_data(self, task_name: str, index: int):
        if not self.local_mode:
            return get_data(index, task_name, self.server_ip, self.server_port)
        return self.evaluation_server.get_task_data_by_index(task_name, index)


class BaseModelAdapter:
    def __init__(
        self,
        server_ip: str,
        server_port: int,
        timeout: int = 1000,
        enable_accelerate: bool = True,
        extra_cfg: str | Dict | None = None,
        local_mode: bool = False,
        task_names: List[str] = None,
        **kwargs,
    ) -> None:
        self.server_ip: str = server_ip
        self.server_port: int = server_port
        self.timeout: int = timeout
        self.task_manager = TaskManager(
            server_ip,
            server_port,
            local_mode=local_mode,
            task_names=task_names,
            **kwargs,
        )
        task_info = self.task_manager.get_task_info()
        self.tasks = task_info["task_names"]

        if isinstance(extra_cfg, str):
            if osp.exists(extra_cfg):
                try:
                    with open(extra_cfg, "r") as f:
                        extra_cfg = json.load(f)
                except Exception as e:
                    logger.info(f"Error loading extra config file: {e}")
            else:
                try:
                    extra_cfg = json.loads(extra_cfg)
                except Exception as e:
                    logger.info(f"Error loading extra config: {e}")

        if extra_cfg is not None:
            task_info.update(extra_cfg)
        self.task_info = task_info
        self.model_name: str = task_info.get("model_name", None)
        if self.model_name is None and "model_path" in task_info:
            self.model_name = osp.basename(task_info["model_path"])
        if not task_info.get("model_path"):
            task_info["model_path"] = self.model_name
        if enable_accelerate:
            self.accelerator = Accelerator()
        else:
            self.accelerator = None
        self.model_init(task_info)

    def model_init(self, task_info: Dict) -> None:
        raise NotImplementedError

    def run(self) -> None:
        for task_name in self.tasks:
            meta_info: Dict[str, Any] = self.task_manager.get_meta_info(task_name)
            if "output_dir" in self.task_info:
                meta_info["output_dir"] = osp.join(
                    self.task_info["output_dir"], task_name
                )
                os.makedirs(meta_info["output_dir"], exist_ok=True)
            # Save task_info
            task_info = copy.deepcopy(self.task_info)
            task_info.pop("task_names")
            task_info.update(meta_info)
            with open(osp.join(meta_info["output_dir"], "task_info.json"), "w") as f:
                json.dump(task_info, f, indent=2, ensure_ascii=True)
            self.run_one_task(task_name, meta_info)
            self.task_manager.submit(
                task_name,
                self.model_name,
                meta_info["output_dir"],
            )

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]) -> None:
        raise NotImplementedError

    def save_result(
        self,
        result: List[Dict[str, Any]],
        meta_info: Dict[str, Any],
        rank: int | None = None,
    ) -> None:
        if rank is None:
            output_file = osp.join(meta_info["output_dir"], f"{meta_info['name']}.json")
        else:
            output_file = osp.join(
                meta_info["output_dir"], f"{meta_info['name']}_rank{rank}.json"
            )
        try:
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.info(f"Error saving result: {e}")
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2, ensure_ascii=True)

    def collect_results_and_save(
        self,
        meta_info: Dict[str, Any],
    ) -> int:
        results_collect = []
        id_set = set()
        for i in range(self.accelerator.state.num_processes):
            with open(
                os.path.join(
                    meta_info["output_dir"], f"{meta_info['name']}_rank{i}.json"
                ),
                "r",
            ) as fin:
                for ans in json.load(fin):
                    if ans["question_id"] not in id_set:
                        id_set.add(ans["question_id"])
                        results_collect.append(ans)

        self.save_result(results_collect, meta_info)
        return len(results_collect)

    def create_data_loader(
        self,
        dataset_cls,
        task_name: str,
        task_type: str = "vqa",
        collate_fn: Optional[Callable] = None,
        batch_size: int = 1,
        num_workers: int = 2,
    ):
        if self.accelerator is not None:
            with self.accelerator.main_process_first():
                dataset = dataset_cls(
                    task_name,
                    self.task_manager,
                    task_type=task_type,
                )
                data_loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    collate_fn=collate_fn,
                    shuffle=False,
                )
            data_loader = self.accelerator.prepare(data_loader)
        else:
            dataset = dataset_cls(
                task_name, self.server_ip, self.server_port, self.timeout
            )
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                num_workers=num_workers,
                collate_fn=collate_fn,
                shuffle=False,
            )
        return data_loader
