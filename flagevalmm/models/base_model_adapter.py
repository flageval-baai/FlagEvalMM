import json
import copy
from typing import List, Dict, Any, Callable, Optional, Union
import os.path as osp
from accelerate import Accelerator
from torch.utils.data import DataLoader

import os

from flagevalmm.server.utils import get_meta, get_task_info, submit, get_data
from flagevalmm.common.logger import get_logger
from flagevalmm.server.evaluation_server import EvaluationServer
from mmengine.config import Config
from flagevalmm.server.utils import maybe_register_class, load_run_cfg
from flagevalmm.models.api_response import ProcessResult

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
        model_path: str = None,
        task_config: Dict = None,
        **kwargs,
    ):
        self.server_ip = server_ip
        self.server_port = server_port
        self.timeout = timeout
        self.local_mode = local_mode
        if local_mode:
            assert task_names is not None, "task_names must be provided in local mode"
        self.task_names = task_names
        debug = task_config.get("debug", False) or task_config.get("try_run", False)

        if local_mode:
            config_dict = load_tasks(
                task_names,
                debug=debug,
                data_root=task_config.get("data_root", None),
            )

            self.evaluation_server = EvaluationServer(
                config_dict,
                output_dir=task_config.get("output_dir", None),
                model_path=model_path,
                debug=debug,
                quiet=task_config.get("quiet", False),
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

        extra_cfg_nested = load_run_cfg(extra_cfg)

        server_cfg = (
            extra_cfg_nested.get("server", {})
            if isinstance(extra_cfg_nested.get("server", {}), dict)
            else {}
        )
        tasks_cfg = (
            extra_cfg_nested.get("tasks", {})
            if isinstance(extra_cfg_nested.get("tasks", {}), dict)
            else {}
        )
        model_cfg = (
            extra_cfg_nested.get("model", {})
            if isinstance(extra_cfg_nested.get("model", {}), dict)
            else {}
        )
        infer_cfg = (
            extra_cfg_nested.get("infer", {})
            if isinstance(extra_cfg_nested.get("infer", {}), dict)
            else {}
        )
        extra_config = (
            extra_cfg_nested.get("extra_config", {})
            if isinstance(extra_cfg_nested.get("extra_config", {}), dict)
            else {}
        )

        server_ip = server_cfg.get("ip", server_ip)
        server_port = server_cfg.get("port", server_port)
        timeout = server_cfg.get("timeout", timeout)
        local_mode = server_cfg.get("local_mode", local_mode)
        task_names = tasks_cfg.get("files", task_names)
        model_path = model_cfg.get("model_path") or model_cfg.get("model_name")

        self.server_ip = server_ip
        self.server_port = server_port
        self.timeout = timeout

        self.task_manager = TaskManager(
            server_ip,
            server_port,
            local_mode=local_mode,
            task_names=task_names,
            model_path=model_path,
            task_config=tasks_cfg,
            **kwargs,
        )
        task_info = self.task_manager.get_task_info()
        self.tasks = task_info["task_names"]
        task_info = self.build_task_info(task_info, model_cfg, infer_cfg, tasks_cfg)

        # We keep both:
        # - `task_info["extra_config"]` for traceability
        # - top-level keys (non-overriding) for adapter convenience
        if local_mode and extra_config:
            task_info["extra_config"] = extra_config
            conflicts = set(task_info.keys()) & set(extra_config.keys())
            if conflicts:
                logger.warning(
                    "extra_config keys already exist in task_info; keeping existing values: "
                    f"{sorted(conflicts)}"
                )
            for k, v in extra_config.items():
                if k not in task_info:
                    task_info[k] = v

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

    def build_task_info(
        self, task_info: Dict, model_cfg: Dict, infer_cfg: Dict, tasks_cfg: Dict
    ) -> Dict:
        """
        Build basic task info from model_cfg and infer_cfg
        """
        base_configs = ["output_dir", "model_path", "num_workers"]
        for config in base_configs:
            if config in model_cfg:
                task_info[config] = model_cfg[config]
            elif config in infer_cfg:
                task_info[config] = infer_cfg[config]
            elif config in tasks_cfg:
                task_info[config] = tasks_cfg[config]
            else:
                raise ValueError(
                    f"Config {config} not found in task_info, model_cfg, or infer_cfg"
                )

        # Preserve full user configs for adapters that need extra knobs.
        # This is backward-compatible since it only adds new keys.
        task_info["model_cfg"] = model_cfg or {}
        task_info["infer_cfg"] = infer_cfg or {}
        task_info["tasks_cfg"] = tasks_cfg or {}
        return task_info

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
        result: List[Union[ProcessResult, Dict[str, Any]]],
        meta_info: Dict[str, Any],
        rank: int | None = None,
    ) -> None:
        if rank is None:
            output_file = osp.join(meta_info["output_dir"], f"{meta_info['name']}.json")
        else:
            output_file = osp.join(
                meta_info["output_dir"], f"{meta_info['name']}_rank{rank}.json"
            )

        # Convert ProcessResult to dictionary format
        serializable_result = []
        for item in result:
            if isinstance(item, dict):
                serializable_result.append(item)
            elif isinstance(item, ProcessResult):
                serializable_result.append(item.to_dict())
            else:
                raise NotImplementedError

        try:
            serializable_result = sorted(
                serializable_result, key=lambda x: x.get("question_id", "")
            )
            with open(output_file, "w") as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.info(f"Error saving result: {e}")
            with open(output_file, "w") as f:
                json.dump(serializable_result, f, indent=2, ensure_ascii=True)

    def save_item(
        self,
        result: Union[ProcessResult, Dict[str, Any]],
        question_id: str,
        meta_info: Dict[str, Any],
    ):
        output_dir = osp.join(meta_info["output_dir"], "items")
        os.makedirs(output_dir, exist_ok=True)

        # Convert ProcessResult to dictionary format
        if isinstance(result, dict):
            serializable_result = result
        else:
            serializable_result = result.to_dict()

        with open(osp.join(output_dir, f"{question_id}.json"), "w") as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)

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
