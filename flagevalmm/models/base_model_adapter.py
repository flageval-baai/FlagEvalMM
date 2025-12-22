import json
import copy
from typing import List, Dict, Any, Callable, Optional, Union
import os.path as osp
from accelerate import Accelerator
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta
from torch.utils.data import DataLoader

import os
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
    task_entries: List[Dict[str, Any]],
    data_root: Optional[str] = None,
    debug: bool = False,
) -> Dict[str, Config]:
    config_dict = {}
    if not isinstance(task_entries, list) or not task_entries:
        raise ValueError("tasks.files must be a non-empty list")

    for idx, task in enumerate(task_entries):
        if not isinstance(task, dict):
            raise ValueError(
                f"tasks.files[{idx}] must be a dict like {{file, data_root}}, got: {type(task)}"
            )
        task_config_file = task.get("file")
        if not task_config_file:
            raise ValueError(f"tasks.files[{idx}] missing required key 'file'")
        # Per-task data_root overrides global fallback.
        task_data_root = task.get("data_root", data_root)
        cfg = Config.fromfile(task_config_file, lazy_import=False)
        task_name = cfg.dataset.name
        cfg = merge_args(cfg, task_config_file, task_data_root, debug)
        maybe_register_class(cfg, task_config_file)
        config_dict[task_name] = cfg
    logger.info(f"Loaded {len(config_dict)} tasks: {config_dict.keys()}")
    return config_dict


class TaskManager:
    def __init__(
        self,
        task_names: List[Dict[str, Any]] = None,
        model_path: str = None,
        task_config: Dict = None,
        **kwargs,
    ):
        assert task_names is not None, "tasks.files must be provided in local mode"
        self.task_names = task_names
        try_run = task_config.get("try_run", False)
        config_dict = load_tasks(
            task_names,
            debug=try_run,
            data_root=task_config.get("data_root", None),
        )

        self.evaluation_server = EvaluationServer(
            config_dict,
            output_dir=task_config.get("output_dir", None),
            model_path=model_path,
            debug=try_run,
            quiet=task_config.get("quiet", False),
        )

    def get_task_info(self):
        task_info = {
            "task_names": list(self.evaluation_server.config_dict.keys()),
            "model_path": self.evaluation_server.model_path,
        }
        return task_info

    def get_meta_info(self, task_name: str):
        return self.evaluation_server.get_task_meta_info(task_name)

    def submit(self, task_name: str, model_name: str, output_dir: str):
        self.evaluation_server.evaluate_task(task_name, model_name, output_dir)

    def get_data(self, task_name: str, index: int):
        return self.evaluation_server.get_task_data_by_index(task_name, index)


class BaseModelAdapter:
    def __init__(
        self,
        enable_accelerate: bool = True,
        extra_cfg: str | Dict | None = None,
        task_names: List[Dict[str, Any]] = None,
        **kwargs,
    ) -> None:

        extra_cfg_nested = load_run_cfg(extra_cfg)

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
        extra_args = (
            extra_cfg_nested.get("extra_args", {})
            if isinstance(extra_cfg_nested.get("extra_args", {}), dict)
            else {}
        )

        task_names = tasks_cfg.get("files", task_names)
        model_path = model_cfg.get("model_path") or model_cfg.get("model_name")

        self.task_manager = TaskManager(
            task_names=task_names,
            model_path=model_path,
            task_config=tasks_cfg,
            **kwargs,
        )
        task_info = self.task_manager.get_task_info()
        self.tasks = task_info["task_names"]
        task_info = self.build_task_info(task_info, model_cfg, infer_cfg, tasks_cfg)

        # We keep both:
        # - `task_info["extra_args"]` for traceability
        # - top-level keys (non-overriding) for adapter convenience
        task_info["extra_args"] = extra_args
        conflicts = set(task_info.keys()) & set(extra_args.keys())
        if conflicts:
            logger.warning(
                "extra_args keys already exist in task_info; keeping existing values: "
                f"{sorted(conflicts)}"
            )
        for k, v in extra_args.items():
            if k not in task_info:
                task_info[k] = v

        self.task_info = task_info
        self.model_name: str = task_info.get("model_name", None)
        if self.model_name is None and "model_path" in task_info:
            self.model_name = osp.basename(task_info["model_path"])
        if not task_info.get("model_path"):
            task_info["model_path"] = self.model_name
        if enable_accelerate:
            kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=5400))
            self.accelerator = Accelerator(kwargs_handlers=[kwargs])
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
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
            # Save task_info
            task_info = copy.deepcopy(self.task_info)
            task_info.pop("task_names")
            task_info.update(meta_info)
            with open(osp.join(meta_info["output_dir"], "task_info.json"), "w") as f:
                json.dump(task_info, f, indent=2, ensure_ascii=True)
            self.run_one_task(task_name, meta_info)
            should_submit = self.accelerator is None or self.accelerator.is_main_process
            if should_submit:
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
        output_dir = self.get_items_dir(meta_info)
        os.makedirs(output_dir, exist_ok=True)

        # Convert ProcessResult to dictionary format
        if isinstance(result, dict):
            serializable_result = result
        else:
            serializable_result = result.to_dict()

        serializable_result = self.preprocess_item_for_save(
            serializable_result, question_id=question_id, meta_info=meta_info
        )

        with open(self.get_item_path(question_id, meta_info), "w") as f:
            json.dump(serializable_result, f, indent=2, ensure_ascii=False)

    def get_items_dir(self, meta_info: Dict[str, Any]) -> str:
        return osp.join(meta_info["output_dir"], "items")

    def get_item_path(self, question_id: str, meta_info: Dict[str, Any]) -> str:
        return osp.join(self.get_items_dir(meta_info), f"{question_id}.json")

    def load_item_if_exists(self, question_id: str, meta_info: Dict[str, Any]) -> Any:
        """
        Try to load cached per-item json from items/{question_id}.json.

        Returns:
            dict if exists and readable, otherwise None.
        """
        path = self.get_item_path(question_id, meta_info)
        if not osp.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read existing item {path}, regenerating: {e}")
            return None

    def preprocess_item_for_save(
        self, item: Dict[str, Any], question_id: str, meta_info: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Hook for subclasses to augment item JSON before writing to disk.

        Default: return item unchanged.
        """
        return item

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
