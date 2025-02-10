import json
from typing import List, Dict, Any, Callable, Optional
import os.path as osp
from accelerate import Accelerator
from torch.utils.data import DataLoader

import os

from flagevalmm.server.utils import get_meta, get_task_info, submit
from flagevalmm.server.server_dataset import ServerDataset
from flagevalmm.common.logger import get_logger

logger = get_logger(__name__)


class BaseModelAdapter:
    def __init__(
        self,
        server_ip: str,
        server_port: int,
        timeout: int = 1000,
        enable_accelerate: bool = True,
        extra_cfg: str | Dict | None = None,
    ) -> None:
        self.server_ip: str = server_ip
        self.server_port: int = server_port

        self.timeout: int = timeout
        task_info = get_task_info(server_ip, server_port)
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
            meta_info: Dict[str, Any] = get_meta(
                task_name, self.server_ip, self.server_port
            )
            if "output_dir" in self.task_info:
                meta_info["output_dir"] = osp.join(
                    self.task_info["output_dir"], task_name
                )
                os.makedirs(meta_info["output_dir"], exist_ok=True)
            self.run_one_task(task_name, meta_info)
            submit(
                task_name,
                self.model_name,
                server_ip=self.server_ip,
                server_port=self.server_port,
                output_dir=meta_info["output_dir"],
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
        dataset_cls: type[ServerDataset],
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
                    self.server_ip,
                    self.server_port,
                    self.timeout,
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
