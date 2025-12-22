import subprocess
import requests
import time
import atexit
import os.path as osp
import argparse
from mmengine.config import Config
from flagevalmm.common.logger import get_logger
from flagevalmm.server.utils import get_random_port, RunCfg, load_run_cfg_with_defaults
from flagevalmm.registry import EVALUATORS, DATASETS
from flagevalmm.server.utils import maybe_register_class, merge_args, parse_args
import os
import signal
from omegaconf import OmegaConf
from typing import Any

logger = get_logger(__name__)


def _resolve_output_dir(cfg_from_file: dict) -> str:
    if isinstance(cfg_from_file, dict):
        tasks_cfg = (
            cfg_from_file.get("tasks", {})
            if isinstance(cfg_from_file.get("tasks", {}), dict)
            else {}
        )
        if tasks_cfg.get("output_dir"):
            return str(tasks_cfg["output_dir"])

    # fallback timestamped dir
    model_name = None
    if isinstance(cfg_from_file, dict):
        model_cfg = (
            cfg_from_file.get("model", {})
            if isinstance(cfg_from_file.get("model", {}), dict)
            else {}
        )
        model_name = (
            model_cfg.get("model_name")
            or model_cfg.get("model_path")
            or cfg_from_file.get("model_name")
        )
        if model_name is None:
            model_name = model_cfg.get("exec")
    if not model_name:
        model_name = "run"
    model_name = str(model_name).split("/")[-1]
    return f"{model_name}_{time.strftime('%Y%m%d_%H%M%S')}"


def _normalize_tasks(tasks_list, global_data_root):
    """
    Normalize task entries to a list of dicts with keys: file, data_root.
    Accepts string entries for backward compatibility.
    """
    if not isinstance(tasks_list, list):
        logger.warning("tasks.files is not a list, nothing to run")
        return []

    normalized = []
    for idx, task in enumerate(tasks_list):
        if isinstance(task, str):
            normalized.append({"file": task, "data_root": global_data_root})
            continue

        if isinstance(task, dict):
            task_file = task.get("file")
            if not task_file:
                logger.warning(f"Task entry at index {idx} missing 'file', skip")
                continue
            data_root = task.get("data_root", global_data_root)
            normalized.append({"file": task_file, "data_root": data_root})
            continue

        logger.warning(f"Unsupported task entry at index {idx}: {task}, skip")
    return normalized


def build_run_cfg(
    cfg_path: str, exec_override: str | None = None
) -> Any:
    """
    Build a structured run config (YAML-friendly) using dataclass defaults + user config file.

    Note: This is runtime config only; task dataset configs in tasks/ are not modified.
    """
    cfg = load_run_cfg_with_defaults(cfg_path)

    # Optional exec override without touching YAML
    if exec_override:
        cfg.setdefault("model", {})
        if isinstance(cfg["model"], dict):
            cfg["model"]["exec"] = exec_override

    # fill output_dir if still missing
    cfg.setdefault("tasks", {})
    if isinstance(cfg["tasks"], dict) and not cfg["tasks"].get("output_dir"):
        cfg["tasks"]["output_dir"] = _resolve_output_dir(cfg)

    return cfg


class ServerWrapper:
    def filter_finished_tasks(self, tasks, output_dir):
        finished_tasks = []
        for task in tasks:
            task_file = task.get("file") if isinstance(task, dict) else task
            if not task_file:
                logger.warning("Task entry without file path, skip")
                continue
            task_cfg = Config.fromfile(task_file)
            task_name = task_cfg.dataset.name
            if osp.exists(osp.join(output_dir, task_name, f"{task_name}.json")):
                logger.info(f"Task {task_name} already finished, skip")
                continue
            finished_tasks.append(task)
        return finished_tasks

    def __init__(self, args):
        self.args = args
        self.exec = args.exec
        self.infer_process = None
        # Register cleanup at exit
        atexit.register(self.cleanup)
        self.run_cfg_path = None
        self.run_cfg = None
        self.output_dir = None

    def start(self):
        """Main method to start the server and run the model"""
        self.run_cfg = build_run_cfg(
            cfg_path=self.args.cfg,
            exec_override=self.exec,
        )
        # Adapter entrypoint: prefer config, then CLI override, then default.
        model_cfg = (
            self.run_cfg.get("model", {})
            if isinstance(self.run_cfg.get("model", {}), dict)
            else {}
        )
        if model_cfg.get("exec"):
            self.exec = model_cfg.get("exec")
        if self.exec is None:
            logger.warning(
                "`--exec` is not provided, using default value: model_zoo/vlm/api_model/model_adapter.py"
            )
            self.exec = "model_zoo/vlm/api_model/model_adapter.py"

        self.output_dir = self.run_cfg["tasks"]["output_dir"]

        tasks_cfg = (
            self.run_cfg.get("tasks", {})
            if isinstance(self.run_cfg.get("tasks", {}), dict)
            else {}
        )
        tasks = _normalize_tasks(tasks_cfg.get("files", []), tasks_cfg.get("data_root"))
        if not tasks:
            logger.info("No tasks to run, exit")
            return
        self.run_cfg["tasks"]["files"] = tasks

        if tasks_cfg.get("skip", True):
            tasks = self.filter_finished_tasks(tasks, self.output_dir)
            if len(tasks) == 0:
                logger.info("No tasks to run after filtering finished tasks, exit")
                return
            self.run_cfg["tasks"]["files"] = tasks

        # Persist run cfg as YAML (so adapters can read it via --cfg PATH)
        os.makedirs(self.output_dir, exist_ok=True)
        self.run_cfg_path = osp.join(self.output_dir, "run_config.yaml")
        OmegaConf.save(config=OmegaConf.create(self.run_cfg), f=self.run_cfg_path)
        self.run_model_adapter()

    def run_model_adapter(self):
        try:
            use_torchrun, num_procs = self._should_use_torchrun()
            command = self._build_command(
                use_torchrun=use_torchrun, num_procs=num_procs
            )
            if use_torchrun:
                logger.info(f"Launching adapter with torchrun ({num_procs} processes)")
            # Create a new process group
            print(f"command: {command}")
            self.infer_process = subprocess.Popen(
                command,
                preexec_fn=os.setsid if os.name != "nt" else None,
                creationflags=(
                    0 if os.name != "nt" else subprocess.CREATE_NEW_PROCESS_GROUP
                ),
            )
            logger.info(f"Started process with PID: {self.infer_process.pid}")
            self.infer_process.wait()
        except KeyboardInterrupt:
            logger.info("Received interrupt, cleaning up...")
            self.cleanup()
        finally:
            logger.info("Command execution finished.")

    def _should_use_torchrun(self) -> tuple[bool, int]:
        """Use torchrun only for uni adapters when num_workers > 1."""
        tasks_cfg = (
            self.run_cfg.get("tasks", {})
            if isinstance(self.run_cfg.get("tasks", {}), dict)
            else {}
        )
        num_workers = max(1, int(tasks_cfg.get("num_workers") or 1))
        exec_path = osp.normpath(self.exec or "")
        uni_marker = f"model_zoo{osp.sep}uni{osp.sep}"
        is_uni_adapter = uni_marker in exec_path and exec_path.endswith(".py")
        return is_uni_adapter and num_workers > 1, num_workers

    def _build_command(self, use_torchrun: bool = False, num_procs: int = 1):
        """Private method to build the command for model execution"""
        command = []
        if self.exec.endswith("py"):
            assert osp.exists(self.exec), f"model path {self.exec} not found"
            if use_torchrun and num_procs > 1:
                command += [
                    "torchrun",
                    "--nproc_per_node",
                    str(num_procs),
                    self.exec,
                ]
            else:
                command += [
                    "python",
                    self.exec,
                ]

        else:
            assert osp.exists(f"{self.exec}/run.sh"), f"run.sh not found in {self.exec}"
            command += [
                "bash",
                f"{self.exec}/run.sh",
            ]

        command.extend(["--cfg", self.run_cfg_path])
        return command


    def cleanup(self):
        if self.infer_process:
            try:
                if os.name != "nt":  # Unix
                    os.killpg(os.getpgid(self.infer_process.pid), signal.SIGTERM)
                    # Add a wait time for the process to complete cleanup
                else:  # Windows
                    self.infer_process.terminate()

                try:
                    self.infer_process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Process didn't terminate, forcing kill")
                    if os.name != "nt":
                        os.killpg(os.getpgid(self.infer_process.pid), signal.SIGKILL)
                    else:
                        self.infer_process.kill()
                    self.infer_process.wait()
            except Exception:
                logger.info(f"The process {self.infer_process.pid} is killed")
            finally:
                self.infer_process = None

        logger.info("ServerWrapper cleanup completed")


def evaluate_only(args):
    # Keep evaluate_only behavior: output_dir resolution uses same logic as run_cfg
    run_cfg = build_run_cfg(
        args.cfg, exec_override=args.exec
    )
    output_root = run_cfg["tasks"]["output_dir"]

    tasks_cfg = (
        run_cfg.get("tasks", {}) if isinstance(run_cfg.get("tasks", {}), dict) else {}
    )
    tasks = _normalize_tasks(tasks_cfg.get("files", []), tasks_cfg.get("data_root"))
    if not tasks:
        logger.info("No tasks to run, exit")
        return

    # Derive runtime flags for task config patching from merged run_cfg.
    try_run = bool(tasks_cfg.get("try_run", False))

    for task in tasks:
        task_file = task.get("file") if isinstance(task, dict) else task
        data_root = task.get("data_root") if isinstance(task, dict) else None
        runtime_args = argparse.Namespace(
            try_run=try_run, data_root=data_root
        )

        task_cfg = Config.fromfile(task_file)
        task_cfg = merge_args(task_cfg, task_file, runtime_args)
        maybe_register_class(task_cfg, task_file)

        if "evaluator" in task_cfg:
            dataset = DATASETS.build(task_cfg.dataset)
            evaluator = EVALUATORS.build(task_cfg.evaluator)

            task_name = task_cfg.dataset.name
            output_dir = osp.join(output_root, task_name)
            model_name = (
                run_cfg.get("model", {}).get("model_name", "")
                if isinstance(run_cfg.get("model", {}), dict)
                else ""
            )
            evaluator.process(dataset, output_dir, model_name=model_name)
        else:
            logger.error(f"No evaluator found in config {task_file}")


def run():
    args = parse_args()
    if args.without_infer:
        evaluate_only(args)
    else:
        server = ServerWrapper(args)
        server.start()


if __name__ == "__main__":
    run()
