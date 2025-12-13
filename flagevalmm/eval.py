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


def build_run_cfg(
    cfg_path: str, chosen_server_port: int, exec_override: str | None = None
) -> Any:
    """
    Build a structured run config (YAML-friendly) using dataclass defaults + user config file.

    Note: This is runtime config only; task dataset configs in tasks/ are not modified.
    """
    cfg = load_run_cfg_with_defaults(cfg_path)

    # Force runtime-chosen port into config to match launched eval server.
    cfg.setdefault("server", {})
    if isinstance(cfg["server"], dict):
        cfg["server"]["port"] = chosen_server_port

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
        for task_file in tasks:
            task_cfg = Config.fromfile(task_file)
            task_name = task_cfg.dataset.name
            if osp.exists(osp.join(output_dir, task_name, f"{task_name}.json")):
                logger.info(f"Task {task_name} already finished, skip")
                continue
            finished_tasks.append(task_file)
        return finished_tasks

    def __init__(self, args):
        self.args = args
        self.exec = args.exec
        self.port = None
        self.evaluation_server_ip = None
        self.local_mode = None
        self.evaluation_server = None
        self.evaluation_server_pid = None
        self.infer_process = None
        # Register cleanup at exit
        atexit.register(self.cleanup)
        self.run_cfg_path = None
        self.run_cfg = None
        self.output_dir = None

    def start(self):
        """Main method to start the server and run the model"""
        # Decide evaluation server port first (remote uses random port; local/disabled uses cfg).
        cfg_preview = load_run_cfg_with_defaults(self.args.cfg)
        server_preview = (
            cfg_preview.get("server", {})
            if isinstance(cfg_preview.get("server", {}), dict)
            else {}
        )
        defaults_obj = RunCfg()

        disable_eval = bool(
            server_preview.get(
                "disable_evaluation_server",
                defaults_obj.server.disable_evaluation_server,
            )
        )
        local_mode = bool(
            server_preview.get("local_mode", defaults_obj.server.local_mode)
        )

        # Base port/ip for local/disabled mode.
        base_port = server_preview.get("port", defaults_obj.server.port)

        if disable_eval or local_mode:
            self.port = int(base_port)
        else:
            self.port = get_random_port()
            logger.info(f"Using port {self.port}")

        # Build config after knowing port
        self.run_cfg = build_run_cfg(
            cfg_path=self.args.cfg,
            chosen_server_port=self.port,
            exec_override=self.exec,
        )
        server_cfg = self.run_cfg.get("server", {})
        if isinstance(server_cfg, dict):
            self.evaluation_server_ip = server_cfg.get("ip", defaults_obj.server.ip)
            self.local_mode = server_cfg.get(
                "local_mode", defaults_obj.server.local_mode
            )
        else:
            self.evaluation_server_ip = defaults_obj.server.ip
            self.local_mode = defaults_obj.server.local_mode

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

        if not self.run_cfg.get("tasks", {}).get("files", []):
            logger.info("No tasks to run, exit")
            return

        tasks = self.run_cfg.get("tasks", {}).get("files", [])
        if self.run_cfg.get("tasks", {}).get("skip", True):
            tasks = self.filter_finished_tasks(tasks, self.output_dir)
            if len(tasks) == 0:
                logger.info("No tasks to run after filtering finished tasks, exit")
                return

        # Persist run cfg as YAML (so adapters can read it via --cfg PATH)
        os.makedirs(self.output_dir, exist_ok=True)
        self.run_cfg_path = osp.join(self.output_dir, "run_config.yaml")
        OmegaConf.save(config=OmegaConf.create(self.run_cfg), f=self.run_cfg_path)

        # Launch server (needs output_dir), then run adapter
        self.maybe_launch_evaluation_server(server_cfg, self.output_dir)
        self.run_model_adapter()

    def run_model_adapter(self):
        try:
            command = self._build_command()
            # Create a new process group
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

    def _build_command(self):
        """Private method to build the command for model execution"""
        command = []
        if self.exec.endswith("py"):
            assert osp.exists(self.exec), f"model path {self.exec} not found"
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

    def maybe_launch_evaluation_server(self, server_cfg, output_dir):
        if server_cfg.get("local_mode", True) or server_cfg.get(
            "disable_evaluation_server", True
        ):
            self.evaluation_server = None
            return
        assert output_dir, "output_dir is required when launching evaluation server"
        # Prefer nested runtime config (run_config.yaml) instead of passing flat flags.
        # run_server.py reads tasks/model/server from --cfg.
        command = [
            "python",
            "flagevalmm/server/run_server.py",
            "--cfg",
            self.run_cfg_path,
        ]
        if server_cfg.get("quiet", False):
            command.append("--quiet")

        self.evaluation_server = subprocess.Popen(command, start_new_session=True)
        # wait for server to start or timeout
        for _ in range(10):
            if self.is_server_running():
                break
            time.sleep(10)
        else:
            raise RuntimeError("Server failed to start")

        self.evaluation_server_pid = self.evaluation_server.pid

    def is_server_running(self):
        try:
            response = requests.get(
                f"{self.evaluation_server_ip}:{self.port}/task_info", timeout=10
            )
            return response.status_code == 200
        except requests.RequestException:
            return False

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

        if not hasattr(self, "evaluation_server") or self.evaluation_server is None:
            return
        try:
            if self.is_server_running():
                # check if eval_finished
                while True:
                    response = requests.get(
                        f"{self.evaluation_server_ip}:{self.port}/eval_finished"
                    )
                    if response.json()["status"] == 1:
                        break
                    time.sleep(10)
                    logger.info("Waiting for eval to finish...")
                self.evaluation_server.terminate()
                self.evaluation_server.wait()
                logger.info(
                    f"Server with PID {self.evaluation_server_pid} has been terminated.\nPort {self.port} is released"
                )
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        logger.info("ServerWrapper cleanup completed")


def evaluate_only(args):
    # Keep evaluate_only behavior: output_dir resolution uses same logic as run_cfg
    cfg_preview = load_run_cfg_with_defaults(args.cfg)
    server_cfg = (
        cfg_preview.get("server", {})
        if isinstance(cfg_preview.get("server", {}), dict)
        else {}
    )
    defaults_obj = RunCfg()
    chosen_port = server_cfg.get("port", defaults_obj.server.port)
    run_cfg = build_run_cfg(
        args.cfg, chosen_server_port=int(chosen_port), exec_override=args.exec
    )
    output_root = run_cfg["tasks"]["output_dir"]

    tasks = run_cfg.get("tasks", {}).get("files", [])
    if not tasks:
        logger.info("No tasks to run, exit")
        return

    # Derive runtime flags for task config patching from merged run_cfg.
    tasks_cfg = (
        run_cfg.get("tasks", {}) if isinstance(run_cfg.get("tasks", {}), dict) else {}
    )
    try_run = bool(tasks_cfg.get("try_run", False))
    debug = bool(tasks_cfg.get("debug", False) or try_run)
    data_root = tasks_cfg.get("data_root", None)
    runtime_args = argparse.Namespace(debug=debug, try_run=try_run, data_root=data_root)

    for task_file in tasks:
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
