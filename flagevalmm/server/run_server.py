import argparse
from typing import List, Dict
from mmengine.config import Config
from flagevalmm.server.evaluation_server import EvaluationServer
from flagevalmm.common.logger import get_logger
from flagevalmm.server.utils import maybe_register_class, merge_args, load_run_cfg

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Run evaluation server")
    # Only nested runtime config is supported.
    parser.add_argument(
        "--cfg", "-c", type=str, required=True, help="runtime config file (yaml/json)"
    )
    # Small runtime overrides (not part of RunCfg; keep minimal CLI surface)
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="server bind host"
    )
    parser.add_argument(
        "--port", type=int, default=None, help="override cfg.server.port"
    )
    parser.add_argument(
        "--quiet",
        action="store_const",
        const=True,
        default=None,
        help="override cfg.server.quiet",
    )
    args = parser.parse_args()
    return args


def load_tasks(
    task_entries: List, runtime_args: argparse.Namespace
) -> Dict[str, Config]:
    config_dict = {}
    if not isinstance(task_entries, list) or not task_entries:
        raise ValueError("cfg.tasks.files must be a non-empty list")

    # Allow per-task data_root: tasks.files = [{file, data_root}, ...]
    global_data_root = getattr(runtime_args, "data_root", None)

    for idx, task in enumerate(task_entries):
        if not isinstance(task, dict):
            raise ValueError(
                f"cfg.tasks.files[{idx}] must be a dict like {{file, data_root}}, got: {type(task)}"
            )
        task_config_file = task.get("file")
        if not task_config_file:
            raise ValueError(f"cfg.tasks.files[{idx}] missing required key 'file'")

        task_data_root = task.get("data_root", global_data_root)
        per_task_args = argparse.Namespace(
            debug=getattr(runtime_args, "debug", False),
            try_run=getattr(runtime_args, "try_run", False),
            data_root=task_data_root,
        )
        cfg = Config.fromfile(task_config_file, lazy_import=False)
        task_name = cfg.dataset.name
        cfg = merge_args(cfg, task_config_file, per_task_args)
        maybe_register_class(cfg, task_config_file)
        config_dict[task_name] = cfg
    logger.info(f"Loaded {len(config_dict)} tasks: {config_dict.keys()}")
    return config_dict


if __name__ == "__main__":
    args = parse_args()

    cfg = load_run_cfg(args.cfg)
    if not isinstance(cfg, dict):
        raise ValueError("Invalid cfg: expected a mapping")

    tasks_cfg = cfg.get("tasks", {}) if isinstance(cfg.get("tasks", {}), dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg.get("model", {}), dict) else {}
    server_cfg = (
        cfg.get("server", {}) if isinstance(cfg.get("server", {}), dict) else {}
    )

    task_files = tasks_cfg.get("files", [])
    if not isinstance(task_files, list) or not task_files:
        raise ValueError("No tasks provided in cfg.tasks.files")

    output_dir = tasks_cfg.get("output_dir")
    if not output_dir:
        raise ValueError("No output_dir provided in cfg.tasks.output_dir")
    output_dir = str(output_dir)

    model_path = model_cfg.get("model_path") or model_cfg.get("model_name")
    model_path = str(model_path) if model_path else None

    port = int(server_cfg.get("port", 5000)) if args.port is None else int(args.port)
    quiet = bool(server_cfg.get("quiet", False))
    if getattr(args, "quiet", None) is True:
        quiet = True

    # Derive runtime flags for task config patching.
    try_run = bool(tasks_cfg.get("try_run", False))
    debug = bool(tasks_cfg.get("debug", False) or try_run)
    data_root = tasks_cfg.get("data_root", None)

    runtime_args = argparse.Namespace(debug=debug, try_run=try_run, data_root=data_root)

    config_dict = load_tasks(task_files, runtime_args)
    server = EvaluationServer(
        config_dict,
        output_dir=output_dir,
        model_path=model_path,
        port=port,
        host=args.host,
        debug=debug,
        quiet=quiet,
    )
    server.run()
