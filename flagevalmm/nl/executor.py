import json
import os
import subprocess
import tempfile


def execute(
    task_paths: list[str],
    model_config: dict,
    output_dir: str,
    try_run: bool = False,
    skip: bool = True,
) -> str:
    """Run FlagEvalMM evaluation via subprocess.

    Returns the output directory path.
    """
    # Write model config to a temp file
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        json.dump(model_config, f)
        cfg_path = f.name

    try:
        cmd = ["flagevalmm", "--tasks"] + task_paths + ["--cfg", cfg_path]
        if try_run:
            cmd.append("--try-run")
        if skip:
            cmd.append("--skip")
        cmd.extend(["--output-dir", output_dir])

        print(f"\n  Running: {' '.join(cmd[:6])} ... ({len(task_paths)} tasks)")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"  Warning: evaluation exited with code {result.returncode}")
    finally:
        os.unlink(cfg_path)

    return output_dir


def execute_comparison(
    task_paths: list[str],
    model_configs: list[tuple[str, dict]],
    base_output_dir: str,
    try_run: bool = False,
) -> list[tuple[str, str]]:
    """Run evaluations for multiple models sequentially.

    Args:
        model_configs: List of (model_display_name, config_dict) tuples.

    Returns:
        List of (model_display_name, output_dir) tuples.
    """
    results = []
    for model_name, config in model_configs:
        safe_name = model_name.replace("/", "_").replace(" ", "_")
        output_dir = os.path.join(base_output_dir, safe_name)
        os.makedirs(output_dir, exist_ok=True)

        config_with_output = {**config, "output_dir": output_dir}
        execute(task_paths, config_with_output, output_dir, try_run=try_run)
        results.append((model_name, output_dir))

    return results
