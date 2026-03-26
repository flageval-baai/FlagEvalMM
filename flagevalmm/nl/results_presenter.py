import json
import os.path as osp


def collect_results(output_dir: str, task_ids: list[str]) -> dict:
    """Read evaluation results from output directory."""
    results = {}
    for task_id in task_ids:
        task_dir = osp.join(output_dir, task_id)
        result_file = osp.join(task_dir, f"{task_id}_result.json")
        pred_file = osp.join(task_dir, f"{task_id}.json")

        if not osp.exists(task_dir):
            continue

        entry: dict = {"task_id": task_id, "status": "not_found"}

        if osp.exists(result_file):
            with open(result_file) as f:
                data = json.load(f)
            # Extract the main accuracy metric
            if "overall" in data:
                overall = data["overall"]
                entry["accuracy"] = overall.get("accuracy", 0)
                entry["num_samples"] = overall.get("total_number", 0)
            elif "accuracy" in data:
                entry["accuracy"] = data["accuracy"]
            else:
                # Try to find any accuracy-like key
                for key in data:
                    if (
                        isinstance(data[key], (int, float))
                        and "accuracy" in key.lower()
                    ):
                        entry["accuracy"] = data[key]
                        break
            entry["status"] = "completed"
            entry["raw_results"] = data

        elif osp.exists(pred_file):
            with open(pred_file) as f:
                preds = json.load(f)
            entry["num_samples"] = len(preds)
            # Try to compute accuracy from predictions
            correct = sum(1 for p in preds if p.get("correct", False))
            if preds:
                entry["accuracy"] = round(correct / len(preds) * 100, 1)
            entry["status"] = "completed"

        if entry["status"] != "not_found":
            results[task_id] = entry

    return results


def format_results_table(
    model_name: str, results: dict, catalog_entries: list[dict]
) -> str:
    """Format results as an ASCII table."""
    entry_map = {e["task_id"]: e for e in catalog_entries}

    lines = []
    lines.append(f"\n{'═' * 60}")
    lines.append(f"  Results: {model_name}")
    lines.append(f"{'═' * 60}")
    lines.append(f"  {'Task':<28} {'Accuracy':>10} {'Samples':>10}")
    lines.append(f"  {'─' * 28} {'─' * 10} {'─' * 10}")

    for task_id, result in sorted(results.items()):
        display = entry_map.get(task_id, {}).get("display_name", task_id)
        if len(display) > 28:
            display = display[:25] + "..."
        acc = result.get("accuracy")
        acc_str = (
            f"{acc * 100:.1f}%"
            if isinstance(acc, float) and acc <= 1
            else f"{acc:.1f}%" if acc is not None else "N/A"
        )
        samples = result.get("num_samples", "?")
        lines.append(f"  {display:<28} {acc_str:>10} {str(samples):>10}")

    lines.append(f"{'═' * 60}\n")
    return "\n".join(lines)


def format_comparison_table(
    model_results: list[tuple[str, dict]], catalog_entries: list[dict]
) -> str:
    """Format comparison results as a side-by-side table."""
    entry_map = {e["task_id"]: e for e in catalog_entries}

    # Collect all task IDs
    all_task_set: set[str] = set()
    for _, results in model_results:
        all_task_set.update(results.keys())
    all_tasks = sorted(all_task_set)

    model_names = [name for name, _ in model_results]
    col_width = max(12, max((len(n) for n in model_names), default=12))

    lines = []
    lines.append(f"\n{'═' * (32 + (col_width + 3) * len(model_names))}")
    lines.append("  Model Comparison")
    lines.append(f"{'═' * (32 + (col_width + 3) * len(model_names))}")

    header = f"  {'Task':<28}"
    for name in model_names:
        short = name.split("/")[-1] if "/" in name else name
        header += f" {short:>{col_width}}"
    lines.append(header)
    lines.append(f"  {'─' * 28}" + f" {'─' * col_width}" * len(model_names))

    for task_id in all_tasks:
        display = entry_map.get(task_id, {}).get("display_name", task_id)
        if len(display) > 28:
            display = display[:25] + "..."
        row = f"  {display:<28}"
        for _, results in model_results:
            result = results.get(task_id)
            if result:
                acc = result.get("accuracy")
                acc_str = (
                    f"{acc * 100:.1f}%"
                    if isinstance(acc, float) and acc <= 1
                    else f"{acc:.1f}%" if acc is not None else "N/A"
                )
            else:
                acc_str = "-"
            row += f" {acc_str:>{col_width}}"
        lines.append(row)

    lines.append(f"{'═' * (32 + (col_width + 3) * len(model_names))}\n")
    return "\n".join(lines)
