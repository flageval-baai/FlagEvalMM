import argparse
import sys
import time

from .config_generator import generate_model_config, select_tasks
from .executor import execute, execute_comparison
from .intent_parser import parse_intent
from .results_presenter import (
    collect_results,
    format_comparison_table,
    format_results_table,
)
from .task_catalog import TaskCatalog


def parse_args():
    parser = argparse.ArgumentParser(
        description="Natural language interface for FlagEvalMM evaluation",
        usage='flagevalmm-nl "Evaluate GPT-5-Mini on math tasks"',
    )
    parser.add_argument("query", type=str, help="Natural language evaluation request")
    parser.add_argument(
        "--yes", "-y", action="store_true", help="Skip confirmation prompt"
    )
    parser.add_argument("--try-run", action="store_true", help="Run on 32 samples only")
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Output directory for results"
    )
    parser.add_argument(
        "--parser-model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use for intent parsing",
    )
    parser.add_argument(
        "--parser-api-key", type=str, default=None, help="API key for the parser model"
    )
    parser.add_argument(
        "--parser-base-url",
        type=str,
        default=None,
        help="Base URL for the parser model",
    )
    parser.add_argument(
        "--catalog", type=str, default=None, help="Path to custom task catalog"
    )
    return parser.parse_args()


def display_plan(intent, selected_tasks, catalog_entries):
    """Display the evaluation plan for user confirmation."""
    entry_map = {e["task_id"]: e for e in catalog_entries}

    print(f"\n{'═' * 50}")
    print("  Evaluation Plan")
    print(f"{'═' * 50}")

    # Models
    print("\n  Model(s):")
    for m in intent.models:
        provider = m.provider or "auto"
        print(f"    - {m.name} (via {provider})")

    # Tasks
    print(f"\n  Tasks ({len(selected_tasks)} selected):")
    for i, task in enumerate(selected_tasks, 1):
        entry = entry_map.get(task["task_id"], task)
        desc = entry.get("description", "")
        if len(desc) > 40:
            desc = desc[:37] + "..."
        print(f"    {i:2d}. {task['task_id']:<28} — {desc}")

    if intent.try_run:
        print("\n  Mode: Try-run (32 samples per task)")
    else:
        print("\n  Mode: Full evaluation")

    print(f"{'═' * 50}")


def confirm_plan(intent, selected_tasks, catalog_entries, auto_yes=False):
    """Show plan and get user confirmation. Returns filtered task list."""
    display_plan(intent, selected_tasks, catalog_entries)

    if auto_yes:
        print("  Auto-confirmed (--yes flag)\n")
        return selected_tasks

    while True:
        response = input("\n  Proceed? [Y/n/edit] ").strip().lower()
        if response in ("", "y", "yes"):
            return selected_tasks
        elif response in ("n", "no"):
            print("  Aborted.")
            sys.exit(0)
        elif response == "edit":
            return _edit_task_list(selected_tasks)
        else:
            print("  Please enter Y, n, or edit")


def _edit_task_list(tasks):
    """Let user toggle tasks by number."""
    enabled = [True] * len(tasks)
    while True:
        print("\n  Tasks (enter number to toggle, 'done' to finish):")
        for i, task in enumerate(tasks):
            status = "✓" if enabled[i] else "✗"
            print(f"    [{status}] {i + 1}. {task['task_id']}")
        response = input("\n  Toggle #: ").strip().lower()
        if response in ("done", "d", ""):
            break
        try:
            idx = int(response) - 1
            if 0 <= idx < len(tasks):
                enabled[idx] = not enabled[idx]
        except ValueError:
            pass
    return [t for t, e in zip(tasks, enabled) if e]


def main():
    args = parse_args()

    # Load catalog
    catalog = TaskCatalog(args.catalog)

    # Parse intent
    print(f'\n  Parsing: "{args.query}"')
    intent = parse_intent(
        args.query,
        catalog,
        api_key=args.parser_api_key,
        base_url=args.parser_base_url,
        model=args.parser_model,
    )

    # Override try_run from CLI flag
    if args.try_run:
        intent.try_run = True

    if not intent.models:
        print(
            "  Error: Could not identify a model to evaluate. Please specify a model name."
        )
        sys.exit(1)

    # Select tasks
    selected = select_tasks(intent, catalog)
    if not selected:
        print("  Error: No matching tasks found for your request.")
        print(f"  Capabilities searched: {intent.capabilities}")
        print(f"  Benchmarks searched: {intent.specific_benchmarks}")
        sys.exit(1)

    # Confirm
    selected = confirm_plan(intent, selected, catalog.entries, auto_yes=args.yes)
    if not selected:
        print("  No tasks selected. Aborted.")
        sys.exit(0)

    # Prepare paths
    task_paths = [t["config_path"] for t in selected]
    task_ids = [t["task_id"] for t in selected]
    timestamp = time.strftime("%Y%m%d_%H%M%S")

    if intent.is_comparison and len(intent.models) > 1:
        # Comparison mode
        base_output = args.output_dir or f"results/comparison_{timestamp}"
        model_configs = [(m.name, generate_model_config(m)) for m in intent.models]
        print(f"\n  Running comparison across {len(intent.models)} models...")
        run_results = execute_comparison(
            task_paths, model_configs, base_output, try_run=intent.try_run
        )

        # Collect and display comparison
        all_model_results = []
        for model_name, output_dir in run_results:
            results = collect_results(output_dir, task_ids)
            all_model_results.append((model_name, results))

        print(format_comparison_table(all_model_results, catalog.entries))
    else:
        # Single model mode
        model_spec = intent.models[0]
        model_config = generate_model_config(model_spec)
        safe_name = model_spec.name.replace("/", "_")
        output_dir = args.output_dir or f"results/{safe_name}_{timestamp}"
        model_config["output_dir"] = output_dir

        print("\n  Starting evaluation...")
        execute(task_paths, model_config, output_dir, try_run=intent.try_run)

        # Collect and display results
        results = collect_results(output_dir, task_ids)
        print(format_results_table(model_spec.name, results, catalog.entries))


if __name__ == "__main__":
    main()
