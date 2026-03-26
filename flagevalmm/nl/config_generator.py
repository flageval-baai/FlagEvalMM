import os
from typing import Optional

from .intent_parser import EvalIntent, ModelSpec
from .task_catalog import TaskCatalog


def select_tasks(intent: EvalIntent, catalog: TaskCatalog) -> list[dict]:
    """Select task catalog entries matching the evaluation intent."""
    results = []

    # Priority 1: explicit benchmark names
    if intent.specific_benchmarks:
        results = catalog.search_by_benchmark_names(intent.specific_benchmarks)

    # Priority 2: capability-based selection
    if not results and intent.capabilities:
        results = catalog.search_by_capability(intent.capabilities)

    # Deduplicate by task_id
    seen = set()
    unique = []
    for entry in results:
        if entry["task_id"] not in seen:
            seen.add(entry["task_id"])
            unique.append(entry)
    return unique


def generate_model_config(spec: ModelSpec) -> dict:
    """Generate a model config dict from a ModelSpec."""
    cfg = {
        "model_name": spec.name,
        "num_workers": 3,
        "max_image_size": 4718592,
        "max_tokens": 28000,
        "max_long_side": 1000,
        "use_cache": False,
    }

    # Determine provider and URL
    provider = spec.provider or _detect_provider(spec.name, spec.url)

    if provider == "openrouter":
        cfg["url"] = "https://openrouter.ai/api/v1/chat/completions"
        cfg["api_key"] = _resolve_api_key(
            spec.api_key_env, ["OPEN_ROUTER_KEY", "OPENROUTER_API_KEY"]
        )
    elif provider == "anthropic":
        cfg["model_type"] = "claude"
        cfg["api_key"] = _resolve_api_key(spec.api_key_env, ["ANTHROPIC_API_KEY"])
    elif provider == "google":
        cfg["model_type"] = "gemini"
        cfg["api_key"] = _resolve_api_key(spec.api_key_env, ["GOOGLE_API_KEY"])
    elif provider == "openai":
        cfg["url"] = "https://api.openai.com/v1/chat/completions"
        cfg["api_key"] = _resolve_api_key(spec.api_key_env, ["OPENAI_API_KEY"])
    elif provider == "local":
        cfg["url"] = spec.url or "http://localhost:8000/v1/chat/completions"
        cfg["api_key"] = "EMPTY"
    else:
        # Default: OpenAI-compatible
        cfg["url"] = spec.url or "https://openrouter.ai/api/v1/chat/completions"
        cfg["api_key"] = _resolve_api_key(
            spec.api_key_env,
            ["OPEN_ROUTER_KEY", "OPENROUTER_API_KEY", "OPENAI_API_KEY"],
        )

    # Override URL if explicitly provided
    if spec.url:
        cfg["url"] = spec.url

    return cfg


def _detect_provider(name: str, url: Optional[str]) -> str:
    name_lower = name.lower()
    if url and ("localhost" in url or "127.0.0.1" in url):
        return "local"
    if "/" in name:
        return "openrouter"
    if "claude" in name_lower:
        return "anthropic"
    if "gemini" in name_lower:
        return "google"
    if any(m in name_lower for m in ["gpt", "o1-", "o3-", "o4-"]):
        return "openai"
    return "openrouter"


def _resolve_api_key(
    explicit_env: Optional[str], fallback_envs: list[str]
) -> Optional[str]:
    if explicit_env:
        val = os.environ.get(explicit_env)
        if val:
            return val
    for env in fallback_envs:
        val = os.environ.get(env)
        if val:
            return val
    return None
