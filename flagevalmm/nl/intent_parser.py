import json
import os
import os.path as osp
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

from .task_catalog import TaskCatalog


@dataclass
class ModelSpec:
    name: str
    provider: Optional[str] = None
    url: Optional[str] = None
    api_key_env: Optional[str] = None


@dataclass
class EvalIntent:
    models: list[ModelSpec] = field(default_factory=list)
    capabilities: list[str] = field(default_factory=list)
    specific_benchmarks: list[str] = field(default_factory=list)
    is_comparison: bool = False
    try_run: bool = False
    max_tasks: Optional[int] = None
    constraints: dict = field(default_factory=dict)


def _load_system_prompt(catalog: TaskCatalog) -> str:
    prompt_path = osp.join(osp.dirname(__file__), "prompts", "intent_system.txt")
    with open(prompt_path) as f:
        template = f.read()
    benchmarks = "\n".join(
        f"- {e['task_id']}: {e['description']} (capabilities: {', '.join(e['capabilities'])})"
        for e in catalog.entries
    )
    return template.replace("{available_benchmarks}", benchmarks)


def parse_intent_with_llm(
    query: str,
    catalog: TaskCatalog,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> EvalIntent:
    api_key = api_key or os.environ.get(
        "OPENAI_API_KEY", os.environ.get("OPEN_ROUTER_KEY", "")
    )
    if base_url is None and os.environ.get("OPEN_ROUTER_KEY"):
        base_url = "https://openrouter.ai/api/v1"
        api_key = os.environ["OPEN_ROUTER_KEY"]
        model = "openai/gpt-4o-mini"

    client = OpenAI(api_key=api_key, base_url=base_url)
    system_prompt = _load_system_prompt(catalog)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )

    raw = response.choices[0].message.content
    data = json.loads(raw)
    return _dict_to_intent(data)


def parse_intent_with_keywords(query: str, catalog: TaskCatalog) -> EvalIntent:
    """Fallback parser using keyword matching when no LLM is available."""
    query_lower = query.lower()

    # Detect models (ordered longest-first to avoid partial matches)
    models = []
    known_models = [
        ("gpt-5-mini", "openai/gpt-5-mini", "openrouter"),
        ("gpt-4o-mini", "openai/gpt-4o-mini", "openrouter"),
        ("gpt-5", "openai/gpt-5", "openrouter"),
        ("gpt-4o", "openai/gpt-4o", "openrouter"),
        ("claude-sonnet", "anthropic/claude-sonnet-4", "openrouter"),
        ("claude-opus", "anthropic/claude-opus-4", "openrouter"),
        ("claude", "anthropic/claude-sonnet-4", "openrouter"),
        ("gemini-2.5-pro", "google/gemini-2.5-pro", "openrouter"),
        ("gemini-2.5-flash", "google/gemini-2.5-flash", "openrouter"),
        ("gemini", "google/gemini-2.5-flash", "openrouter"),
    ]
    matched_spans: list[tuple[int, int]] = []
    for keyword, name, provider in known_models:
        idx = query_lower.find(keyword)
        if idx >= 0:
            # Check the match is not part of a longer word (e.g. "gpt-5" in "gpt-5-nano")
            end = idx + len(keyword)
            if end < len(query_lower) and (
                query_lower[end].isalnum() or query_lower[end] in "-."
            ):
                continue
            # Check not already covered by a longer match
            if not any(idx >= s and end <= e for s, e in matched_spans):
                models.append(ModelSpec(name=name, provider=provider))
                matched_spans.append((idx, end))
    # If no known model matched, try to extract a model-like token from the query
    if not models:
        import re

        # Match patterns like "gpt-5-nano", "claude-3.5-sonnet", "my-model"
        model_pattern = re.findall(r"\b([a-z][\w.-]*(?:-[\w.]+)+)\b", query_lower)
        if model_pattern:
            # Pick the most model-like token (contains a digit)
            for candidate in model_pattern:
                if any(c.isdigit() for c in candidate):
                    models.append(ModelSpec(name=candidate, provider=None))
                    break
        if not models:
            models.append(ModelSpec(name="openai/gpt-4o-mini", provider="openrouter"))

    # Detect capabilities
    cap_keywords = {
        "vqa": ["vqa", "visual question", "question answering"],
        "math": ["math", "mathematical", "arithmetic", "calculation"],
        "ocr": ["ocr", "text recognition", "text reading"],
        "hallucination": ["hallucination", "hallucinate"],
        "spatial": ["spatial", "3d", "depth", "position"],
        "knowledge": ["knowledge", "academic", "mmmu", "college"],
        "video": ["video", "temporal"],
        "chart": ["chart", "diagram", "graph understanding"],
        "safety": ["safety", "harmful", "toxic"],
        "generation": ["generation", "t2i", "t2v", "text-to-image"],
        "retrieval": ["retrieval", "search", "matching"],
        "embodied": ["embodied", "robot", "robotics"],
        "perception": ["perception", "visual", "recognition", "general"],
        "measurement": ["measurement", "meter", "gauge", "instrument"],
        "grounding": ["grounding", "referring", "bbox", "refcoco"],
    }
    capabilities = []
    for cap, keywords in cap_keywords.items():
        if any(k in query_lower for k in keywords):
            capabilities.append(cap)

    # Detect specific benchmarks
    specific = []
    for entry in catalog.entries:
        tid = entry["task_id"].lower().replace("_", " ")
        if tid in query_lower or entry["task_id"].lower() in query_lower:
            specific.append(entry["task_id"])

    is_comparison = any(
        w in query_lower for w in ["compare", "comparison", "vs", "versus"]
    )
    try_run = any(
        w in query_lower for w in ["try", "quick", "sample", "debug", "test run"]
    )

    # Detect max tasks constraint (e.g. "at most 2 datasets", "use 3 benchmarks")
    import re

    max_tasks = None
    max_match = re.search(
        r"(?:at most|up to|max|only|just|use)\s+(\d+)\s+(?:dataset|benchmark|task)",
        query_lower,
    )
    if max_match:
        max_tasks = int(max_match.group(1))
    # Also match "N datasets" without a qualifier
    if max_tasks is None:
        max_match = re.search(r"(\d+)\s+(?:dataset|benchmark|task)", query_lower)
        if max_match:
            max_tasks = int(max_match.group(1))

    # Default to general evaluation if nothing matched
    if not capabilities and not specific:
        capabilities = ["vqa", "knowledge", "perception"]

    return EvalIntent(
        models=models,
        capabilities=capabilities,
        specific_benchmarks=specific,
        is_comparison=is_comparison,
        try_run=try_run,
        max_tasks=max_tasks,
    )


def parse_intent(
    query: str,
    catalog: TaskCatalog,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    model: str = "gpt-4o-mini",
) -> EvalIntent:
    """Parse intent using LLM with keyword fallback."""
    try:
        return parse_intent_with_llm(query, catalog, api_key, base_url, model)
    except Exception:
        return parse_intent_with_keywords(query, catalog)


def _dict_to_intent(data: dict) -> EvalIntent:
    models = []
    for m in data.get("models", []):
        if isinstance(m, str):
            models.append(ModelSpec(name=m))
        else:
            models.append(
                ModelSpec(
                    name=m.get("name", ""),
                    provider=m.get("provider"),
                    url=m.get("url"),
                    api_key_env=m.get("api_key_env"),
                )
            )
    return EvalIntent(
        models=models,
        capabilities=data.get("capabilities", []),
        specific_benchmarks=data.get("specific_benchmarks", []),
        is_comparison=data.get("is_comparison", False),
        try_run=data.get("try_run", False),
        max_tasks=data.get("max_tasks"),
        constraints=data.get("constraints", {}),
    )
