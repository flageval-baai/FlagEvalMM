import json
import os.path as osp
from typing import Optional


class TaskCatalog:
    def __init__(self, catalog_path: Optional[str] = None):
        if catalog_path is None:
            catalog_path = osp.join(osp.dirname(__file__), "task_catalog.json")
        with open(catalog_path) as f:
            data = json.load(f)
        self.entries = (
            data["tasks"] if isinstance(data, dict) and "tasks" in data else data
        )
        self._by_id = {e["task_id"]: e for e in self.entries}

    def search_by_capability(self, capabilities: list[str]) -> list[dict]:
        caps = set(capabilities)
        return [e for e in self.entries if caps & set(e["capabilities"])]

    def search_by_keywords(self, query: str) -> list[dict]:
        query_lower = query.lower()
        tokens = query_lower.split()
        results = []
        for entry in self.entries:
            searchable = " ".join(
                [entry["task_id"], entry["display_name"], entry["description"]]
                + entry["keywords"]
            ).lower()
            if any(t in searchable for t in tokens):
                results.append(entry)
        return results

    def search_by_benchmark_names(self, names: list[str]) -> list[dict]:
        results = []
        for name in names:
            name_lower = name.lower().replace("-", "_").replace(" ", "_")
            for entry in self.entries:
                entry_id = entry["task_id"].lower()
                entry_keywords = [k.lower() for k in entry["keywords"]]
                if (
                    name_lower in entry_id
                    or entry_id in name_lower
                    or name_lower in entry_keywords
                ):
                    if entry not in results:
                        results.append(entry)
        return results

    def get_by_id(self, task_id: str):  # type: ignore
        return self._by_id.get(task_id)

    def all_task_ids(self) -> list[str]:
        return [e["task_id"] for e in self.entries]

    def all_capabilities(self) -> list[str]:
        caps = set()
        for e in self.entries:
            caps.update(e["capabilities"])
        return sorted(caps)
