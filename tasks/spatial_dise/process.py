import json
import os
import os.path as osp
import tarfile
from pathlib import Path

import pandas as pd
import tqdm
from huggingface_hub import snapshot_download


BENCHMARK_CSV = "DISE-bench/DISE-benchmark.csv"


def process(cfg):
    dataset_root = _dataset_root(cfg.dataset_path)
    output_dir = osp.join(cfg.processed_dataset_path, cfg.split)
    image_output_dir = osp.join(output_dir, "images")
    os.makedirs(image_output_dir, exist_ok=True)

    csv_path = osp.join(dataset_root, BENCHMARK_CSV)
    if not osp.isfile(csv_path):
        raise FileNotFoundError(f"Spatial-DISE benchmark CSV not found: {csv_path}")

    raw = pd.read_csv(csv_path, skipinitialspace=True)
    raw.columns = [str(col).strip() for col in raw.columns]
    for col in raw.columns:
        if raw[col].dtype == object:
            raw[col] = raw[col].map(lambda x: x.strip() if isinstance(x, str) else x)

    tar_index = _build_tar_index(dataset_root)
    content = []
    missing = []
    for row_id, row in tqdm.tqdm(raw.iterrows(), total=len(raw)):
        member = _csv_path_to_tar_member(row["image"])
        shard = tar_index.get(member)
        if shard is None:
            missing.append(str(row["image"]))
            continue

        img_path = osp.join("images", member)
        _extract_image(shard, member, osp.join(output_dir, img_path))
        content.append(
            {
                "question_id": f"benchmark_{row_id}",
                "question": str(row["question"]).strip(),
                "question_type": "multiple-choice",
                "answer": str(row["answer"]).strip().upper(),
                "img_path": img_path,
                "category": str(row.get("category", "")).strip(),
                "difficulty": str(row.get("difficulty", "")).strip(),
                "source": str(row.get("source", "")).strip(),
                "dise_category": str(row.get("dise_category", "")).strip(),
            }
        )

    if missing:
        examples = ", ".join(missing[:5])
        raise FileNotFoundError(
            f"{len(missing)} Spatial-DISE image references were not found in tar shards. "
            f"Examples: {examples}"
        )

    with open(osp.join(output_dir, "data.json"), "w") as fout:
        json.dump(content, fout, indent=2)


def _dataset_root(repo_id: str) -> str:
    local_root = os.environ.get("SPATIAL_DISE_ROOT")
    if local_root:
        local_root = osp.expanduser(osp.expandvars(local_root))
        if osp.isdir(local_root):
            return local_root

    return snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision="main",
        allow_patterns=[BENCHMARK_CSV, "image/*.tar"],
    )


def _csv_path_to_tar_member(path: str) -> str:
    path = str(path).strip()
    if path.startswith("images/"):
        path = path[len("images/") :]
    return path.lstrip("/\\")


def _build_tar_index(dataset_root: str) -> dict:
    image_dir = osp.join(dataset_root, "image")
    tar_paths = sorted(Path(image_dir).glob("*.tar"))
    if not tar_paths:
        raise FileNotFoundError(f"No Spatial-DISE tar shards found under {image_dir}")

    tar_index = {}
    for tar_path in tar_paths:
        with tarfile.open(tar_path) as tf:
            for member in tf.getmembers():
                if member.isfile():
                    tar_index[member.name] = str(tar_path)
    return tar_index


def _extract_image(shard: str, member: str, target: str) -> None:
    if osp.exists(target):
        return

    os.makedirs(osp.dirname(target), exist_ok=True)
    with tarfile.open(shard) as tf:
        image_file = tf.extractfile(member)
        if image_file is None:
            raise FileNotFoundError(f"{member} not found in {shard}")
        with open(target, "wb") as fout:
            fout.write(image_file.read())
