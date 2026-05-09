import json
import os
import os.path as osp
import tarfile
from pathlib import Path

import pandas as pd
import tqdm
from huggingface_hub import snapshot_download


BENCHMARK_CSV = "DISE-bench/DISE-benchmark.csv"
MERGE_IMAGE_COLUMNS = [
    ("image", "merged full image"),
]
SEPARATE_IMAGE_COLUMNS = [
    ("question_image_path", "separate question image"),
    ("question_image_1_path", "separate question image 1"),
    ("question_image_2_path", "separate question image 2"),
    ("option_a_image_path", "separate option A image"),
    ("option_b_image_path", "separate option B image"),
    ("option_c_image_path", "separate option C image"),
    ("option_d_image_path", "separate option D image"),
]


def process(cfg):
    dataset_root = _dataset_root(cfg.dataset_path)
    image_mode = getattr(cfg, "image_mode", "merge")
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
        image_refs, row_missing = _image_refs(row, tar_index, image_mode)
        if row_missing:
            missing.extend(row_missing)
            continue
        if not image_refs:
            missing.append(f"image={row.get('image', '')}")
            continue

        img_paths = []
        for ref in image_refs:
            img_path = osp.join("images", ref["path"])
            _extract_image(ref["shard"], ref["path"], osp.join(output_dir, img_path))
            img_paths.append(img_path)

        content.append(
            {
                "question_id": f"benchmark_{row_id}",
                "question": _format_question(str(row["question"]).strip(), image_refs, image_mode),
                "question_type": "multiple-choice",
                "answer": str(row["answer"]).strip().upper(),
                "img_path": img_paths,
                "image_roles": [ref["role"] for ref in image_refs],
                "image_mode": image_mode,
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


def _image_refs(row, tar_index: dict, image_mode: str) -> tuple:
    refs = []
    missing = []
    seen = set()
    columns = SEPARATE_IMAGE_COLUMNS if image_mode == "separate" else MERGE_IMAGE_COLUMNS
    for column, role in columns:
        value = row.get(column, "")
        if pd.isna(value):
            continue
        value = str(value).strip()
        if not value:
            continue
        member = _csv_path_to_tar_member(value)
        if member in seen:
            continue
        shard = tar_index.get(member)
        if shard is None:
            missing.append(f"{column}={value}")
            continue
        refs.append({"role": role, "path": member, "shard": shard})
        seen.add(member)
    return refs, missing


def _format_question(question: str, image_refs: list, image_mode: str) -> str:
    if image_mode != "separate":
        return question

    image_tokens = " ".join(f"<image {idx + 1}>" for idx in range(len(image_refs)))
    image_order = "; ".join(
        f"<image {idx + 1}>: {ref['role']}" for idx, ref in enumerate(image_refs)
    )
    return (
        f"{image_tokens}\n"
        f"Images are provided as separate question/view/option images ({image_order}). "
        "Use all images together.\n"
        f"{question}"
    )


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
