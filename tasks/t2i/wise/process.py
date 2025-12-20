import json
import os
import os.path as osp
import requests


def process(cfg):
    dataset_path, split = cfg.dataset_path, cfg.split
    if dataset_path.startswith("http"):
        data = json.loads(requests.get(dataset_path).text)
    else:
        data = json.load(open(dataset_path))
    output_dir = osp.join(cfg.processed_dataset_path, split)
    os.makedirs(output_dir, exist_ok=True)
    content = []
    for k, v in data.items():
        info = {
            "id": v["id"],
            "prompt": v["prompt"],
            "prompt_cn": v["prompt in Chinese"],
        }
        content.append(info)
    json.dump(
        content,
        open(osp.join(output_dir, "data.json"), "w"),
        indent=2,
        ensure_ascii=False,
    )
