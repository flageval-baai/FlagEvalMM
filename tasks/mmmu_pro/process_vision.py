import json
import os
import os.path as osp
from datasets import load_dataset


def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split

    name = cfg.get("dataset_name", "")
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    content = []
    data = load_dataset(data_dir, name=name, split=split)
    os.makedirs(osp.join(output_dir, "img"), exist_ok=True)
    for annotation in data:
        info = {
            "question_id": annotation["id"],
            "answer": annotation["answer"],
            "subject": annotation["subject"],
            "question": "",
            "question_type": "vision",
        }
        image_path = f"img/{annotation['id']}.png"
        annotation["image"].save(osp.join(output_dir, image_path))
        info["img_path"] = image_path
        content.append(info)
    json.dump(content, open(osp.join(output_dir, "data.json"), "w"), indent=2)
