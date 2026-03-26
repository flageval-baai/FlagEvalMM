import os
import os.path as osp
import json
from datasets import load_dataset


def process(cfg):
    split = cfg.split
    dataset = load_dataset(cfg.dataset_path, split=split)
    output_dir = osp.join(cfg.processed_dataset_path, split)
    img_dir = osp.join(output_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    content = []
    for annotation in dataset:
        item = {
            "question_id": annotation["question_id"],
            "question": annotation["question"],
            "question_type": "open",
            "image_type": annotation["image_type"],
            "design": annotation["design"],
            "evaluator": annotation["evaluator"],
            "evaluator_kwargs": json.loads(annotation["evaluator_kwargs"]),
        }
        img_path = osp.join("img", f"{item['question_id']}.png")
        item["img_path"] = img_path
        annotation["image"].save(osp.join(output_dir, img_path))
        content.append(item)

    with open(osp.join(output_dir, "data.json"), "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)
