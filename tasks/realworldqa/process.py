import json
import os
import os.path as osp

from datasets import load_dataset


def process(cfg):
    """Process the dataset and save it in a standard format"""
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")

    # build output path
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    img_dir = osp.join(output_dir, "img")
    os.makedirs(img_dir, exist_ok=True)
    # load dataset
    data = load_dataset(data_dir, name=name, split=split)

    content = []
    for index, annotation in enumerate(data):
        img_path = osp.join("img", annotation["image_path"])
        info = {
            "question": annotation["question"],
            "question_id": index,
            "question_type": "open",
            "img_path": img_path,
            "answer": annotation["answer"],
        }
        annotation["image"].save(osp.join(img_dir, annotation["image_path"]))
        content.append(info)
    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(content)} items. Data saved to {output_file}")
