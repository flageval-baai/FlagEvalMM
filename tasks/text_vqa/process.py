import os.path as osp
import json
from datasets import load_dataset
import tqdm
import os


def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split
    dataset = load_dataset(data_dir, split=split, streaming=True)
    name = cfg.get("dataset_name", "")
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    os.makedirs(osp.join(output_dir, "images"), exist_ok=True)
    content = []
    select_keys = ["question", "question_id", "image_classes"]
    for data in tqdm.tqdm(dataset):
        new_data = {}
        for key in select_keys:
            new_data[key] = data[key]
        new_data["img_path"] = osp.join("images", f"{data['image_id']}.png")
        new_data["question_type"] = "short-answer"
        new_data["answer"] = data["answers"]
        content.append(new_data)
        data["image"].convert("RGB").save(osp.join(output_dir, new_data["img_path"]))
    with open(osp.join(output_dir, "data.json"), "w") as f:
        json.dump(content, f)
