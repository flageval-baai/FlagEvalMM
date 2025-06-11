import os.path as osp
import json
from datasets import load_dataset
import tqdm
import os


def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split
    dataset = load_dataset(data_dir, split=split)
    name = cfg.get("dataset_name", "")
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    os.makedirs(osp.join(output_dir, "images"), exist_ok=True)
    content = []
    select_keys = ["question", "answer", "solution", "level", "subject"]
    for data in tqdm.tqdm(dataset):
        new_data = {
            "question_id": data["id"],
        }

        for key in select_keys:
            new_data[key] = data[key]
        if new_data["question"].count("<image") == 1:
            new_data["question"] = new_data["question"].replace("<image", "<image ")

        new_data["img_path"] = data["image"]
        if data["options"]:
            new_data["options"] = data["options"]
            new_data["question_type"] = "multiple-choice"
        else:
            new_data["question_type"] = "open"
        if not osp.exists(osp.join(output_dir, new_data["img_path"])):
            data["decoded_image"].save(osp.join(output_dir, new_data["img_path"]))
        content.append(new_data)
    with open(osp.join(output_dir, "data.json"), "w") as f:
        json.dump(content, f, indent=2)
