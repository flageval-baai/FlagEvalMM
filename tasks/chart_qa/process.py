import json
from datasets import load_dataset
import os.path as osp
import os
import tqdm


def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split
    dataset = load_dataset(data_dir, split=split)
    name = cfg.get("dataset_name", "")
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    content = []
    os.makedirs(osp.join(output_dir, "images"), exist_ok=True)
    select_keys = ["question", "answer"]
    for i, data in tqdm.tqdm(enumerate(dataset)):
        new_data = {key: data[key] for key in select_keys}
        new_data["question_id"] = str(i)
        new_data["ori_type"] = data["type"]
        new_data["question_type"] = "short-answer"
        new_data["img_path"] = osp.join("images", f"{i}.png")
        data["image"].save(osp.join(output_dir, new_data["img_path"]))
        content.append(new_data)
    json.dump(
        content,
        open(osp.join(output_dir, "data.json"), "w"),
        indent=2,
        ensure_ascii=False,
    )
