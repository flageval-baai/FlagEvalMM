import json
from datasets import load_dataset
import os.path as osp
import os
import tqdm
import base64


def process(cfg):

    data_dir, split = cfg.dataset_path, cfg.split
    # 加载原始数据
    dataset = load_dataset(data_dir, split=split)
    name = cfg.get("dataset_name", "")
    # 设置输出路径
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    

    content = []
    select_keys = [
        "idx",
        "id",
        "question",
        "type",
        "task",
        "filename",
        "image",
        "prompt",
        "choices",
        "answer",
        "image_url",
    ]
    

    for data in tqdm.tqdm(dataset):

        if data.get('type') == 'Robustness':  # 不考虑robustness的题目
            continue

        new_data = {key: data[key] for key in select_keys}
        new_data["question_id"] = new_data.pop("idx")
        new_data["img_path"] = new_data.pop("filename")
        new_data["question_type"] = "multiple-choice"
        new_data["options"] = new_data.pop("choices")
        new_data["question"] = new_data.pop("prompt")

        if data["image"].mode != "RGB":
            data["image"] = data["image"].convert("RGB")
            
        img_path = osp.join(output_dir, new_data["filename"])
        if not osp.exists(osp.dirname(img_path)):
            os.makedirs(osp.dirname(img_path))
        data["image"].save(img_path)

        content.append(new_data)

    out_file = osp.join(output_dir, "data.json")
    with open(out_file, "w") as fout:
        json.dump(content, fout, indent=2, ensure_ascii=False)
