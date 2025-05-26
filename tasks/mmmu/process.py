import json
import os
import os.path as osp
import ast
from datasets import load_dataset


def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split

    name = cfg.get("dataset_name", "")
    print(data_dir, split, name)
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    content = []
    data = load_dataset(data_dir, name=name, split=split)
    os.makedirs(osp.join(output_dir, "img"), exist_ok=True)
    fileds = [
        "id",
        "question",
        "explanation",
        "answer",
        "topic_difficulty",
    ]
    for annotation in data:
        info = {}
        for field in fileds:
            info[field] = annotation[field]
        info["question_id"] = info.pop("id")
        options = ast.literal_eval(annotation["options"])
        # flatten options
        if len(options) > 0 and isinstance(options[0], list):
            options = options[0]
        info["options"] = options

        img_type = ast.literal_eval(annotation["img_type"])
        info["img_type"] = img_type
        info["question_type"] = annotation.get("question_type", "multiple-choice")
        if "subfield" in annotation:
            info["subfield"] = annotation["subfield"]
            info["subject"] = info["question_id"].split("_")[1]
        elif "subject" in annotation:
            info["subject"] = annotation["subject"]
        image_path = []
        for i in range(1, 8):
            if annotation[f"image_{i}"] is None:
                break
            img_name = f"img/{annotation['id']}_{i}.png"
            annotation[f"image_{i}"].save(osp.join(output_dir, img_name))
            question = info["question"]
            if "options" in info:
                question += "\nOptions:"
                question += "".join(info["options"])
            if f"<image {i}>" not in question:
                print(f"<image {i}> not in {question}")
            else:
                image_path.append(img_name)
        info["img_path"] = image_path
        content.append(info)

    json.dump(content, open(osp.join(output_dir, "data.json"), "w"), indent=2)
