import json
import os
import os.path as osp
from datasets import load_dataset
import tqdm
import re


def replace_images(input_string, id_mp):
    pattern = r'<img="([^"]+)">'

    def replace(match):
        replacement = f"<image {id_mp[match.group(1)]}>"
        return replacement

    # get the pattern from the input_string
    img_name = re.findall(pattern, input_string)
    s = len(id_mp)
    for p in img_name:
        if p not in id_mp:
            s += 1
            id_mp[p] = s

    output_string = re.sub(pattern, replace, input_string)
    return output_string, id_mp


def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    os.makedirs(osp.join(output_dir, "image"), exist_ok=True)
    dataset = load_dataset(data_dir, name, split=split)

    content = []
    option_num = 4
    type_map = {
        "选择": "multiple-choice",
        "填空": "fill-in-the-blank",
        "判断": "yes-no",
    }
    fileds = [
        "id",
        "type",
        "source_type",
        "source",
        "question",
        "answer",
        "analysis",
        "distribution",
        "difficulty_level",
        "subcategory",
        "category",
        "subfield",
        "img_type",
    ]
    max_image_num = 5
    for row in tqdm.tqdm(dataset):
        info = {}
        for filed in fileds:
            info[filed] = row[filed]
        info["question"], id_mp = replace_images(row["question"], {})
        info["question_id"] = info.pop("id")
        options = []
        for i in range(option_num):
            if row[f"option{i + 1}"] != "-":
                new_option, id_mp = replace_images(row[f"option{i + 1}"], id_mp)
                options.append(new_option)
        info["question_type"] = type_map[info.pop("type")]
        info["options"] = options
        image_path = []
        for i in range(max_image_num):
            img = row[f"image_{i + 1}"]
            if img is None:
                break
            if img.mode != "RGB":
                img = img.convert("RGB")
            img_name = row[f"image_{i+1}_filename"]
            file_name = f"image/{img_name}"
            # img.save(osp.join(output_dir, file_name))
            image_path.append(file_name)
        info["img_path"] = image_path
        content.append(info)
    json.dump(
        content,
        open(osp.join(output_dir, "data.json"), "w"),
        indent=2,
        ensure_ascii=False,
    )
