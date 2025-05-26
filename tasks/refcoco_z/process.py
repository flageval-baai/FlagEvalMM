
import re
import os
import tqdm
import json
import os.path as osp
from datasets import load_dataset


def remove_leading_articles(text):
    # 使用正则表达式去除开头的 'a', 'an', 'the'，并忽略大小写
    result = re.sub(r"^(a|an|the)\s+", "", text, flags=re.IGNORECASE)
    return result


def process(cfg):
    """处理原始数据集
    
    处理后的数据必须包含以下字段：
    - question_id: 问题唯一标识
    - img_path: 图片路径
    - question: 问题文本
    - question_type: 问题类型
    """
    # 加载原始数据
    data_dir, split = cfg.dataset_path, cfg.split
    dataset = load_dataset(data_dir, split=split)
    # 设置输出路径
    output_dir = osp.join(cfg.processed_dataset_path, cfg.split, cfg.get("dataset_name", ""))
    os.makedirs(output_dir, exist_ok=True)
    # 处理数据
    processed_data = []
    for i, item in enumerate(tqdm.tqdm(dataset)):
        processed_item = {
            "question_id": item["id"],
            "img_path": osp.join("/share/project/eval_images/ref-l4", item["file_name"]),
            "question": item["caption"],
            "answer": item["bbox"],
            'height': item['height'],
            'width': item['width'],
            "question_type": "short-answer",
            "type_level_1": '_'.join(item["file_name"].split('_')[:2]) + "/" + item['split'] + "/" + f"{item['height']}_{item['width']}",
            "type_level_2": "ref-l4/" + item['file_name']
        }
        processed_data.append(processed_item)

    with open(osp.join(output_dir, "data.json"), "w") as f:
        json.dump(processed_data, f, indent=4)
        
