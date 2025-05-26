import os.path as osp
import json
from datasets import load_dataset
import tqdm
import os

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
            "img_path": "/home/vlm/train_images/" + item["image"],
            "question": item["question"],
            "answer": item["answer"],
            "question_type": "short-answer",
            "type_level_1": f"{item['image']}[{item['width']}x{item['height']}]",
            "type_level_2": "baai-vg"
        }
        processed_data.append(processed_item)
    
    # 保存处理后的数据
    with open(osp.join(output_dir, "data.json"), "w") as f:
        json.dump(processed_data, f, indent=4)