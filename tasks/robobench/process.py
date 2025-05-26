import re
import os
import tqdm
import json
import os.path as osp



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
    print(data_dir)
    # dataset = load_dataset(data_dir, split=split)
    with open(data_dir, "r") as f:
        dataset = json.load(f)
        
    # 设置输出路径
    output_dir = osp.join(cfg.processed_dataset_path, cfg.split, cfg.get("dataset_name", ""))
    os.makedirs(output_dir, exist_ok=True)
    # 处理数据
    processed_data = []
    for i, item in enumerate(tqdm.tqdm(dataset)):
        response = item["response"]
        steps = [step["step_description"]  for step in response["steps"]]
        processed_item = {
            "question_id": i,
            "img_path": item["id"].replace("/mnt/hpfs/baaiei", "/home") + "/frame_0.png",
            "question": response["task_summary"],
            "answer":  steps,
            "question_type": 'short-answer',
        }
        processed_data.append(processed_item)
    print(output_dir)
    with open(osp.join(output_dir, "data.json"), "w") as f:
        json.dump(processed_data, f, indent=4)
        
    
    