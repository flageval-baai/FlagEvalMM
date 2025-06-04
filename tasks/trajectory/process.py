
import re
import os
import tqdm
import json
import os.path as osp
from datasets import load_dataset
from PIL import Image

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
    # dataset = load_dataset(data_dir, split=split)
    # with open(f"{data_dir}/roboos_affordance_test_data.json", "r") as f:
    #     affordance = json.load(f)
    
    # with open(f"{data_dir}/roboos_pointing_test_data.json", "r") as f:
    #     pointing = json.load(f)    

    with open(f"{data_dir}/traj_data_test.json", "r") as f:
        data = json.load(f)  
    
    # 设置输出路径
    output_dir = osp.join(cfg.processed_dataset_path, cfg.split, cfg.get("dataset_name", ""))
    os.makedirs(output_dir, exist_ok=True)
    # 处理数据
    # data = {"affordance": affordance, "pointing": pointing, "trajectory": trajectory}
    

    processed_data = []
    for i, item in enumerate(data):
        conversations = item["conversations"][:2]
        question, answer = "", ""
        for conv in conversations:
            if conv["from"] == "human":
                question = conv["value"]
            if conv["from"] == "gpt":
                answer = conv["value"]
        if  not question or not answer:
            continue
        
        image_path = osp.join("/share/project/eval_images", item["image"][0])
        
        # 读取图片
        image = Image.open(image_path)

        # 分别获取宽度和高度
        width, height = image.size
        
        processed_item = {
            "question_id": item["id"],
            "img_path": image_path,
            "question": question.replace("Please predict up to 10 key trajectory points to complete the task.", "Please predict up to 10 key trajectory points to complete the task. Your answer should be formatted as a list of tuples, i.e. [[x1, y1], [x2, y2], ...], where each tuple contains the x and y coordinates of a point.").replace("<image>", ''),
            "answer": answer,
            "question_type": "short-answer",
            "width": width,
            "height": height
        }
        processed_data.append(processed_item)

    with open(osp.join(output_dir, f"data.json"), "w") as f:
        json.dump(processed_data, f, indent=4)
        
