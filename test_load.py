#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
import re
import glob
from pathlib import Path

# 训练数据目录
DATA_DIR = "/share/project/xuyijie/agibot/training_data_2"

def count_steps(response_text):
    """计算GPT回答中的实际步骤数量"""
    # 特殊情况处理："Task completed."
    if "Task completed" in response_text:
        return 0
        
    # 正则匹配数字项
    steps = re.findall(r'^\d+\.', response_text, re.MULTILINE)
    return len(steps)

def update_json_file(file_path):
    """更新单个JSON文件中的步骤数量描述"""
    print(f"处理文件: {file_path}")
    
    # 读取JSON文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 跟踪更改数量
    changes_count = 0
    
    # 处理每个对话
    for item in data:
        if "conversations" not in item:
            continue
            
        conversations = item["conversations"]
        if len(conversations) < 2:
            continue
            
        # 确保第一个是human，第二个是gpt
        if conversations[0]["from"] == "human" and conversations[1]["from"] == "gpt":
            human_message = conversations[0]["value"]
            gpt_response = conversations[1]["value"]
            
            # 检查是否包含"what are the next 5 steps?"
            if "what are the next 5 steps?" in human_message:
                # 计算GPT回答中的实际步骤数
                steps_count = count_steps(gpt_response)
                
                if steps_count == 0:
                    # Task completed 不需要steps
                    new_text = human_message.replace("what are the next 5 steps?", "what should I do next?")
                else:
                    # 替换步骤数量
                    step_text = "step" if steps_count == 1 else "steps"
                    new_text = human_message.replace("what are the next 5 steps?", f"what are the next {steps_count} {step_text}?")
                
                # 如果有变化，更新文本
                if new_text != human_message:
                    conversations[0]["value"] = new_text
                    changes_count += 1
    
    # 如果有更改，保存文件
    if changes_count > 0:
        print(f"  更新了 {changes_count} 个对话")
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
    else:
        print("  没有需要更新的内容")

def main():
    """处理所有训练数据JSON文件"""
    # 获取所有JSON文件路径
    json_files = glob.glob(os.path.join(DATA_DIR, "task_*.json"))
    
    if not json_files:
        print(f"在 {DATA_DIR} 中未找到任何JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    # 处理每个文件
    for file_path in json_files:
        update_json_file(file_path)
    
    print("所有文件处理完成!")

if __name__ == "__main__":
    main() 