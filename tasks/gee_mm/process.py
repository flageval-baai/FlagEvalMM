"""
GEE-MM 数据处理脚本
用于处理国考最终汇总.json数据，转换为FlagEvalMM标准格式
"""

import json
import os
import os.path as osp
import tqdm
from datasets import load_dataset


def has_images(image_list_str: str) -> bool:
    """
    检查是否有图片
    """
    if not image_list_str or image_list_str.strip() == "":
        return False
    
    try:
        # 尝试解析图片列表字符串
        if image_list_str.startswith('[') and image_list_str.endswith(']'):
            image_list = eval(image_list_str)
            return len(image_list) > 0
        return False
    except:
        return False


def parse_image_paths(image_list_str: str) -> list:
    """
    解析图片路径列表
    """
    if not image_list_str or image_list_str.strip() == "":
        return []
    
    try:
        if image_list_str.startswith('[') and image_list_str.endswith(']'):
            image_list = eval(image_list_str)
            return image_list if isinstance(image_list, list) else []
        return []
    except:
        return []


def process(cfg):
    """
    处理原始数据集
    
    处理后的数据必须包含以下字段：
    - question_id: 问题唯一标识
    - img_path: 图片路径（可以是列表）
    - question: 问题文本
    - question_type: 问题类型
    - answer: 正确答案（用于评估）
    - has_image: 是否包含图片
    """
    
    # 设置输出路径
    output_dir = osp.join(cfg.processed_dataset_path, cfg.split, cfg.get("dataset_name", ""))
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始处理GEE-MM数据集...")
    print(f"数据集路径: {cfg.dataset_path}")
    print(f"输出目录: {output_dir}")
    
    try:
        # 尝试从Hugging Face加载数据
        if "/" in cfg.dataset_path and not osp.exists(cfg.dataset_path):
            print("从Hugging Face加载数据...")
            dataset = load_dataset(cfg.dataset_path, split=cfg.split)
            raw_data = list(dataset)
        else:
            # 本地文件加载
            print("从本地文件加载数据...")
            if cfg.dataset_path.endswith('.json'):
                with open(cfg.dataset_path, 'r', encoding='utf-8') as f:
                    raw_data = json.load(f)
            else:
                raise ValueError(f"不支持的文件格式: {cfg.dataset_path}")
        
        print(f"原始数据加载完成")
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        # 如果Hugging Face加载失败，尝试加载本地的国考最终汇总.json
        local_json_path = "国考最终汇总.json"
        if osp.exists(local_json_path):
            print(f"尝试加载本地文件: {local_json_path}")
            with open(local_json_path, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        else:
            raise ValueError(f"无法加载数据集: {cfg.dataset_path}")
    
    # 处理数据
    processed_data = []
    question_id_counter = 0
    
    # 处理从Hugging Face加载的标准格式数据
    if isinstance(raw_data, list) and len(raw_data) > 0 and isinstance(raw_data[0], dict):
        # 检查是否是标准的Hugging Face数据格式
        sample_item = raw_data[0]
        if 'question' in sample_item or 'text' in sample_item:
            print("处理Hugging Face标准格式数据...")
            for i, item in enumerate(tqdm.tqdm(raw_data, desc="处理数据")):
                question_id_counter += 1
                
                # 获取图片路径
                image_paths = []
                if 'image' in item:
                    if isinstance(item['image'], list):
                        image_paths = item['image']
                    elif item['image']:
                        image_paths = [item['image']]
                elif 'img_path' in item:
                    if isinstance(item['img_path'], list):
                        image_paths = item['img_path']
                    elif item['img_path']:
                        image_paths = [item['img_path']]
                
                # 获取问题文本
                question = item.get('question', item.get('text', f"题目{i+1}"))
                
                # 获取答案
                answer = item.get('answer', item.get('label', ""))
                
                # 获取科目类型
                question_type = item.get('subject', item.get('category', item.get('question_type', '国考')))
                
                processed_item = {
                    "question_id": f"question_{question_id_counter}",
                    "img_path": image_paths,
                    "question": question,
                    "question_type": question_type,
                    "answer": answer,
                    "has_image": len(image_paths) > 0,
                    "image_count": len(image_paths)
                }
                
                processed_data.append(processed_item)
        else:
            print("处理国考格式数据...")
            # 按照国考最终汇总.json的格式处理
            for i, item in enumerate(tqdm.tqdm(raw_data, desc="处理数据")):
                question_id_counter += 1
                
                # 解析图片信息
                image_list_str = item.get("所用到的图片", "")
                image_paths = parse_image_paths(image_list_str)
                has_img = has_images(image_list_str)
                
                # 组合问题文本：题目描述 + 材料 + 题干 + 选项
                question_parts = []
                
                # 添加题目描述（如果有）
                description = item.get("题目描述", "")
                if description:
                    question_parts.append(description)
                
                # 添加材料（如果有）
                material = item.get("材料", "")
                if material:
                    question_parts.append(f"材料：{material}")
                
                # 添加题干
                question_stem = item.get("题干", "")
                if question_stem:
                    question_parts.append(question_stem)
                
                # 添加选项
                options = []
                for option_key in ["选项A", "选项B", "选项C", "选项D"]:
                    option_value = item.get(option_key, "")
                    if option_value:
                        option_letter = option_key[-1]  # 获取A、B、C、D
                        options.append(f"{option_letter}. {option_value}")
                
                if options:
                    question_parts.append("\n".join(options))
                
                # 组合完整问题
                full_question = "\n\n".join(question_parts) if question_parts else f"国考第{i+1}题"
                
                # 获取正确答案
                correct_answer = item.get("正确选项", item.get("answer", item.get("答案", "")))
                
                # 确定问题类型
                exam_type = item.get("卷子", "国考")
                year = item.get("年份", "")
                question_type = f"{exam_type}_{year}" if year else exam_type
                
                processed_item = {
                    "question_id": f"gee_{question_id_counter}",
                    "img_path": image_paths,
                    "question": full_question,
                    "question_type": question_type,
                    "answer": correct_answer,
                    "has_image": has_img,
                    "image_count": len(image_paths),
                    "raw_image_info": image_list_str,
                    "exam_info": {
                        "exam_type": exam_type,
                        "year": year,
                        "description": description,
                        "material": material
                    }
                }
                
                processed_data.append(processed_item)
    
    # 处理按类型分类的字典格式数据
    elif isinstance(raw_data, dict):
        print("处理按类型分类的数据...")
        for category, questions in raw_data.items():
            if not isinstance(questions, list):
                continue
                
            print(f"处理类别: {category}, 题目数量: {len(questions)}")
            
            for i, question_item in enumerate(tqdm.tqdm(questions, desc=f"处理{category}题目")):
                question_id_counter += 1
                
                # 解析图片信息
                image_list_str = question_item.get("所用到的图片", "")
                image_paths = parse_image_paths(image_list_str)
                has_img = has_images(image_list_str)
                
                # 组合问题文本
                question_parts = []
                
                # 添加题目描述
                description = question_item.get("题目描述", "")
                if description:
                    question_parts.append(description)
                
                # 添加材料
                material = question_item.get("材料", "")
                if material:
                    question_parts.append(f"材料：{material}")
                
                # 添加题干
                question_stem = question_item.get("题干", "")
                if question_stem:
                    question_parts.append(question_stem)
                
                # 添加选项
                options = []
                for option_key in ["选项A", "选项B", "选项C", "选项D"]:
                    option_value = question_item.get(option_key, "")
                    if option_value:
                        option_letter = option_key[-1]
                        options.append(f"{option_letter}. {option_value}")
                
                if options:
                    question_parts.append("\n".join(options))
                
                # 组合完整问题
                full_question = "\n\n".join(question_parts) if question_parts else f"{category}第{i+1}题"
                
                # 获取正确答案
                correct_answer = question_item.get("正确选项", "")
                
                processed_item = {
                    "question_id": f"{category}_{i+1}",
                    "img_path": image_paths,
                    "question": full_question,
                    "question_type": category,
                    "answer": correct_answer,
                    "has_image": has_img,
                    "image_count": len(image_paths),
                    "raw_image_info": image_list_str
                }
                
                processed_data.append(processed_item)
    
    else:
        raise ValueError("不支持的数据格式")
    
    print(f"数据处理完成，共处理 {len(processed_data)} 个问题")
    
    # 统计信息
    stats = {
        "total_questions": len(processed_data),
        "with_images": sum(1 for item in processed_data if item["has_image"]),
        "without_images": sum(1 for item in processed_data if not item["has_image"]),
        "question_types": {}
    }
    
    # 按类型统计
    for item in processed_data:
        q_type = item["question_type"]
        if q_type not in stats["question_types"]:
            stats["question_types"][q_type] = {"total": 0, "with_images": 0}
        stats["question_types"][q_type]["total"] += 1
        if item["has_image"]:
            stats["question_types"][q_type]["with_images"] += 1
    
    print("\n数据统计:")
    print(f"总题目数: {stats['total_questions']}")
    print(f"有图片题目: {stats['with_images']}")
    print(f"无图片题目: {stats['without_images']}")
    print("\n各类型统计:")
    for q_type, count_info in stats["question_types"].items():
        print(f"  {q_type}: {count_info['total']} 题 (其中 {count_info['with_images']} 题有图片)")
    
    # 保存处理后的数据
    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w", encoding='utf-8') as f:
        json.dump(processed_data, f, ensure_ascii=False, indent=2)
    
    # 保存统计信息
    stats_file = osp.join(output_dir, "statistics.json")
    with open(stats_file, "w", encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"\n数据处理完成！")
    print(f"处理后的数据保存在: {output_file}")
    print(f"统计信息保存在: {stats_file}")
