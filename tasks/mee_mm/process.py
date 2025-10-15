"""
MEE-MM 数据处理脚本
用于处理多种考试数据，转换为FlagEvalMM标准格式
支持处理高考、国考等各种考试数据格式
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


def detect_data_format(data_sample: dict) -> str:
    """
    检测数据格式类型
    """
    # 检查是否为高考格式
    if "所用到的图片" in data_sample and any(key in data_sample for key in ["题干", "选项A"]):
        # 进一步区分高考和国考
        if "卷子" in data_sample or "年份" in data_sample:
            return "gaokao_guokao"  # 国考格式（包含更多字段）
        else:
            return "gaokao"  # 高考格式
    
    # 检查是否为标准Hugging Face格式
    elif "question" in data_sample or "text" in data_sample:
        return "huggingface"
    
    return "unknown"


def build_question_text(item: dict, data_format: str) -> str:
    """
    根据数据格式构建问题文本
    """
    question_parts = []
    
    if data_format in ["gaokao", "gaokao_guokao"]:
        # 添加题目描述（国考特有）
        if data_format == "gaokao_guokao":
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
                option_letter = option_key[-1]
                options.append(f"{option_letter}. {option_value}")
        
        if options:
            question_parts.append("\n".join(options))
    
    elif data_format == "huggingface":
        # 标准格式直接获取问题
        question_parts.append(item.get('question', item.get('text', '')))
    
    return "\n\n".join(question_parts) if question_parts else ""


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
    
    print(f"开始处理MEE-MM数据集...")
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
        raise ValueError(f"无法加载数据集: {cfg.dataset_path}")
    
    # 处理数据
    processed_data = []
    question_id_counter = 0
    
    # 处理列表格式数据
    if isinstance(raw_data, list) and len(raw_data) > 0:
        # 检测数据格式
        data_format = detect_data_format(raw_data[0])
        print(f"检测到数据格式: {data_format}")
        
        for i, item in enumerate(tqdm.tqdm(raw_data, desc="处理数据")):
            question_id_counter += 1
            
            # 处理图片信息
            image_paths = []
            has_img = False
            
            if data_format in ["gaokao", "gaokao_guokao"]:
                # 高考/国考格式
                image_list_str = item.get("所用到的图片", "")
                image_paths = parse_image_paths(image_list_str)
                has_img = has_images(image_list_str)
            elif data_format == "huggingface":
                # Hugging Face格式
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
                has_img = len(image_paths) > 0
            
            # 构建问题文本
            question_text = build_question_text(item, data_format)
            if not question_text:
                question_text = f"题目{i+1}"
            
            # 获取答案
            if data_format in ["gaokao", "gaokao_guokao"]:
                answer = item.get("正确选项", item.get("answer", item.get("答案", "")))
            else:
                answer = item.get('answer', item.get('label', ""))
            
            # 确定问题类型
            if data_format == "gaokao_guokao":
                exam_type = item.get("卷子", "考试")
                year = item.get("年份", "")
                question_type = f"{exam_type}_{year}" if year else exam_type
            elif data_format == "gaokao":
                question_type = item.get("question_type", item.get("科目", "未知"))
            else:
                question_type = item.get('subject', item.get('category', item.get('question_type', '通用')))
            
            processed_item = {
                "question_id": f"mee_{question_id_counter}",
                "img_path": image_paths,
                "question": question_text,
                "question_type": question_type,
                "answer": answer,
                "has_image": has_img,
                "image_count": len(image_paths),
                "data_format": data_format
            }
            
            # 添加额外信息
            if data_format in ["gaokao", "gaokao_guokao"]:
                processed_item["raw_image_info"] = item.get("所用到的图片", "")
            
            if data_format == "gaokao_guokao":
                processed_item["exam_info"] = {
                    "exam_type": item.get("卷子", ""),
                    "year": item.get("年份", ""),
                    "description": item.get("题目描述", ""),
                    "material": item.get("材料", "")
                }
            
            processed_data.append(processed_item)
    
    # 处理字典格式数据
    elif isinstance(raw_data, dict):
        print("处理按类型分类的数据...")
        for category, questions in raw_data.items():
            if not isinstance(questions, list):
                continue
                
            print(f"处理类别: {category}, 题目数量: {len(questions)}")
            
            # 检测数据格式
            if questions:
                data_format = detect_data_format(questions[0])
                print(f"  数据格式: {data_format}")
            
            for i, question_item in enumerate(tqdm.tqdm(questions, desc=f"处理{category}题目")):
                question_id_counter += 1
                
                # 处理图片信息
                image_paths = []
                has_img = False
                
                if data_format in ["gaokao", "gaokao_guokao"]:
                    image_list_str = question_item.get("所用到的图片", "")
                    image_paths = parse_image_paths(image_list_str)
                    has_img = has_images(image_list_str)
                
                # 构建问题文本
                question_text = build_question_text(question_item, data_format)
                if not question_text:
                    question_text = f"{category}第{i+1}题"
                
                # 获取答案
                answer = question_item.get("正确选项", question_item.get("answer", question_item.get("答案", "")))
                
                processed_item = {
                    "question_id": f"{category}_{i+1}",
                    "img_path": image_paths,
                    "question": question_text,
                    "question_type": category,
                    "answer": answer,
                    "has_image": has_img,
                    "image_count": len(image_paths),
                    "data_format": data_format
                }
                
                if data_format in ["gaokao", "gaokao_guokao"]:
                    processed_item["raw_image_info"] = question_item.get("所用到的图片", "")
                
                processed_data.append(processed_item)
    
    else:
        raise ValueError("不支持的数据格式")
    
    print(f"数据处理完成，共处理 {len(processed_data)} 个问题")
    
    # 统计信息
    stats = {
        "total_questions": len(processed_data),
        "with_images": sum(1 for item in processed_data if item["has_image"]),
        "without_images": sum(1 for item in processed_data if not item["has_image"]),
        "question_types": {},
        "data_formats": {}
    }
    
    # 按类型和格式统计
    for item in processed_data:
        q_type = item["question_type"]
        data_fmt = item.get("data_format", "unknown")
        
        if q_type not in stats["question_types"]:
            stats["question_types"][q_type] = {"total": 0, "with_images": 0}
        stats["question_types"][q_type]["total"] += 1
        if item["has_image"]:
            stats["question_types"][q_type]["with_images"] += 1
        
        if data_fmt not in stats["data_formats"]:
            stats["data_formats"][data_fmt] = 0
        stats["data_formats"][data_fmt] += 1
    
    print("\n数据统计:")
    print(f"总题目数: {stats['total_questions']}")
    print(f"有图片题目: {stats['with_images']}")
    print(f"无图片题目: {stats['without_images']}")
    print("\n各类型统计:")
    for q_type, count_info in stats["question_types"].items():
        print(f"  {q_type}: {count_info['total']} 题 (其中 {count_info['with_images']} 题有图片)")
    print("\n数据格式分布:")
    for data_fmt, count in stats["data_formats"].items():
        print(f"  {data_fmt}: {count} 题")
    
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
