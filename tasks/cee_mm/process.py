"""
CEE-MM 数据处理脚本
用于处理高考最终汇总.json数据，转换为FlagEvalMM标准格式
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
    - question_type: 问题类型（科目）
    - answer: 正确答案（用于评估）
    - has_image: 是否包含图片
    """
    
    # 设置输出路径
    output_dir = osp.join(cfg.processed_dataset_path, cfg.split, cfg.get("dataset_name", ""))
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"开始处理CEE-MM数据集...")
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
        # 如果Hugging Face加载失败，尝试加载本地的高考最终汇总.json
        local_json_path = "高考最终汇总.json"
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
                question_type = item.get('subject', item.get('category', item.get('question_type', '未知')))
                
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
            print("未识别的数据格式，尝试处理为高考格式...")
            # 按照高考最终汇总.json的格式处理
            for i, item in enumerate(tqdm.tqdm(raw_data, desc="处理数据")):
                question_id_counter += 1
                
                # 解析图片信息
                image_list_str = item.get("所用到的图片", "")
                image_paths = parse_image_paths(image_list_str)
                has_img = has_images(image_list_str)
                
                processed_item = {
                    "question_id": f"question_{question_id_counter}",
                    "img_path": image_paths,
                    "question": item.get("question", item.get("题目", f"题目{i+1}")),
                    "question_type": item.get("question_type", item.get("科目", "未知")),
                    "answer": item.get("answer", item.get("答案", "")),
                    "has_image": has_img,
                    "image_count": len(image_paths),
                    "raw_image_info": image_list_str
                }
                
                processed_data.append(processed_item)
    
    # 处理按科目分类的字典格式数据（高考最终汇总.json格式）
    elif isinstance(raw_data, dict):
        print("处理按科目分类的数据...")
        for subject, questions in raw_data.items():
            if not isinstance(questions, list):
                continue
                
            print(f"处理科目: {subject}, 题目数量: {len(questions)}")
            
            for i, question_item in enumerate(tqdm.tqdm(questions, desc=f"处理{subject}题目")):
                question_id_counter += 1
                
                # 解析图片信息
                image_list_str = question_item.get("所用到的图片", "")
                image_paths = parse_image_paths(image_list_str)
                has_img = has_images(image_list_str)
                
                # 构建处理后的数据项
                # 组合问题文本：题干 + 选项
                question_parts = []
                
                # 添加题干
                question_stem = question_item.get("题干", "")
                if question_stem:
                    question_parts.append(question_stem)
                
                # 添加选项
                options = []
                for option_key in ["选项A", "选项B", "选项C", "选项D"]:
                    option_value = question_item.get(option_key, "")
                    if option_value:
                        option_letter = option_key[-1]  # 获取A、B、C、D
                        options.append(f"{option_letter}. {option_value}")
                
                if options:
                    question_parts.append("\n".join(options))
                
                # 组合完整问题
                full_question = "\n\n".join(question_parts) if question_parts else f"{subject}第{i+1}题"
                
                # 获取正确答案
                correct_answer = question_item.get("正确选项", question_item.get("answer", question_item.get("答案", "")))
                
                processed_item = {
                    "question_id": f"{subject}_{i+1}",
                    "img_path": image_paths,
                    "question": full_question,
                    "question_type": subject,
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
        "subjects": {}
    }
    
    # 按科目统计
    for item in processed_data:
        subject = item["question_type"]
        if subject not in stats["subjects"]:
            stats["subjects"][subject] = {"total": 0, "with_images": 0}
        stats["subjects"][subject]["total"] += 1
        if item["has_image"]:
            stats["subjects"][subject]["with_images"] += 1
    
    print("\n数据统计:")
    print(f"总题目数: {stats['total_questions']}")
    print(f"有图片题目: {stats['with_images']}")
    print(f"无图片题目: {stats['without_images']}")
    print("\n各科目统计:")
    for subject, count_info in stats["subjects"].items():
        print(f"  {subject}: {count_info['total']} 题 (其中 {count_info['with_images']} 题有图片)")
    
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


if __name__ == "__main__":
    # 测试代码
    import tempfile
    import shutil
    
    print("开始测试CEE-MM数据处理脚本...")
    
    # 创建测试配置
    class TestConfig:
        def __init__(self, dataset_path):
            self.dataset_path = dataset_path
            self.split = "test"
            self.processed_dataset_path = "test_output"  # 保存到当前目录的test_output文件夹
            self.dataset_name = ""
        
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    # 测试1: 创建模拟的高考数据格式
    print("\n=== 测试1: 高考数据格式 ===")
    test_gaokao_data = {
        "数学": [
            {
                "所用到的图片": "['高考_副本/2008高考/数学/image1.jpeg']",
                "题干": "如图所示，求这个图形的面积",
                "选项A": "20平方米",
                "选项B": "24平方米", 
                "选项C": "30平方米",
                "选项D": "36平方米",
                "正确选项": "B"
            },
            {
                "所用到的图片": "",
                "题干": "计算下列积分 ∫₀^π sin(x)dx",
                "选项A": "0",
                "选项B": "1",
                "选项C": "2", 
                "选项D": "π",
                "正确选项": "C"
            }
        ],
        "物理": [
            {
                "所用到的图片": "['高考_副本/2009高考/物理/image1.png', '高考_副本/2009高考/物理/image2.png']",
                "题干": "根据电路图，计算总电阻",
                "选项A": "10Ω",
                "选项B": "15Ω",
                "选项C": "20Ω",
                "选项D": "25Ω",
                "正确选项": "B"
            }
        ]
    }
    
    # 保存测试数据
    test_file1 = "test_gaokao_data.json"
    with open(test_file1, "w", encoding="utf-8") as f:
        json.dump(test_gaokao_data, f, ensure_ascii=False, indent=2)
    
    try:
        cfg1 = TestConfig(test_file1)
        process(cfg1)
        
        # 验证输出
        output_file1 = osp.join(cfg1.processed_dataset_path, cfg1.split, "data.json")
        if osp.exists(output_file1):
            with open(output_file1, "r", encoding="utf-8") as f:
                processed_data1 = json.load(f)
            print(f"✅ 测试1通过: 处理了 {len(processed_data1)} 个问题")
            
            # 显示前2个样本
            for i, item in enumerate(processed_data1[:2]):
                print(f"  样本{i+1}: {item['question_id']} - {item['question_type']} - 有图片: {item['has_image']}")
        else:
            print("❌ 测试1失败: 输出文件不存在")
            
    except Exception as e:
        print(f"❌ 测试1失败: {e}")
    finally:
        # 清理测试文件
        if osp.exists(test_file1):
            os.remove(test_file1)
        # 保留测试输出目录，不删除
    
    # 测试2: 创建模拟的Hugging Face数据格式
    print("\n=== 测试2: Hugging Face数据格式 ===")
    test_hf_data = [
        {
            "question": "What is shown in this image?",
            "answer": "A cat",
            "image": ["path/to/image1.jpg"],
            "subject": "生物"
        },
        {
            "question": "Solve this equation: x + 5 = 10",
            "answer": "x = 5",
            "image": [],
            "subject": "数学"
        }
    ]
    
    test_file2 = "test_hf_data.json"
    with open(test_file2, "w", encoding="utf-8") as f:
        json.dump(test_hf_data, f, ensure_ascii=False, indent=2)
    
    try:
        cfg2 = TestConfig(test_file2)
        process(cfg2)
        
        # 验证输出
        output_file2 = osp.join(cfg2.processed_dataset_path, cfg2.split, "data.json")
        if osp.exists(output_file2):
            with open(output_file2, "r", encoding="utf-8") as f:
                processed_data2 = json.load(f)
            print(f"✅ 测试2通过: 处理了 {len(processed_data2)} 个问题")
            
            # 显示样本
            for i, item in enumerate(processed_data2):
                print(f"  样本{i+1}: {item['question_id']} - {item['question_type']} - 有图片: {item['has_image']}")
        else:
            print("❌ 测试2失败: 输出文件不存在")
            
    except Exception as e:
        print(f"❌ 测试2失败: {e}")
    finally:
        # 清理测试文件
        if osp.exists(test_file2):
            os.remove(test_file2)
        # 保留测试输出目录，不删除
    
    # 测试3: 真实高考数据（如果存在）
    print("\n=== 测试3: 真实高考数据 ===")
    real_gaokao_file = "高考最终汇总.json"
    if osp.exists(real_gaokao_file):
        try:
            cfg3 = TestConfig(real_gaokao_file)
            process(cfg3)
            
            # 验证输出
            output_file3 = osp.join(cfg3.processed_dataset_path, cfg3.split, "data.json")
            stats_file3 = osp.join(cfg3.processed_dataset_path, cfg3.split, "statistics.json")
            
            if osp.exists(output_file3) and osp.exists(stats_file3):
                with open(stats_file3, "r", encoding="utf-8") as f:
                    stats = json.load(f)
                print(f"✅ 测试3通过: 真实数据处理成功")
                print(f"  总题目数: {stats['total_questions']}")
                print(f"  有图片题目: {stats['with_images']}")
                print(f"  科目分布: {list(stats['subjects'].keys())}")
            else:
                print("❌ 测试3失败: 输出文件不完整")
                
        except Exception as e:
            print(f"❌ 测试3失败: {e}")
        # 保留测试输出目录，不删除
    else:
        print("⚠️  跳过测试3: 真实数据文件不存在")
    
    print("\n=== 测试完成 ===")
    print("如果所有测试都通过，说明process.py工作正常！")
    print(f"测试输出已保存到: {osp.abspath('test_output')}")
    print("您可以查看 test_output/test/ 目录下的 data.json 和 statistics.json 文件")
