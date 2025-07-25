"""
MEE-MM 评估脚本
支持多种考试类型的动态评估
支持精确匹配和包含匹配两种评估模式
"""

import json
import re
from collections import defaultdict


def normalize_answer(answer: str) -> str:
    """
    标准化答案文本
    """
    if not answer:
        return ""
    
    # 转换为小写
    answer = answer.lower().strip()
    
    # 移除标点符号和特殊字符
    answer = re.sub(r'[^\w\s\u4e00-\u9fff]', '', answer)
    
    # 移除多余空格
    answer = re.sub(r'\s+', ' ', answer).strip()
    
    return answer


def check_answer(predicted: str, ground_truth: str, match_mode="contains") -> bool:
    """
    检查答案是否正确
    
    Args:
        predicted: 预测答案
        ground_truth: 正确答案
        match_mode: 匹配模式，"exact" 或 "contains"
    
    Returns:
        是否匹配
    """
    if not ground_truth:
        return False
    
    pred_norm = normalize_answer(predicted)
    gt_norm = normalize_answer(ground_truth)
    
    if not pred_norm or not gt_norm:
        return False
    
    if match_mode == "exact":
        return pred_norm == gt_norm
    elif match_mode == "contains":
        # 检查预测答案是否包含正确答案，或者正确答案是否包含预测答案
        return gt_norm in pred_norm or pred_norm in gt_norm
    else:
        return False


def calculate_type_accuracy(predictions: list, references: list, match_mode="contains") -> dict:
    """
    按问题类型计算准确率
    
    Args:
        predictions: 预测列表，每个元素包含 question_type 和 predicted_answer
        references: 参考答案列表，每个元素包含 question_type 和 ground_truth
        match_mode: 匹配模式
    
    Returns:
        包含各类型准确率的字典
    """
    # 按类型分组统计
    type_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    # 确保预测和参考答案数量一致
    if len(predictions) != len(references):
        print(f"警告：预测数量({len(predictions)})与参考答案数量({len(references)})不一致")
    
    # 遍历所有问题
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        # 获取问题类型
        question_type = ref.get("question_type", "未知")
        
        # 获取预测答案和正确答案
        predicted_answer = pred.get("predicted_answer", "")
        ground_truth = ref.get("answer", "")
        
        # 更新统计
        type_stats[question_type]["total"] += 1
        
        # 检查答案正确性
        if check_answer(predicted_answer, ground_truth, match_mode):
            type_stats[question_type]["correct"] += 1
    
    # 计算准确率
    results = {}
    overall_correct = 0
    overall_total = 0
    
    for question_type, stats in type_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        results[question_type] = {
            "accuracy": accuracy,
            "correct": stats["correct"],
            "total": stats["total"]
        }
        
        overall_correct += stats["correct"]
        overall_total += stats["total"]
    
    # 计算总体准确率
    overall_accuracy = overall_correct / overall_total if overall_total > 0 else 0.0
    results["overall"] = {
        "accuracy": overall_accuracy,
        "correct": overall_correct,
        "total": overall_total
    }
    
    return results


def calculate_format_accuracy(predictions: list, references: list, match_mode="contains") -> dict:
    """
    按数据格式计算准确率
    """
    format_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        # 获取数据格式
        data_format = ref.get("data_format", "unknown")
        
        # 获取预测答案和正确答案
        predicted_answer = pred.get("predicted_answer", "")
        ground_truth = ref.get("answer", "")
        
        # 更新统计
        format_stats[data_format]["total"] += 1
        
        # 检查答案正确性
        if check_answer(predicted_answer, ground_truth, match_mode):
            format_stats[data_format]["correct"] += 1
    
    # 计算准确率
    results = {}
    for data_format, stats in format_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        results[data_format] = {
            "accuracy": accuracy,
            "correct": stats["correct"],
            "total": stats["total"]
        }
    
    return results


def calculate_image_accuracy(predictions: list, references: list, match_mode="contains") -> dict:
    """
    按是否包含图片计算准确率
    """
    image_stats = {
        "with_image": {"correct": 0, "total": 0},
        "without_image": {"correct": 0, "total": 0}
    }
    
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        # 获取图片信息
        has_image = ref.get("has_image", False)
        category = "with_image" if has_image else "without_image"
        
        # 获取预测答案和正确答案
        predicted_answer = pred.get("predicted_answer", "")
        ground_truth = ref.get("answer", "")
        
        # 更新统计
        image_stats[category]["total"] += 1
        
        # 检查答案正确性
        if check_answer(predicted_answer, ground_truth, match_mode):
            image_stats[category]["correct"] += 1
    
    # 计算准确率
    results = {}
    for category, stats in image_stats.items():
        accuracy = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        results[category] = {
            "accuracy": accuracy,
            "correct": stats["correct"],
            "total": stats["total"]
        }
    
    return results


def run_evaluate(answers: list, references: list, cfg=None) -> dict:
    """
    执行MEE-MM评估
    
    Args:
        answers: 模型预测结果列表
        references: 参考答案列表
        cfg: 配置对象（包含评估参数）
    
    Returns:
        评估结果字典
    """
    
    print(f"开始MEE-MM评估...")
    print(f"预测答案数量: {len(answers)}")
    print(f"参考答案数量: {len(references)}")
    
    # 设置匹配模式（优先使用包含匹配，更宽松）
    match_mode = getattr(cfg, 'match_mode', 'contains')
    print(f"匹配模式: {match_mode}")
    
    # 数据验证
    if len(answers) != len(references):
        print(f"警告：答案数量不匹配！")
        min_len = min(len(answers), len(references))
        answers = answers[:min_len]
        references = references[:min_len]
        print(f"截取前 {min_len} 个进行评估")
    
    # 1. 按问题类型计算准确率
    print("\n计算按问题类型的准确率...")
    type_accuracy = calculate_type_accuracy(answers, references, match_mode)
    
    # 2. 按数据格式计算准确率
    print("计算按数据格式的准确率...")
    format_accuracy = calculate_format_accuracy(answers, references, match_mode)
    
    # 3. 按图片情况计算准确率
    print("计算按图片情况的准确率...")
    image_accuracy = calculate_image_accuracy(answers, references, match_mode)
    
    # 输出详细结果
    print("\n=== MEE-MM 评估结果 ===")
    print(f"\n总体准确率: {type_accuracy['overall']['accuracy']:.4f} "
          f"({type_accuracy['overall']['correct']}/{type_accuracy['overall']['total']})")
    
    print("\n按问题类型统计:")
    type_results = {k: v for k, v in type_accuracy.items() if k != "overall"}
    sorted_types = sorted(type_results.items(), key=lambda x: x[1]["accuracy"], reverse=True)
    
    for question_type, stats in sorted_types:
        print(f"  {question_type}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
    
    print("\n按数据格式统计:")
    for data_format, stats in format_accuracy.items():
        print(f"  {data_format}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
    
    print("\n按图片情况统计:")
    for category, stats in image_accuracy.items():
        display_name = "有图片题目" if category == "with_image" else "无图片题目"
        print(f"  {display_name}: {stats['accuracy']:.4f} ({stats['correct']}/{stats['total']})")
    
    # 构建返回结果
    eval_results = {
        "overall_accuracy": type_accuracy["overall"]["accuracy"],
        "total_questions": type_accuracy["overall"]["total"],
        "correct_answers": type_accuracy["overall"]["correct"],
        "match_mode": match_mode,
        "type_accuracy": type_accuracy,
        "format_accuracy": format_accuracy,
        "image_accuracy": image_accuracy
    }
    
    # 添加一些有用的统计信息
    eval_results["statistics"] = {
        "question_types": list(type_results.keys()),
        "data_formats": list(format_accuracy.keys()),
        "best_performing_type": max(type_results.items(), key=lambda x: x[1]["accuracy"])[0] if type_results else None,
        "worst_performing_type": min(type_results.items(), key=lambda x: x[1]["accuracy"])[0] if type_results else None
    }
    
    print(f"\nMEE-MM 评估完成！")
    print(f"最佳表现类型: {eval_results['statistics']['best_performing_type']}")
    print(f"表现最差类型: {eval_results['statistics']['worst_performing_type']}")
    
    return eval_results
