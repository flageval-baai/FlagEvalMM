"""
CEE-MM 评估脚本
用于评估高考多模态题目的模型性能
"""

import json
from typing import Dict, List
import re


def normalize_answer(answer: str) -> str:
    """
    标准化答案，去除多余的空格、标点符号等
    """
    if not answer:
        return ""
    
    # 转换为小写
    answer = answer.lower().strip()
    
    # 去除多余的空格
    answer = re.sub(r'\s+', ' ', answer)
    
    # 去除常见的标点符号
    answer = re.sub(r'[。，、；：！？""''（）【】\[\]().,;:!?"\'\[\]]', '', answer)
    
    return answer.strip()


def exact_match_score(pred: str, target: str) -> int:
    """
    计算精确匹配分数
    """
    pred_norm = normalize_answer(pred)
    target_norm = normalize_answer(target)
    
    return 1 if pred_norm == target_norm else 0


def contains_match_score(pred: str, target: str) -> int:
    """
    计算包含匹配分数（预测答案是否包含正确答案）
    """
    pred_norm = normalize_answer(pred)
    target_norm = normalize_answer(target)
    
    if not target_norm:
        return 0
    
    return 1 if target_norm in pred_norm else 0


def calculate_subject_accuracy(predictions: List[Dict], subject: str) -> Dict[str, float]:
    """
    计算特定科目的准确率（仅包含匹配）
    """
    subject_preds = [p for p in predictions if p.get('question_type', '').startswith(subject)]
    
    if not subject_preds:
        return {
            f'{subject}_accuracy': 0.0,
            f'{subject}_count': 0
        }
    
    contains_matches = sum(
        contains_match_score(pred.get('prediction', ''), pred.get('answer', ''))
        for pred in subject_preds
    )
    
    total = len(subject_preds)
    
    return {
        f'{subject}_accuracy': contains_matches / total,
        f'{subject}_count': total
    }


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    """
    评估模型预测结果
    
    Args:
        annotations: 标注数据（包含原始数据信息）
        predictions: 模型预测结果列表，每个元素包含：
            - question_id: 问题ID
            - prediction: 模型预测的答案
            - answer: 正确答案（用于计算准确率）
            - question_type: 问题类型（科目）
            
    Returns:
        Dict: 包含评估指标的字典
    """
    if not predictions:
        return {"error": "No predictions provided"}
    
    # 整体准确率（包含匹配）
    total_contains_matches = sum(
        contains_match_score(pred.get('prediction', ''), pred.get('answer', ''))
        for pred in predictions
    )
    
    total_count = len(predictions)
    
    results = {
        'overall_accuracy': total_contains_matches / total_count,
        'total_questions': total_count
    }
    
    # 按科目分别计算准确率
    subjects = ['数学', '物理', '化学', '生物', '地理', '历史']
    
    for subject in subjects:
        subject_results = calculate_subject_accuracy(predictions, subject)
        results.update(subject_results)
    
    return results
