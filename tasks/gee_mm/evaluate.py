"""
GEE-MM 评估脚本
用于评估国考多模态题目的模型性能
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


def contains_match_score(pred: str, target: str) -> int:
    """
    计算包含匹配分数（预测答案是否包含正确答案）
    """
    pred_norm = normalize_answer(pred)
    target_norm = normalize_answer(target)
    
    if not target_norm:
        return 0
    
    return 1 if target_norm in pred_norm else 0


def calculate_type_accuracy(predictions: List[Dict], question_type: str) -> Dict[str, float]:
    """
    计算特定类型的准确率（仅包含匹配）
    """
    type_preds = [p for p in predictions if p.get('question_type', '').startswith(question_type)]
    
    if not type_preds:
        return {
            f'{question_type}_accuracy': 0.0,
            f'{question_type}_count': 0
        }
    
    contains_matches = sum(
        contains_match_score(pred.get('prediction', ''), pred.get('answer', ''))
        for pred in type_preds
    )
    
    total = len(type_preds)
    
    return {
        f'{question_type}_accuracy': contains_matches / total,
        f'{question_type}_count': total
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
            - question_type: 问题类型
            
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
    
    # 获取所有出现的题目类型
    question_types = set(pred.get('question_type', '') for pred in predictions)
    question_types = [t for t in question_types if t]  # 移除空字符串
    
    # 按类型分别计算准确率
    for q_type in question_types:
        type_results = calculate_type_accuracy(predictions, q_type)
        results.update(type_results)
    
    return results
