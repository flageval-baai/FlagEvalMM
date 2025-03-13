import json
import os.path as osp

from typing import List
from flagevalmm.registry import EVALUATORS
from flagevalmm.evaluator import BaseEvaluator
from collections import defaultdict


@EVALUATORS.register_module()
class MLUVDatasetEvaluator(BaseEvaluator):
    def __init__(self, metrics: List = ["accuracy"]) -> None:
        self.metrics = metrics

    def check_ans(pred, gt):
        flag = False
        
        pred_list = pred.lower().split(' ')
        pred_option, pred_content = pred_list[0], ' '.join(pred_list[1:])
        gt_list = gt.lower().split(' ')
        gt_option, gt_content = gt_list[0], ' '.join(gt_list[1:])
        if gt_content[-1] == '.':
            gt_content = gt_content[:-1]
        
        if pred_option.replace('.', '') in gt_option:
            flag = True
        elif gt_option in pred_option:
            flag = True
            
        return flag

    def process(self, dataset, output_dir, **kwargs):
        """
        Args:
            dataset (Dataset): dataset instance
            output_dir: str
        """
        annotation = dataset.get_annotation()
        result_file = osp.join(output_dir, dataset.name + ".json")
        predictions = json.load(open(result_file))
        acc_dict = defaultdict(lambda: [0, 0])
        for pred in predictions:
            question_id = pred["question_id"]
            pred = pred["answer"]
            gt = annotation[question_id]["answer"]
            task_type = annotation[question_id]["data"]["task_type"]
            acc_dict[task_type][1] += 1
            if self.check_ans(pred, gt):
                acc_dict[task_type][0] += 1
        results = dict()
        total = 0
        idx = 0
        for k, v in acc_dict.items():
            idx += 1
            results[k] = v[0] / v[1] * 100
            total += results[k]
        results["accuracy"] = total / idx
        print(results)
        self.save(results, predictions, dataset.name, output_dir)
        return results