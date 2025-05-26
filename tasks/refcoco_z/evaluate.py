from datasets import load_dataset
from typing import Dict, List, Tuple
from collections import defaultdict
import json
import math
import torch
import matplotlib.image as plt
import numpy as np
from statistics import mean
import pandas as pd
import ast


def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False, eps=1e-6):
    """Calculate overlap between two set of bboxes.

    FP16 Contributed by https://github.com/open-mmlab/mmdetection/pull/4889
    Note:
        Assume bboxes1 is M x 4, bboxes2 is N x 4, when mode is 'iou',
        there are some new generated variable when calculating IOU
        using bbox_overlaps function:

        1) is_aligned is False
            area1: M x 1
            area2: N x 1
            lt: M x N x 2
            rb: M x N x 2
            wh: M x N x 2
            overlap: M x N x 1
            union: M x N x 1
            ious: M x N x 1

            Total memory:
                S = (9 x N x M + N + M) * 4 Byte,

            When using FP16, we can reduce:
                R = (9 x N x M + N + M) * 4 / 2 Byte
                R large than (N + M) * 4 * 2 is always true when N and M >= 1.
                Obviously, N + M <= N * M < 3 * N * M, when N >=2 and M >=2,
                           N + 1 < 3 * N, when N or M is 1.

            Given M = 40 (ground truth), N = 400000 (three anchor boxes
            in per grid, FPN, R-CNNs),
                R = 275 MB (one times)

            A special case (dense detection), M = 512 (ground truth),
                R = 3516 MB = 3.43 GB

            When the batch size is B, reduce:
                B x R

            Therefore, CUDA memory runs out frequently.

            Experiments on GeForce RTX 2080Ti (11019 MiB):

            |   dtype   |   M   |   N   |   Use    |   Real   |   Ideal   |
            |:----:|:----:|:----:|:----:|:----:|:----:|
            |   FP32   |   512 | 400000 | 8020 MiB |   --   |   --   |
            |   FP16   |   512 | 400000 |   4504 MiB | 3516 MiB | 3516 MiB |
            |   FP32   |   40 | 400000 |   1540 MiB |   --   |   --   |
            |   FP16   |   40 | 400000 |   1264 MiB |   276MiB   | 275 MiB |

        2) is_aligned is True
            area1: N x 1
            area2: N x 1
            lt: N x 2
            rb: N x 2
            wh: N x 2
            overlap: N x 1
            union: N x 1
            ious: N x 1

            Total memory:
                S = 11 x N * 4 Byte

            When using FP16, we can reduce:
                R = 11 x N * 4 / 2 Byte

        So do the 'giou' (large than 'iou').

        Time-wise, FP16 is generally faster than FP32.

        When gpu_assign_thr is not -1, it takes more time on cpu
        but not reduce memory.
        There, we can reduce half the memory and keep the speed.

    If ``is_aligned`` is ``False``, then calculate the overlaps between each
    bbox of bboxes1 and bboxes2, otherwise the overlaps between each aligned
    pair of bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (B, m, 4) in <x1, y1, x2, y2> format or empty.
        bboxes2 (Tensor): shape (B, n, 4) in <x1, y1, x2, y2> format or empty.
            B indicates the batch dim, in shape (B1, B2, ..., Bn).
            If ``is_aligned`` is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union), "iof" (intersection over
            foreground) or "giou" (generalized intersection over union).
            Default "iou".
        is_aligned (bool, optional): If True, then m and n must be equal.
            Default False.
        eps (float, optional): A value added to the denominator for numerical
            stability. Default 1e-6.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2)
        >>> assert overlaps.shape == (3, 3)
        >>> overlaps = bbox_overlaps(bboxes1, bboxes2, is_aligned=True)
        >>> assert overlaps.shape == (3, )

    Example:
        >>> empty = torch.empty(0, 4)
        >>> nonempty = torch.FloatTensor([[0, 0, 10, 9]])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """

    assert mode in ['iou', 'iof', 'giou'], f'Unsupported mode {mode}'
    # Either the boxes are empty or the length of boxes' last dimension is 4
    assert (bboxes1.size(-1) == 4 or bboxes1.size(0) == 0)
    assert (bboxes2.size(-1) == 4 or bboxes2.size(0) == 0)

    # Batch dim must be the same
    # Batch dim: (B1, B2, ... Bn)
    assert bboxes1.shape[:-2] == bboxes2.shape[:-2]
    batch_shape = bboxes1.shape[:-2]

    rows = bboxes1.size(-2)
    cols = bboxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        if is_aligned:
            return bboxes1.new(batch_shape + (rows, ))
        else:
            return bboxes1.new(batch_shape + (rows, cols))

    area1 = (bboxes1[..., 2] - bboxes1[..., 0]) * (
        bboxes1[..., 3] - bboxes1[..., 1])
    area2 = (bboxes2[..., 2] - bboxes2[..., 0]) * (
        bboxes2[..., 3] - bboxes2[..., 1])

    if is_aligned:
        lt = torch.max(bboxes1[..., :2], bboxes2[..., :2])  # [B, rows, 2]
        rb = torch.min(bboxes1[..., 2:], bboxes2[..., 2:])  # [B, rows, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1 + area2 - overlap
        else:
            union = area1
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :2], bboxes2[..., :2])
            enclosed_rb = torch.max(bboxes1[..., 2:], bboxes2[..., 2:])
    else:
        lt = torch.max(bboxes1[..., :, None, :2],
                       bboxes2[..., None, :, :2])  # [B, rows, cols, 2]
        rb = torch.min(bboxes1[..., :, None, 2:],
                       bboxes2[..., None, :, 2:])  # [B, rows, cols, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlap = wh[..., 0] * wh[..., 1]

        if mode in ['iou', 'giou']:
            union = area1[..., None] + area2[..., None, :] - overlap
        else:
            union = area1[..., None]
        if mode == 'giou':
            enclosed_lt = torch.min(bboxes1[..., :, None, :2],
                                    bboxes2[..., None, :, :2])
            enclosed_rb = torch.max(bboxes1[..., :, None, 2:],
                                    bboxes2[..., None, :, 2:])

    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlap / union
    if mode in ['iou', 'iof']:
        return ious
    # calculate gious
    enclose_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclose_area = enclose_wh[..., 0] * enclose_wh[..., 1]
    enclose_area = torch.max(enclose_area, eps)
    gious = ious - (enclose_area - union) / enclose_area
    return gious


mapping_dict = {
    "refcoco_1": "o365_1",
    "refcoco_10": "o365_41",
    "refcoco_11": "o365_177",
    "refcoco_13": "o365_128",
    "refcoco_14": "o365_250",
    "refcoco_15": "o365_25",
    "refcoco_16": "o365_56",
    "refcoco_17": "o365_140",
    "refcoco_18": "o365_93",
    "refcoco_19": "o365_79",
    "refcoco_2": "o365_47",
    "refcoco_20": "o365_100",
    "refcoco_21": "o365_97",
    "refcoco_22": "o365_145",
    "refcoco_23": "o365_296",
    "refcoco_24": "o365_179",
    "refcoco_25": "o365_181",
    "refcoco_27": "o365_39",
    "refcoco_28": "o365_40",
    "refcoco_3": "o365_6",
    "refcoco_31": "o365_13",
    "refcoco_32": "o365_44",
    "refcoco_33": "o365_194",
    "refcoco_34": "o365_220",
    "refcoco_35": "o365_119",
    "refcoco_36": "o365_174",
    "refcoco_37": "o365_100000",    # This is a placeholder, the actual value is not provided
    "refcoco_38": "o365_155",
    "refcoco_39": "o365_138",
    "refcoco_4": "o365_59",
    "refcoco_40": "o365_114",
    "refcoco_41": "o365_146",
    "refcoco_42": "o365_147",
    "refcoco_43": "o365_205",
    "refcoco_44": "o365_9",
    "refcoco_46": "o365_36",
    "refcoco_47": "o365_11",
    "refcoco_48": "o365_89",
    "refcoco_49": "o365_85",
    "refcoco_5": "o365_115",
    "refcoco_50": "o365_94",
    "refcoco_51": "o365_26",
    "refcoco_52": "o365_113",
    "refcoco_53": "o365_83",
    "refcoco_54": "o365_266",
    "refcoco_55": "o365_104",
    "refcoco_56": "o365_142",
    "refcoco_57": "o365_153",
    "refcoco_58": "o365_235",
    "refcoco_59": "o365_144",
    "refcoco_6": "o365_56",
    "refcoco_60": "o365_151",
    "refcoco_61": "o365_98",
    "refcoco_62": "o365_3",
    "refcoco_63": "o365_51",
    "refcoco_64": "o365_26",
    "refcoco_65": "o365_76",
    "refcoco_67": "o365_98",
    "refcoco_7": "o365_117",
    "refcoco_70": "o365_154",
    "refcoco_72": "o365_37",
    "refcoco_73": "o365_74",
    "refcoco_74": "o365_116",
    "refcoco_75": "o365_133",
    "refcoco_76": "o365_107",
    "refcoco_77": "o365_62",
    "refcoco_78": "o365_164",
    "refcoco_79": "o365_135",
    "refcoco_8": "o365_66",
    "refcoco_80": "o365_278",
    "refcoco_81": "o365_82",
    "refcoco_82": "o365_134",
    "refcoco_84": "o365_19",
    "refcoco_85": "o365_95",
    "refcoco_86": "o365_31",
    "refcoco_87": "o365_170",
    "refcoco_88": "o365_70",
    "refcoco_89": "o365_328",
    "refcoco_9": "o365_22",
    "refcoco_90": "o365_227",
}


def evaluate_resoce(predictions, data_dir, save_file=None, 
                    split="all", 
                    ann_level_acc_ths=[0.5, 0.75, 0.9], 
                    ann_level_macc_ths=[i/100 for i in range(50,100,5)], 
                    small_size_th=128,
                    large_size_th=256,
                    size_level_acc_ths=[0.5, ],
                    size_level_macc_ths=[i/100 for i in range(50,100,5)],
                    avg_cls_level_acc_ths=[0.5, ],
                    avg_cls_level_macc_ths=[i/100 for i in range(50,100,5)],
                    ):
    """
    evaluate_resoce given dataset and predictions.

    Parameters:
    - predictions (List(Dict)): The predictions to evaluate_resoce. 
        Each item in the list is a dict, containing the keys: 'pred_bbox', 'id' and 'format', 
        where 'id' is the annotation id, 'format' is the bbox format 'xyxy' or 'xywh'.
        e.g.:
        [
            {
            'pred_bbox': [x1, y1, x2, y2],
            'id': '000000',
            'format': 'xyxy'
            },
            ...
        ]
    - save_file (str): The file to save the evaluation results to.
    """
    if(len(predictions)==0):
        print("Warning: No predictions found.")
        return dict()
    
    # convert predictions to a dict, key is the id, raise error if there are duplicate ids
    predictions_dict={int(pred['id']):pred for pred in predictions}
    if(len(predictions)!=len(predictions_dict)):
        raise ValueError("Duplicate ids found in the predictions.")

    gt_bboxes = []
    pred_bboxes = []
    dataset_split = load_dataset(data_dir, split=split)

    for idx, gt_data in enumerate(dataset_split):
        gt_bbox = gt_data['bbox']
        gt_bboxes.append([gt_bbox[0], gt_bbox[1], gt_bbox[0]+gt_bbox[2], gt_bbox[1]+gt_bbox[3]])
        
        # raise error if the id canot be found in the predictions
        if gt_data['id'] not in predictions_dict:
            # continue
            raise ValueError(f"Id {gt_data['id']} not found in the predictions.")

        pred_bbox = predictions_dict[gt_data['id']]['pred_bbox']
        if predictions_dict[gt_data['id']]['format'] == 'xywh':
            pred_bbox = [pred_bbox[0], pred_bbox[1], pred_bbox[0]+pred_bbox[2], pred_bbox[1]+pred_bbox[3]]
        pred_bboxes.append(pred_bbox)


    # calculate_iou
    iou = bbox_overlaps(torch.tensor(gt_bboxes), torch.tensor(pred_bboxes), mode='iou', is_aligned=True).numpy()

    acc_all = dict()

    # Annotation level evaluation
    for th in ann_level_acc_ths:
        acc = (iou > th).sum().item() / len(iou)
        key = f"Ann-level acc iou {th}"
        acc_all[key] = acc * 100
    
    macc = []
    for th in ann_level_macc_ths:
        acc = (iou > th).sum().item() / len(iou)
        macc.append(acc)
    key = f"Ann-level macc iou {ann_level_macc_ths[0]}:{ann_level_macc_ths[-1]}"
    acc_all[key] = mean(macc) * 100

    # get the acc for copy, so that we can output the results as a table, round to 2 decimal places
    # acc_all['Ann-level accs for copy'] = [acc_all[key] for key in acc_all]
    acc_all['Ann-level accs for copy'] = [
        round(acc_all[key], 2) for key in acc_all
    ]

    # Size evaluation
    small_size_list = []
    medium_size_list = []
    large_size_list = []
    for idx, gt_data in enumerate(dataset_split):
        gt_bbox = gt_data['bbox']
        obj_size = math.sqrt(gt_bbox[2]*gt_bbox[3])
        iou_item = iou[idx]

        # gather the small, medium and large size bboxes
        if obj_size < small_size_th:
            small_size_list.append(iou_item)
        elif obj_size <= large_size_th:
            medium_size_list.append(iou_item)
        else:
            large_size_list.append(iou_item)
    
    #  small size evaluation
    for th in size_level_acc_ths:
        acc_small = sum(i > th for i in small_size_list) / len(small_size_list) * 100
        key = f"Small acc iou {th}"
        acc_all[key] = acc_small 
            
    macc_small = []
    for th in size_level_macc_ths:
        small_size_list = np.array(small_size_list)
        acc_small = (small_size_list > th).sum().item() / len(small_size_list) * 100
        macc_small.append(acc_small)
    key = f"Small macc iou {size_level_macc_ths[0]}:{size_level_macc_ths[-1]}"
    acc_all[key] = mean(macc_small)

    # medium size evaluation
    for th in size_level_acc_ths:
        acc_medium = sum(i > th for i in medium_size_list) / len(medium_size_list) * 100
        key = f"Medium acc iou {th}"
        acc_all[key] = acc_medium

    macc_medium = []
    for th in size_level_macc_ths:
        medium_size_list = np.array(medium_size_list)
        acc_medium = (medium_size_list > th).sum().item() / len(medium_size_list) * 100
        macc_medium.append(acc_medium)
    key = f"Medium macc iou {size_level_macc_ths[0]}:{size_level_macc_ths[-1]}"
    acc_all[key] = mean(macc_medium)

    # large size evaluation
    for th in size_level_acc_ths:
        acc_large = sum(i > th for i in large_size_list) / len(large_size_list) * 100
        key = f"Large acc iou {th}"
        acc_all[key] = acc_large

    macc_large = []
    for th in size_level_macc_ths:
        large_size_list = np.array(large_size_list)
        acc_large = (large_size_list > th).sum().item() / len(large_size_list) * 100
        macc_large.append(acc_large)
    key = f"Large macc iou {size_level_macc_ths[0]}:{size_level_macc_ths[-1]}"
    acc_all[key] = mean(macc_large)

    # get the size-level acc for copy, so that we can output the results as a table, round to 2 decimal places
    acc_all['Size level accs for copy'] = [
        round(acc_all[key], 2) for key in acc_all
        if 'Small' in key or 'Medium' in key or 'Large' in key
    ]

    # Average class-level evaluation
    iou_avg_cls_level_acc_ths = dict()
    for idx, gt_data in enumerate(dataset_split):
        iou_item = iou[idx]
        if gt_data['ori_category_id'] in mapping_dict:
            ori_category_id = mapping_dict[gt_data['ori_category_id']]
        else:
            ori_category_id = gt_data['ori_category_id']

        if ori_category_id not in iou_avg_cls_level_acc_ths:
            iou_avg_cls_level_acc_ths[ori_category_id] = []
        iou_avg_cls_level_acc_ths[ori_category_id].append(iou_item)
    
    for th in avg_cls_level_acc_ths:
        acc_list = []
        for key in iou_avg_cls_level_acc_ths:
            iou_array = np.array(iou_avg_cls_level_acc_ths[key])
            acc = (iou_array > th).sum().item() / len(iou_array) * 100
            acc_list.append(acc)
        key = f"Average class-level acc iou {th}"
        acc_all[key] = mean(acc_list)

    # macc
    macc_list = []
    for th in avg_cls_level_macc_ths:
        acc_list = []
        for key in iou_avg_cls_level_acc_ths:
            iou_array = np.array(iou_avg_cls_level_acc_ths[key])
            acc = (iou_array > th).sum().item() / len(iou_avg_cls_level_acc_ths[key]) * 100
            acc_list.append(acc)
        macc_list.append(mean(acc_list))
    key = f"Average class-level macc iou {avg_cls_level_macc_ths[0]}:{avg_cls_level_macc_ths[-1]}"
    acc_all[key] = mean(macc_list)

    # get the avg_cls-level acc for copy, so that we can output the results as a table, round to 2 decimal places
    acc_all['Avg class-level accs for copy'] = [
        round(acc_all[key], 2) for key in acc_all
        if 'Average class-level' in key
    ]


    # Output as table
    table = []
    table.append([f"Item for split {split}", "Value"])
    for k, v in acc_all.items():
        if isinstance(v, list):
            table.append([k, ", ".join(map(str, v))])
        else:
            table.append([k, v])

    # Define where to add horizontal lines
    horizontal_lines = {1, 6, 13}  # After header, IoU, and Subject evaluations

    # Print table with selective horizontal lines
    max_len = max(len(row[0]) for row in table)
    for i, row in enumerate(table):
        if i in horizontal_lines:
            print('-' * (max_len + 3 + max(len(str(r[1])) for r in table)))
        print(f"{row[0].ljust(max_len)} | {row[1]}")

    if(save_file is not None):
        acc_all.pop('Ann-level accs for copy')
        acc_all.pop('Size level accs for copy')
        acc_all.pop('Avg class-level accs for copy')
        df=pd.DataFrame(acc_all, index=[0])
        df.to_csv(save_file)        
                    
    return acc_all



def center_to_corners(box, image_path):
    
    try:
        if "json" in box:
            json_str = box.strip('```json\n').strip('```').strip()
            data = json.loads(json_str)
            box = data[0].get("bbox_2d", [0, 0, 0, 0])
        else:
            box =  ast.literal_eval(box)
        if len(box) == 1:
            x, y, x_prime, y_prime = map(float, box[0])
            w = x_prime - x
            h = y_prime - y
            return [x, y, abs(w), abs(h)]
    except:
        box = [0, 0, 0, 0]
    print(box, len(box))
    if len(box) < 4:
        return [0, 0, 0, 0]
    x1, y1, x2, y2 = box[:4]
    if type(x1) != int:
        return [0, 0, 0, 0]
    if x1 > 1 or y1 > 1 or x2 > 1 or y2 > 1:
        return [x1, y1, x2, y2]

    image = plt.imread(image_path)
    img_height, img_weith = image.shape[:2]   
    x_min = x1 * img_weith
    y_min = y1 * img_height
    x_max = x2 * img_weith
    y_max = y2 * img_height
    return [x_min, y_min, x_max, y_max]



def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    """评估模型预测结果
    
    Args:
        annotations: 标注数据
        predictions: 模型预测结果
        
    Returns:
        Dict: 包含评估指标的字典
    """

    results = defaultdict(lambda: {"num": 0, "correct": 0})
    prediction_data = []
    annotation_data = []
    for pred in predictions:
        question_id = str(pred["question_id"])
        gt = annotations[question_id]
        answer_box = gt["answer"]
        image_path = gt["img_path"]
        print(pred)
        pred_box = center_to_corners(pred["answer"], image_path)
        print(f"-----绝对坐标----{pred_box}")
        prediction_data.append({
            "pred_bbox": pred_box,
            "id": question_id,
            "format": "xyxy"
        })
        annotation_data.append({
            "id": question_id,
            "bbox": answer_box,
            
        })
    results = evaluate_resoce(prediction_data, data_dir="/share/project/benchmarks/Ref-L4", split="all")
    return results
