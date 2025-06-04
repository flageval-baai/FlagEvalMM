      
import json
import re
import numpy as np
from scipy.spatial.distance import directed_hausdorff, cdist
from typing import Dict, List, Tuple

import ast
def extract_points(point_str, width, height):

    try:
        aa = ast.literal_eval(point_str)
        bb = []
        for item in aa:
            if item[0] > 1 or item[1] > 1:
                bb.append((item[0]/width, item[1]/height))
            else:
                bb.append((item[0], item[1]))
        return bb
    except Exception:
        pass
    point_str = point_str.replace('–', '-').replace('−', '-')  
    
    list_match = re.search(r'\[(\s*\[[^\]]+\](?:\s*,\s*\[[^\]]+\])*\s*)\]', point_str)
    if list_match:
        try:
            bb = []
            for  item in  eval(list_match.group(0)):  
                if item[0] > 1 or item[1] > 1:
                    bb.append((item[0]/width, item[1]/height))
                else:
                    bb.append((item[0], item[1]))
            return bb
        except:
            pass
    
    points = []
    for line in point_str.split('\n'):
        clean_line = re.sub(r'#.*$', '', line).strip()
        
        if match := re.search(r'\[([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\]', clean_line):
            x = float(match.group(1))
            y = float(match.group(2))
            if x > 1:
                x = x / width
            if y > 1:
                y = y / height
            points.append([x, y])
    
    return points


def discrete_frechet_distance(P, Q):
    n = len(P)
    m = len(Q)
    ca = np.zeros((n, m))
    ca.fill(-1.0)
    dist_matrix = cdist(P, Q, 'euclidean')
    def compute_ca(i, j):
        if ca[i, j] > -1.0:
            return ca[i, j]
        if i == 0 and j == 0:
            ca[i, j] = dist_matrix[0, 0]
        elif i == 0:
            ca[i, j] = max(compute_ca(0, j-1), dist_matrix[i, j])
        elif j == 0:
            ca[i, j] = max(compute_ca(i-1, 0), dist_matrix[i, j])
        else:
            ca[i, j] = max(min(compute_ca(i-1, j), 
                               compute_ca(i-1, j-1), 
                               compute_ca(i, j-1)), 
                           dist_matrix[i, j])
        return ca[i, j]
    return compute_ca(n-1, m-1)

def hausdorff_distance(P, Q):
    return max(directed_hausdorff(P, Q)[0], directed_hausdorff(Q, P)[0])

def root_mean_square_error(P, Q):
    min_len = min(len(P), len(Q))
    if min_len == 0:
        return 0.0
    squared_distances = [np.sum((np.array(P[i]) - np.array(Q[i]))**2) 
                         for i in range(min_len)]
    return np.sqrt(np.mean(squared_distances))

def calculate_metrics(data):
    # with open(input_file, 'r') as f:
    #     data = json.load(f)
    dfd_list = []
    hd_list = []
    rmse_list = []
    for item in data:
        width = item["width"]
        height = item["height"]
        try:
            pred_points = extract_points(item['answer'], width, height)
            gt_points = extract_points(item['gt'], width, height)
            if not pred_points:
                pred_points = [(0,0)]
            pred_array = np.array(pred_points)
            gt_array = np.array(gt_points)
            dfd = discrete_frechet_distance(pred_array, gt_array)
            hd = hausdorff_distance(pred_array, gt_array)
            rms = root_mean_square_error(pred_points, gt_points)
            dfd_list.append(dfd)
            hd_list.append(hd)
            rmse_list.append(rms)
        except Exception as e:
            print(f"Error processing item {item['id']}: {e}")
            
    avg_dfd = np.mean(dfd_list) if dfd_list else 0.0
    avg_hd = np.mean(hd_list) if hd_list else 0.0
    avg_rmse = np.mean(rmse_list) if rmse_list else 0.0
    return {
        'average_discrete_frechet_distance': avg_dfd,
        'average_hausdorff_distance': avg_hd,
        'average_root_mean_square_error': avg_rmse
    }


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    result = []
    for idx, pred in enumerate(predictions):
        question_id = str(pred["question_id"])
        gt_data = annotations[question_id]
        gt_answer = gt_data["answer"]
        pred["gt"] = gt_answer
        pred["height"] = gt_data["height"]
        pred["width"] = gt_data["width"]
        result.append(pred)

    acc = calculate_metrics(result)
    return acc
    




# 使用示例
if __name__ == "__main__":
    # input_file = "roboos_trajectory_new_with_gt.json"
    input_file = "qwenvl_trajectory_new_with_gt.json"
    metrics = calculate_metrics(input_file)
    print("Average Metrics:")
    print(f"Discrete Fréchet Distance: {metrics['average_discrete_frechet_distance']:.4f}")
    print(f"Hausdorff Distance: {metrics['average_hausdorff_distance']:.4f}")
    print(f"Root Mean Square Error: {metrics['average_root_mean_square_error']:.4f}")

    