from typing import Dict, List, Tuple
import re
from collections import defaultdict
import ast
from flagevalmm.evaluator.pre_process import normalize_string
import numpy as np
import os.path as osp
import os
from PIL import Image, ImageDraw


# From the official evaluation code of RoboSpatial-Home: https://github.com/chanhee-luke/RoboSpatial-Eval/blob/master/evaluation.py
def point_in_polygon(x, y, poly):
    """
    Check if the point (x, y) lies within the polygon defined by a list of (x, y) tuples.
    Uses the ray-casting algorithm.
    """
    num = len(poly)
    inside = False
    p1x, p1y = poly[0]
    for i in range(1, num + 1):
        p2x, p2y = poly[i % num]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if p1y != p2y:
                    xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                else:
                    xinters = p1x
                if p1x == p2x or x <= xinters:
                    inside = not inside
        p1x, p1y = p2x, p2y
    return inside


def evaluate_answer(ground_truth, generated_answer):
    """
    Evaluates if the generated answer is correct based on the ground truth.
    Returns a tuple of (is_correct, is_binary_answer, parsed_answer, is_parsable).
    """
    gen_answer = generated_answer.strip().lower()
    gt_lower = ground_truth.strip().lower()

    # Check if this is a binary yes/no question
    if gt_lower in ["yes", "no"]:
        is_binary = True
        is_gt_yes = gt_lower == "yes"
        # Binary answers are always considered parsable if they contain text
        is_parsable = len(gen_answer) > 0
        if is_gt_yes:
            correct = gen_answer.startswith("yes")
        else:
            correct = gen_answer.startswith("no")
        return correct, is_binary, gen_answer, is_parsable
    else:
        # Numeric evaluation: ground_truth is a list of points defining a polygon
        is_binary = False
        parsed_answer = None
        is_parsable = False  # Default to not parsable until we successfully parse

        try:
            gt_polygon = ast.literal_eval(ground_truth)
            if not isinstance(gt_polygon, list) or len(gt_polygon) < 3:
                return False, is_binary, parsed_answer, is_parsable

            # Extract the first coordinate pair using regex
            # Look for patterns like (0.1,0.2) or (0.1, 0.2) or [0.1, 0.2] or [0.1,0.2]
            # This approach is more robust than trying to parse the entire list

            # Try to match tuple format (x,y) or (x, y)
            tuple_match = re.search(
                r"\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)", generated_answer
            )
            if tuple_match:
                try:
                    x = float(tuple_match.group(1))
                    y = float(tuple_match.group(2))
                    parsed_answer = (x, y)
                    is_parsable = True
                    correct = point_in_polygon(x, y, gt_polygon)
                    return correct, is_binary, parsed_answer, is_parsable
                except (ValueError, TypeError):
                    pass  # Continue to other formats if float conversion fails

            # Try to match list format [x,y] or [x, y]
            list_match = re.search(
                r"\[\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\]", generated_answer
            )
            if list_match:
                try:
                    x = float(list_match.group(1))
                    y = float(list_match.group(2))
                    parsed_answer = (x, y)
                    is_parsable = True
                    correct = point_in_polygon(x, y, gt_polygon)
                    return correct, is_binary, parsed_answer, is_parsable
                except (ValueError, TypeError):
                    pass  # Continue to other formats if float conversion fails

            # Fall back to the original approach but with extra safety
            try:
                # Extract the first list (text between square brackets) from generated_answer
                # Use a regex that can handle multi-line content
                match = re.search(r"\[(.*?)\]", generated_answer, re.DOTALL)
                if match is None:
                    return False, is_binary, parsed_answer, is_parsable

                # Add spaces after commas if not present (to help ast.literal_eval)
                list_content = match.group(1)
                list_content = re.sub(r",(\S)", r", \1", list_content)

                # Try to fix truncated tuples by adding closing parenthesis and brackets if needed
                list_content = list_content.strip()
                if list_content.endswith(","):
                    list_content = list_content[:-1]

                list_str = "[" + list_content + "]"

                # Try to parse the list directly
                try:
                    gen_val = ast.literal_eval(list_str)
                except (SyntaxError, ValueError):
                    # If direct parsing fails, try to extract just the first tuple
                    tuple_match = re.search(
                        r"\(\s*(\d+\.?\d*)\s*,\s*(\d+\.?\d*)\s*\)", list_content
                    )
                    if tuple_match:
                        x = float(tuple_match.group(1))
                        y = float(tuple_match.group(2))
                        parsed_answer = (x, y)
                        is_parsable = True
                        correct = point_in_polygon(x, y, gt_polygon)
                        return correct, is_binary, parsed_answer, is_parsable
                    else:
                        return False, is_binary, parsed_answer, is_parsable

                # Handle different formats for points
                if isinstance(gen_val, list):
                    if len(gen_val) == 0:
                        return False, is_binary, parsed_answer, is_parsable

                    # Case 1: The list itself is a point coordinates [x, y]
                    if len(gen_val) == 2 and all(
                        isinstance(v, (int, float)) for v in gen_val
                    ):
                        gen_point = tuple(gen_val)  # Convert [x, y] to (x, y)
                    # Case 2: The list contains points [(x, y), ...]
                    elif isinstance(gen_val[0], tuple):
                        gen_point = gen_val[0]
                    # Case 3: The list contains coordinate pairs as lists [[x, y], ...]
                    elif isinstance(gen_val[0], list) and len(gen_val[0]) == 2:
                        gen_point = tuple(gen_val[0])  # Convert [x, y] to (x, y)
                    else:
                        return False, is_binary, parsed_answer, is_parsable
                elif isinstance(gen_val, tuple):
                    gen_point = gen_val
                else:
                    return False, is_binary, parsed_answer, is_parsable

                if not (isinstance(gen_point, tuple) and len(gen_point) == 2):
                    return False, is_binary, parsed_answer, is_parsable

                x, y = float(gen_point[0]), float(gen_point[1])
                parsed_answer = (x, y)
                is_parsable = True
                correct = point_in_polygon(x, y, gt_polygon)
                return correct, is_binary, parsed_answer, is_parsable
            except Exception:
                # If all parsing attempts fail, return False
                return False, is_binary, parsed_answer, is_parsable

        except Exception as e:
            print(f"Error evaluating answer: {e}")
            return False, is_binary, parsed_answer, is_parsable


def text2pts(text, width, height):
    # Answer is in the last line
    text = text.strip().split("\n")[-1]
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)
    points = []
    for match in matches:
        vector = [float(num) for num in match.split(",")]
        if len(vector) == 2:
            x, y = vector
            points.append((int(x * width), int(y * height)))
    return points


def evaluate_answer_v2(gt, normed_answer):
    # use where2place's evaluation method
    gt_lower = gt["answer"].strip().lower()

    # Check if this is a binary yes/no question
    if gt_lower in ["yes", "no"]:
        return normed_answer.lower().startswith(gt_lower)

    mask_img = Image.open(osp.join(gt["data_root"], gt["mask_path"]))

    points = text2pts(normed_answer, width=gt["image_width"], height=gt["image_height"])
    points_array = np.array(points)
    mask = np.array(mask_img) / 255.0
    acc = 0
    if len(points) > 0:
        in_range = (
            (points_array[:, 0] >= 0)
            & (points_array[:, 0] < mask.shape[1])
            & (points_array[:, 1] >= 0)
            & (points_array[:, 1] < mask.shape[0])
        )
        acc = float(
            np.concatenate(
                [
                    mask[points_array[in_range, 1], points_array[in_range, 0]],
                    np.zeros(points_array.shape[0] - in_range.sum()),
                ]
            ).mean()
        )
    # draw_result(gt, mask_img, acc, points)
    return acc


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    results = defaultdict(
        lambda: {"num_correct": 0, "total": 0, "illformed_responses": 0}
    )
    results_v2 = defaultdict(lambda: {"num_correct": 0, "total": 0})
    num_correct = 0
    num_correct_v2 = 0
    for pred in predictions:
        question_id = str(pred["question_id"])
        gt = annotations[question_id]
        pred["raw_answer"] = pred["answer"]
        norm_pred = normalize_string(pred["answer"].strip().split("\n")[-1])
        correct, is_binary, parsed_answer, is_parsable = evaluate_answer(
            gt["answer"], norm_pred
        )
        category = gt["category"]
        if not is_parsable:
            results[category]["illformed_responses"] += 1
        results[category]["total"] += 1
        if correct:
            num_correct += 1
            results[category]["num_correct"] += 1
        pred["correct"] = correct
        pred["label"] = gt["answer"]
        pred["answer"] = norm_pred

        correct_v2 = evaluate_answer_v2(gt, norm_pred)
        num_correct_v2 += correct_v2
        results_v2[category]["total"] += 1
        results_v2[category]["num_correct"] += correct_v2

    for category, result in results.items():
        result["accuracy"] = round(result["num_correct"] / result["total"] * 100, 4)
    for category, result in results_v2.items():
        result["accuracy"] = round(result["num_correct"] / result["total"] * 100, 4)
    final_results = {}
    final_results["accuracy_ori"] = round(num_correct / len(predictions) * 100, 4)
    final_results["accuracy"] = round(num_correct_v2 / len(predictions) * 100, 4)
    final_results["results_ori"] = results
    final_results["results_v2"] = results_v2
    return final_results


def draw_result(gt: Dict, mask_img: Image, score: float, points: List[Tuple[int, int]]):
    """
    Draws the result of a prediction on an image, including a mask overlay, points, and a score.
    Parameters:
        gt (Dict): Ground truth data containing metadata such as the image path and question ID.
        mask_img (Image): Binary mask image indicating regions of interest.
        score (float): Prediction score to display on the image.
        points (List[Tuple[int, int]]): List of (x, y) coordinates to mark on the image.
    Side Effects:
        Saves the resulting image with overlays and annotations to the 'output/imgs' directory.
    """
    # For debug
    # Load the original image
    img = Image.open(osp.join(gt["data_root"], gt["img_path"]))
    img = img.convert("RGBA")

    # Convert mask to numpy array and create overlay
    mask_array = np.array(mask_img)
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]  # Take first channel if RGB
    mask_array = mask_array / 255.0  # Normalize to 0-1

    # Create semi-transparent green overlay for mask
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_array = np.array(overlay)

    # Apply green color where mask is positive
    mask_indices = mask_array > 0.5
    overlay_array[mask_indices] = [0, 255, 0, 100]  # Semi-transparent green

    overlay = Image.fromarray(overlay_array)
    img = Image.alpha_composite(img, overlay)

    # Convert back to RGB for drawing
    img = img.convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw points as red circles
    points_array = np.array(points)
    for point in points_array:
        x, y = int(point[0]), int(point[1])
        radius = 3
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill="red",
            outline="darkred",
        )

    score_text = f"Score: {score:.3f}"
    text_bbox = draw.textbbox((0, 0), score_text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    # Position text in top-left corner with padding
    text_x, text_y = 10, 10

    # Draw text background
    draw.rectangle(
        [text_x - 5, text_y - 5, text_x + text_width + 5, text_y + text_height + 5],
        fill="white",
        outline="black",
    )

    # Draw text
    draw.text((text_x, text_y), score_text, fill="black")

    output_dir = "output/imgs"
    os.makedirs(output_dir, exist_ok=True)
    img.save(osp.join(output_dir, f"{gt['question_id']}.png"))
