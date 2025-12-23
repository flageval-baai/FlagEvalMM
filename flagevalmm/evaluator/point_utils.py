import re
import os
import os.path as osp
import numpy as np
from PIL import Image, ImageDraw
from typing import List, Tuple, Dict


def text2pts(text: str, width: int = 640, height: int = 480) -> List[Tuple[int, int]]:
    """
    Parses point coordinates from text string.
    Supports point format (x,y) and bbox format (x0,y0,x1,y1).
    For bboxes, it returns all integer coordinates within the box.
    """
    # Use the last line of the answer
    text = text.strip().split("\n")[-1]
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, text)
    points = []

    for match in matches:
        vector = [float(num) if "." in num else int(num) for num in match.split(",")]

        # Case 1: Point (x, y)
        if len(vector) == 2:
            x, y = vector
            if isinstance(x, float) or isinstance(y, float):
                x = int(x * width)
                y = int(y * height)
            points.append((x, y))

        # Case 2: BBox (x0, y0, x1, y1) -> expand to all pixels inside
        elif len(vector) == 4:
            x0, y0, x1, y1 = vector
            if isinstance(x0, float):
                x0 = int(x0 * width)
                y0 = int(y0 * height)
                x1 = int(x1 * width)
                y1 = int(y1 * height)

            # Create a temporary mask to find all indices
            mask = np.zeros((height, width), dtype=bool)
            # Ensure coordinates are within bounds for slicing
            y0, y1 = max(0, y0), min(height, y1)
            x0, x1 = max(0, x0), min(width, x1)

            mask[y0:y1, x0:x1] = True
            y_coords, x_coords = np.where(mask)
            points.extend(list(np.stack([x_coords, y_coords], axis=1)))

    return points


def calculate_mask_score(points: List[Tuple[int, int]], mask_img: Image.Image) -> float:
    """
    Calculates the accuracy score based on whether points fall into the positive regions of the mask.
    """
    mask = np.array(mask_img)
    if len(mask.shape) == 3:
        mask = mask[:, :, 0]  # Handle RGB masks if necessary

    mask = mask / 255.0
    acc = 0.0

    if len(points) > 0:
        points_array = np.array(points)

        # Filter points that are out of image bounds
        in_range = (
            (points_array[:, 0] >= 0)
            & (points_array[:, 0] < mask.shape[1])
            & (points_array[:, 1] >= 0)
            & (points_array[:, 1] < mask.shape[0])
        )

        valid_points = points_array[in_range]
        if len(valid_points) > 0:
            # Check mask values at point coordinates. Note: mask is [y, x]
            hit_values = mask[valid_points[:, 1], valid_points[:, 0]]

            # Combine hits with misses (points out of range count as 0/miss)
            # Total score is mean of all predicted points
            total_values = np.concatenate(
                [hit_values, np.zeros(points_array.shape[0] - in_range.sum())]
            )
            acc = float(total_values.mean())
        else:
            # All points out of range
            acc = 0.0

    return acc


def text2pts_norm1k(text, width=640, height=480):
    if not text:
        return []

    line = text.strip().split("\n")[-1]
    pattern = r"\(([-+]?\d+\.?\d*(?:,\s*[-+]?\d+\.?\d*)*?)\)"
    matches = re.findall(pattern, line)
    points = []

    for match in matches:
        vector = [float(num) for num in match.split(",")]

        if len(vector) == 2:
            x, y = vector
            x = int((x / 1000.0) * width)
            y = int((y / 1000.0) * height)

            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))

            points.append((x, y))

        elif len(vector) == 4:
            x0, y0, x1, y1 = vector

            x0 = int((x0 / 1000.0) * width)
            y0 = int((y0 / 1000.0) * height)
            x1 = int((x1 / 1000.0) * width)
            y1 = int((y1 / 1000.0) * height)

            x_min = max(0, min(x0, x1, width))
            x_max = max(0, min(max(x0, x1), width))
            y_min = max(0, min(y0, y1, height))
            y_max = max(0, min(max(y0, y1), height))

            if x_max > x_min and y_max > y_min:
                mask = np.zeros((height, width), dtype=bool)
                mask[y_min:y_max, x_min:x_max] = True
                y_coords, x_coords = np.where(mask)
                points.extend(list(np.stack([x_coords, y_coords], axis=1)))

    return points


def draw_point_result(
    gt: Dict,
    mask_img: Image.Image,
    score: float,
    points: List[Tuple[int, int]],
    output_dir: str = "output/imgs",
):
    """
    Visualizes the prediction: overlays ground truth mask and draws predicted points.
    """
    img_path = osp.join(gt["data_root"], gt["img_path"])
    img = Image.open(img_path).convert("RGBA")

    # Prepare Mask Overlay
    mask_array = np.array(mask_img)
    if len(mask_array.shape) == 3:
        mask_array = mask_array[:, :, 0]
    mask_array = mask_array / 255.0

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    overlay_array = np.array(overlay)

    # Green overlay for mask
    mask_indices = mask_array > 0.5
    overlay_array[mask_indices] = [0, 255, 0, 100]

    overlay = Image.fromarray(overlay_array)
    img = Image.alpha_composite(img, overlay)
    img = img.convert("RGB")

    draw = ImageDraw.Draw(img)

    # Draw Points
    for point in points:
        x, y = int(point[0]), int(point[1])
        radius = 3
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill="red",
            outline="darkred",
        )

    # Draw Score Text
    score_text = f"Score: {score:.3f}"
    text_bbox = draw.textbbox((0, 0), score_text)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]

    text_x, text_y = 10, 10
    draw.rectangle(
        [text_x - 5, text_y - 5, text_x + text_width + 5, text_y + text_height + 5],
        fill="white",
        outline="black",
    )
    draw.text((text_x, text_y), score_text, fill="black")

    os.makedirs(output_dir, exist_ok=True)
    save_path = osp.join(output_dir, f"{gt['question_id']}.png")
    img.save(save_path)
    # print(f"Saved visualization to {save_path}")
