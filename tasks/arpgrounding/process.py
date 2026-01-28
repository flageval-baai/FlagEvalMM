import json
import os
import os.path as osp
from datasets import load_dataset

BBOX_PROMPT = (
    "Please generate a set of bounding box (bbox) coordinates based on the image and description."
    "The bbox coordinate format is (top-left x, top-left y, bottom-right x, bottom-right y)."
    "All values must be floating-point numbers between 0 and 1, inclusive."
    "For example, a valid bbox might be (0.1, 0.2, 0.5, 0.6)."
)


def process(cfg):
    """
    Expect HF dataset rows like:
      - image_id: int/str
      - image: PIL Image (datasets Image feature)
      - pairs: list[dict] with keys pos_phrase, pos_bbox_xywh (and optional neg_phrase/neg_bbox_xywh)
    """

    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")

    # build output path (same style as RefCOCO)
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    img_dir = osp.join(output_dir, "img")
    os.makedirs(img_dir, exist_ok=True)

    # load dataset (handle optional "name")
    if name:
        data = load_dataset(data_dir, name=name, split=split)
    else:
        data = load_dataset(data_dir, split=split)

    content = []
    for row in data:
        image_id = row["image_id"]
        image = row["image"].convert("RGB")
        w, h = image.size

        # save image into cache
        img_name = f"img/{image_id}.jpg"
        image_save_path = osp.join(output_dir, img_name)
        if not osp.exists(image_save_path):
            image.save(image_save_path)

        # explode pairs -> samples
        pairs = row.get("pairs", [])
        for i, pair in enumerate(pairs):
            pos_phrase = str(pair["pos_phrase"]).lower()

            x, y, bw, bh = pair["pos_bbox_xywh"]
            x1, y1, x2, y2 = x, y, x + bw, y + bh

            # clamp to image bounds
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            # ensure valid box (avoid negative area)
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            pos_bbox = [
                round(x1 / w, 4),
                round(y1 / h, 4),
                round(x2 / w, 4),
                round(y2 / h, 4),
            ]

            x, y, bw, bh = pair["neg_bbox_xywh"]
            x1, y1, x2, y2 = x, y, x + bw, y + bh

            # clamp to image bounds
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            # ensure valid box (avoid negative area)
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1

            neg_bbox = [
                round(x1 / w, 4),
                round(y1 / h, 4),
                round(x2 / w, 4),
                round(y2 / h, 4),
            ]
            item = {
                "question_id": f"{split}_{image_id}_{i}",
                "img_path": img_name,  # relative path like RefCOCO
                "question": f"{BBOX_PROMPT} Description: {pos_phrase}",
                "pos_phrase": pos_phrase,
                "question_type": "bbox",
                "answer": pos_bbox,
                "image_width": w,
                "image_height": h,
            }

            # optional (won't affect bbox eval)
            if "neg_phrase" in pair:
                item["neg_phrase"] = str(pair["neg_phrase"]).lower()
                item["neg_answer"] = neg_bbox

            content.append(item)

    # save data
    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(content)} items. Data saved to {output_file}")
