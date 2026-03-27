import os
import os.path as osp
import json

from datasets import load_dataset


def parse_question(question_id, question):
    """Parse question text, extracting image references.

    Returns the reformatted question and list of image paths.
    """
    splits = question.split("<IMG>")
    image_paths = []
    new_question = ""
    cnt = 0
    for split in splits:
        if split.startswith("<image_"):
            idx = split[len("<image_") : split.index(">")]
            img_filename = f"{question_id}_image_{idx}.png"
            image_paths.append(f"images/{img_filename}")
            rest = split[split.index(">") + 1 :]
            new_question += f"<image {cnt + 1}>" + rest
            cnt += 1
        else:
            new_question += split
    return new_question.strip(), image_paths


def process(cfg):
    data_dir, split = cfg.dataset_path, cfg.split
    output_dir = osp.join(cfg.processed_dataset_path, split)
    img_dir = osp.join(output_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    data = load_dataset(data_dir, split=split)
    data_reformat = []
    id_st = set()
    for d in data:
        qid = d["id"]
        question, image_paths = parse_question(qid, d["question"])
        if qid in id_st:
            print(f"Duplicate id: {qid}")

        id_st.add(qid)

        # save images
        for img_path in image_paths:
            basename = osp.splitext(osp.basename(img_path))[0]
            # basename is like "v2_0_image_0", extract the image index
            idx = basename.rsplit("_", 1)[1]
            img = d[f"image_{idx}"]
            if img is not None:
                img.save(osp.join(output_dir, img_path))

        new_item = {
            "question_id": qid,
            "question": question,
            "question_type": "open",
            "answer": d["answer"],
            "capability": d["capability"],
            "added_in": d["added_in"],
            "img_path": image_paths,
        }
        data_reformat.append(new_item)
    with open(osp.join(output_dir, "data.json"), "w") as f:
        json.dump(data_reformat, f, indent=2)
    print(f"Processed {len(data_reformat)} items. Data saved to {output_dir}")
