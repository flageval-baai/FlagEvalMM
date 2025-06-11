import json
import os
import os.path as osp
from datasets import load_dataset


def format_options(items):
    options = [
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
        "G",
        "H",
        "I",
        "J",
        "K",
        "L",
        "M",
        "N",
        "O",
        "P",
        "Q",
        "R",
        "S",
        "T",
        "U",
        "V",
        "W",
        "X",
        "Y",
        "Z",
    ]
    if len(items) > len(options):
        raise ValueError("Options not in A-Z. Please check the number of options. ")
    formatted_options = "\n".join(
        f"{options[i]}. {item}" for i, item in enumerate(items)
    )
    return f"Options:\n{formatted_options}"


def process(cfg):
    """Process the dataset and save it in a standard format"""
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")

    # build output path
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    cmd = f"huggingface-cli download --repo-type dataset {data_dir} --local-dir {output_dir}"
    os.system(cmd)

    # load dataset
    data = load_dataset(data_dir, name=name, split=split)
    content = []

    # process each item
    for index, annotation in enumerate(data):
        question_id = annotation["question_id"]
        # build information dictionary
        info = {
            "question_id": question_id,
            "raw_question_id": annotation["raw_question_id"],
            "raw_question": annotation["question"],
            "answer": annotation["answer"],
            "sub_task": annotation["sub_task"],
            "question_type": annotation["question_type"],
            "source": annotation["source"],
            "img_path": annotation["img_path"],
            "video_path": annotation["video_path"],
        }

        # format question
        question = annotation["question"]
        if annotation.get("options"):
            question += "\n" + format_options(annotation["options"])

        if annotation["source"] == "VSI-Bench":
            question = f"These are frames of a video.\n{question}"
            if annotation["question_type"] == "multiple-choice":
                question += (
                    "\nAnswer with the option's letter from the given choices directly."
                )
            else:
                question += (
                    "\nPlease answer the question using a single word or phrase."
                )
        else:
            question += (
                "\nThe last line of your response should be of the following format: "
                "'Answer: $LETTER' (without quotes) where LETTER is one of options."
            )
        info["question"] = question
        content.append(info)

    # save data
    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(content)} items. Data saved to {output_file}")
