import json
import os
import os.path as osp

from datasets import load_dataset


def save_image(question_id, image, output_dir) -> str:
    image_path = osp.join("img", f"{question_id}.jpg")
    full_image_path = osp.join(output_dir, image_path)
    os.makedirs(osp.dirname(full_image_path), exist_ok=True)
    try:
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image.save(full_image_path)
    except Exception as e:
        print(f"Error saving image {question_id}: {e}")
    return image_path


def process(cfg):
    """Process the dataset and save it in a standard format"""
    data_dir, split = cfg.dataset_path, cfg.split
    name = cfg.get("dataset_name", "")

    # build output path
    output_dir = osp.join(cfg.processed_dataset_path, name, split)
    img_dir = osp.join(output_dir, "img")
    os.makedirs(img_dir, exist_ok=True)
    # load dataset
    data = load_dataset(data_dir, name=name, split=split)

    content = []
    for index, annotation in enumerate(data):
        question_id = index + 1
        item = {
            "question_id": question_id,
            "raw_question": annotation["question"],
            "category": annotation["catagory"],
            "question_type": "open",
            "answer": annotation["answer"],
        }
        pre_prompt = f'You are an expert assistant. '
        if annotation.get("image"):
            item["img_path"] = save_image(
                question_id, annotation["image"], output_dir
            )
            pre_prompt += "Solve the following question according to the given picture step-by-step.\n\n"
        else:
            item["img_path"] = None
            pre_prompt += "Solve the following question step-by-step.\n\n"
        item['question'] = pre_prompt + (
            "At the VERY END of your answer, output ONLY the FINAL ANSWER in this format:\n\n"
            "\\[\n\\boxed{your_final_answer_here}\n\\]\n\n"
            " You MUST put the final answer in the \\boxed{} environment.\n"
            " This applies even if the answer is a text explanation like \"The singlet state is lower in energy.\"\n"
            "Do NOT include multiple boxes.\n"
            "Do NOT include \\boxed anywhere else in your reasoning.\n"
            " The box must appear on the last line of the response.\n\n"
            "WARNING: DO NOT forget to include \\boxed{} with the final answer. Responses without it will be considered INVALID.\n\n"  # noqa: E501
            "Example:\n"
            "Question: What is the energy difference between n=2 and n=1 in hydrogen?\n"
            "Answer: The energy levels are E_n = -13.6 / n² (in eV).\n"
            "E_2 = -13.6 / 4 = -3.4 eV\n"
            "E_1 = -13.6 eV\n"
            "ΔE = 13.6 - 3.4 = 10.2 eV\n"
            "\\[\n\\boxed{10.2\\ \\text{eV}}\n\\]\n\n"
            f"Question: {annotation['question']}\nAnswer:"
        )
        content.append(item)
    output_file = osp.join(output_dir, "data.json")
    with open(output_file, "w") as f:
        json.dump(content, f, indent=2, ensure_ascii=False)

    print(f"Processed {len(content)} items. Data saved to {output_file}")