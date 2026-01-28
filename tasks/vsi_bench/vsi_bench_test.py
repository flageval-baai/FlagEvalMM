config = dict(
    dataset_path="nyu-visionx/VSI-Bench",
    split="test",
    processed_dataset_path="VSI-Bench",
    processor="process.py",
)

MCA_QUESTION_TYPES = set(
    [
        "object_rel_direction_easy",
        "object_rel_direction_medium",
        "object_rel_direction_hard",
        "object_rel_distance",
        "route_planning",
        "obj_appearance_order",
    ]
)
NA_QUESTION_TYPES = set(
    [
        "object_abs_distance",
        "object_counting",
        "object_size_estimation",
        "room_size_estimation",
    ]
)

pre_prompt = "These are frames of a video.\n"


def post_prompt(annotation: dict, **kwargs) -> str:
    question_type = annotation.get("question_type", "")
    prompt = "Carefully analyze the question above and reason through it step by step. Conclude your response with a line in the following format:"
    if question_type in MCA_QUESTION_TYPES:
        return f"{prompt}\nAnswer: $LETTER (without quotes), where $LETTER corresponds to the correct option."
    elif question_type in NA_QUESTION_TYPES:
        return f"{prompt}\nAnswer: $NUMBER (without quotes), where $NUMBER is a number (integer or float) corresponds to the correct answer."
    else:
        raise ValueError(f"Unknown question type: {question_type}")


dataset = dict(
    type="VideoDataset",
    config=config,
    anno_file="data.json",
    prompt_template=dict(
        type="PromptTemplate", pre_prompt=pre_prompt, post_prompt=post_prompt
    ),
    name="vsi_bench_test",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
