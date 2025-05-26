config = dict(
    dataset_path="/share/project/benchmarks/VSI-Bench",
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


def post_prompt(question_type: str, **kwargs) -> str:
    if question_type in MCA_QUESTION_TYPES:
        return "Answer with the option's letter from the given choices directly."
    elif question_type in NA_QUESTION_TYPES:
        return "Please answer the question using a single word or phrase."
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
