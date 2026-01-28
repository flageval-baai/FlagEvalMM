config = dict(
    dataset_path="FlagEval/ERQAPlus",
    split="test",
    processor="process.py",
    processed_dataset_path="ERQAPlus",
)


def post_prompt_func(annotation: dict, **kwargs):
    evaluator = annotation.get("evaluator", None)

    # English version (original)
    base_prompt = "**Finalize your output with:** `Final Answer:`, "
    cot = "Carefully analyze the question above and reason through it step by step. "

    base_prompt = cot + base_prompt

    if evaluator == "choices_matching":
        return (
            base_prompt
            + "followed by a letter or a comma-separated list of letters which means the option of correct answer.\n**Format example:** `Final Answer: A`"
        )
    elif evaluator == "ordered_list_matching":  # only add uniform suffix for char items
        return (
            base_prompt
            + "followed by a comma-separated list.\n**Format example:** `Final Answer: A, B, C, D, E`"
        )
    elif evaluator == "number_matching":
        return (
            base_prompt
            + "followed by a number.\n**Format example:** `Final Answer: 123`"
        )
    # Combination judgment
    elif evaluator == "bool_list_matching":
        return (
            base_prompt
            + "followed by a comma-separated list composed of 0 or 1.\n**Format example:**`Final Answer:[1,1,0]`"
        )
    else:
        return base_prompt + "followed by a string representing the correct answer."


dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=post_prompt_func,
    ),
    config=config,
    name="ERQAPlus",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
