config = dict(
    dataset_path="FlagEval/SAT",
    split="test",
    processed_dataset_path="SAT",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt="Carefully analyze the multiple-choice question above and reason through it step by step. Conclude your response with a line in the following format: Answer: $LETTER (without quotes), where $LETTER is the letter of the correct choice.",
    ),
    config=config,
    name="SAT",
)

evaluator = dict(type="BaseEvaluator", detailed_keys=["sub_task"])
