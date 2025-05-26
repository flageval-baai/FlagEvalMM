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
        post_prompt="Answer the multiple-choice question above. Think through the problem step by step before providing your final answer. Your final response should end with a line in the following format: Answer: $LETTER (without quotes), where $LETTER corresponds to the correct option.",
    ),
    config=config,
    name="SAT",
)

evaluator = dict(type="BaseEvaluator", detailed_keys=["sub_task"])
