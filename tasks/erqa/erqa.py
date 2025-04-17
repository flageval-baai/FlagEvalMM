config = dict(
    dataset_path="FlagEval/ERQA",
    split="test",
    processed_dataset_path="ERQA",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt="Answer the multiple-choice question above. Think through the problem step by step before providing your final answer. Your final response should end with a line in the following format: Answer: $LETTER (without quotes), where $LETTER corresponds to the correct option.",
    ),
    config=config,
    name="erqa",
)

evaluator = dict(type="BaseEvaluator")
