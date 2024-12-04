task_name = "vqa"

config = dict(
    dataset_path="BLINK-Benchmark/BLINK",
    split="val",
    processed_dataset_path="BLINK",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt="The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options.",
    ),
    name="blink_val",
)

evaluator = dict(type="BaseEvaluator", detailed_keys=["sub_task"])
