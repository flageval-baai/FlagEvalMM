
config = dict(
    dataset_path="RunsenXu/MMSI-Bench",
    split="test",
    processed_dataset_path="MMSI-Bench",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt="Answer with the option's letter from the given choices directly. Enclose the option's letter within ``.",
    ),
    config=config,
    name="MMSI-Bench",
)

evaluator = dict(type="BaseEvaluator")
