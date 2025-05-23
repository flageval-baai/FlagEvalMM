config = dict(
    dataset_path="FlagEval/EmbSpatial-Bench",
    split="test",
    processed_dataset_path="EmbSpatialBench",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate"),
    config=config,
    name="embspatial_bench",
)

evaluator = dict(type="BaseEvaluator", detailed_keys=["sub_task"])
