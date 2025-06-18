config = dict(
    dataset_path="lmms-lab/RealWorldQA",
    split="test",
    processed_dataset_path="RealWorldQA",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate", post_prompt=""),
    config=config,
    name="realworldqa",
)

evaluator = dict(type="BaseEvaluator")
