config = dict(
    dataset_path="MMMU/MMMU_Pro",
    dataset_name="standard (10 options)",
    split="test",
    processed_dataset_path="MMMU_Pro",
    processor="../mmmu/process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate"),
    config=config,
    name="mmmu_pro_standard_test",
)

evaluator = dict(type="MmmuEvaluator")
