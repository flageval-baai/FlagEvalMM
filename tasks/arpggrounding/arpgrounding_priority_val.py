config = dict(
    dataset_path="duzetao/ARPGrounding",
    split="priority",
    processed_dataset_path="ARPGrounding",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate"),
    config=config,
    name="arpgrounding_priority_val",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
