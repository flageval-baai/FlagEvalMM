config = dict(
    dataset_path="duzetao/ARPGrounding",
    split="relation",
    processed_dataset_path="ARPGrounding",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate"),
    config=config,
    name="arpgrounding_relation_val",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
