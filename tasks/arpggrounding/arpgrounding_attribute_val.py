config = dict(
    dataset_path="duzetao/ARPGrounding",
    split="attribute",
    processed_dataset_path="ARPGrounding",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate"),
    config=config,
    name="arpgrounding_attribute_val",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
