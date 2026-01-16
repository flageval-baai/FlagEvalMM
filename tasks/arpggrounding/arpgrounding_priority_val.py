config = dict(
    dataset_path="duzetao/ARPGrounding",
    split="priority",
    processed_dataset_path="ARPGrounding",
    processor="process.py",
    # 如果你用 load_dataset(data_dir, name=...) 才需要这个
    # dataset_name="",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate"),
    config=config,
    name="arpgrounding_priority_val",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")