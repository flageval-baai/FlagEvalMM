config = dict(
    dataset_path="lmms-lab/RefCOCOplus",
    split="val",
    processed_dataset_path="RefCOCOplus",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate"),
    config=config,
    name="refcoco_plus_val",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
