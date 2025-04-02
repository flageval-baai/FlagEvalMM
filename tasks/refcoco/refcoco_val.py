config = dict(
    dataset_path="lmms-lab/RefCOCO",
    split="val",
    processed_dataset_path="RefCOCO",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate"),
    config=config,
    name="refcoco_val",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
