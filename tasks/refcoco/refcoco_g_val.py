config = dict(
    dataset_path="lmms-lab/RefCOCOg",
    split="val",
    processed_dataset_path="RefCOCOg",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate"),
    config=config,
    name="refcoco_g_val",
)

evaluator = dict(
    type="BaseEvaluator",
    eval_func="evaluate.py",
)
