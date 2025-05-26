config = dict(
    dataset_path="/home/vlm/benchmarks/CMMMU",
    split="all",
    processed_dataset_path="CMMMU",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(type="PromptTemplate"),
    name="cmmmu_all",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
