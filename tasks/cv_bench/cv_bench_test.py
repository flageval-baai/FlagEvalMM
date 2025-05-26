config = dict(
    dataset_path="/share/project/benchmarks/CV-Bench",
    split="test",
    processed_dataset_path="CV-Bench",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate"),
    config=config,
    name="cv_bench_test",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
