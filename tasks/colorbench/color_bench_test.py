config = dict(
    dataset_path="umd-zhou-lab/ColorBench",
    split="test",
    processed_dataset_path="ColorBench",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate"),
    config=config,
    name="color_bench_test",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
