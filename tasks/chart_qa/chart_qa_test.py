task_name = "vqa"

config = dict(
    dataset_path="/home/vlm/benchmarks/lmms-lab/ChartQA",
    split="test",
    processed_dataset_path="ChartQA",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(type="PromptTemplate"),
    name="chart_qa",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
