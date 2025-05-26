config = dict(
    dataset_path="/home/vlm/benchmarks/lmms-lab/textvqa/data",
    split="validation",
    processed_dataset_path="textvqa",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(type="PromptTemplate"),
    name="text_vqa",
)

evaluator = dict(type="CocoEvaluator")
