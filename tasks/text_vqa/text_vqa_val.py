config = dict(
    dataset_path="facebook/textvqa",
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
