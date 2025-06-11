register_evaluator = {"true_evaluator.py": "TRUEEvaluator"}

config = dict(
    dataset_path="BAAI/TRUE",
    split="textvqa_edited",
    processed_dataset_path="TRUE",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate", post_prompt=""),
    config=config,
    name="true_textvqa_edited",
)

evaluator = dict(type="TRUEEvaluator")
