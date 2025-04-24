import os

config = dict(
    dataset_path="WYLing/VisualSimpleQA",
    split="data",
    processed_dataset_path="visual_simpleqa",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate", post_prompt=""),
    name="visual_simpleqa",
    config=config,
)


evaluator = dict(
    type="ExtractEvaluator",
    eval_method="simpleqa",
    eval_model_name="gpt-4o-mini-2024-07-18",
    use_llm_evaluator=True,
    use_cache=True,
    base_url=os.getenv("FLAGEVAL_BASE_URL"),
    api_key=os.getenv("FLAGEVAL_API_KEY"),
)
