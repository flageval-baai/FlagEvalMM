import os

config = dict(
    dataset_path="~/.cache/flagevalmm/datasets/CharXiv",
    split="val",
    processed_dataset_path="CharXiv",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(type="PromptTemplate", post_prompt=""),
    name="charxiv_val",
)

evaluator = dict(
    type="BaseEvaluator",
    eval_func="evaluate.py",
    use_llm_evaluator=True,
    use_cache=True,
    base_url=os.getenv("FLAGEVAL_BASE_URL"),
    api_key=os.getenv("FLAGEVAL_API_KEY"),
    eval_model_name="gpt-4o-mini-2024-07-18",
    chat_name="charxiv_val",
)
