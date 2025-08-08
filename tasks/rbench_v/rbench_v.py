import os
config = dict(
    dataset_path="R-Bench/R-Bench-V",
    split="full",
    processed_dataset_path="R-Bench-V",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate", post_prompt=''),
    config=config,
    name="R-Bench-V",
)

evaluator = dict(
    type="BaseEvaluator",
    use_cache=False,
    use_llm_evaluator=True,
    eval_func="evaluate.py",
    base_url=os.getenv("FLAGEVAL_BASE_URL"),
    api_key=os.getenv("FLAGEVAL_API_KEY"),
    model_name="gpt-4o-mini-2024-07-18",
    chat_name="rbench_v",
)