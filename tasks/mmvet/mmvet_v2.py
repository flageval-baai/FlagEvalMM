import os

config = dict(
    dataset_path="whyu/mm-vet-v2",
    processed_dataset_path="mm-vet-v2",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate", post_prompt=""),
    name="mmvet_v2",
    config=config,
)

evaluator = dict(
    type="BaseEvaluator",
    eval_func="evaluate.py",
    use_llm_evaluator=True,
    use_cache=True,
    base_url=os.getenv("FLAGEVAL_BASE_URL"),
    api_key=os.getenv("FLAGEVAL_API_KEY"),
    model_name="gpt-4o-mini-2024-07-18",
    chat_name="mmvet_v2_eval",
)
