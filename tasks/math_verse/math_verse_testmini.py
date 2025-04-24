import os

config = dict(
    dataset_path="AI4Math/MathVerse",
    split="testmini",
    dataset_name="testmini",
    processed_dataset_path="MathVerse",
    processor="process.py",
)


def pre_prompt(question_type: str, **kwargs) -> str:
    if question_type == "multi-choice":
        prompt = "According to the question shown in the image, please directly answer the question and provide the correct option letter, e.g., A, B, C, D.\nQuestion: "
    else:
        prompt = "According to the question shown in the image, please directly answer the question and provide the final value, e.g., 1, 2.5, 300.\nQuestion: "
    return prompt


dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(type="PromptTemplate", pre_prompt=pre_prompt, post_prompt=""),
    name="math_verse_testmini",
)

evaluator = dict(
    type="BaseEvaluator",
    eval_func="evaluate.py",
    use_llm_evaluator=True,
    use_cache=True,
    base_url=os.getenv("FLAGEVAL_BASE_URL"),
    api_key=os.getenv("FLAGEVAL_API_KEY"),
    model_name="gpt-4o-mini-2024-07-18",
    chat_name="math_verse_eval",
)
