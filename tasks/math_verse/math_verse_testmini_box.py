import os

config = dict(
    dataset_path="AI4Math/MathVerse",
    split="testmini",
    dataset_name="testmini",
    processed_dataset_path="MathVerse",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(
        type="PromptTemplate",
        pre_prompt="",
        post_prompt="Please solve the problem and put your answer in one '\\boxed{}'. If it is a multiple choice question, only one letter is allowed in the '\\boxed{}'.",
    ),
    name="math_verse_testmini",
)


evaluator = dict(
    type="ExtractEvaluator",
    eval_model_name="gpt-4o-mini",
    use_llm_evaluator=True,
    use_cache=True,
    base_url=os.getenv("FLAGEVAL_BASE_URL"),
    api_key=os.getenv("FLAGEVAL_API_KEY"),
)
