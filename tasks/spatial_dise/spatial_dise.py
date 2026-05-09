config = dict(
    dataset_path="TACPS-liv/Spatial-DISE",
    split="benchmark",
    processed_dataset_path="Spatial-DISE",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=(
            "Please select the correct answer and respond with only one option letter."
        ),
    ),
    config=config,
    name="spatial_dise",
)

evaluator = dict(
    type="BaseEvaluator",
    detailed_keys=["category", "difficulty", "dise_category"],
)
