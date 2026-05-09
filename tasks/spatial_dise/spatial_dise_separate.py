config = dict(
    dataset_path="TACPS-liv/Spatial-DISE",
    split="benchmark",
    processed_dataset_path="Spatial-DISE-separate",
    processor="process.py",
    image_mode="separate",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt=(
            "The answer choices are labeled A, B, C, and D. Please select the correct answer and "
            "respond with only one letter: A, B, C, or D."
        ),
    ),
    config=config,
    name="spatial_dise_separate",
)

evaluator = dict(
    type="BaseEvaluator",
    detailed_keys=["category", "difficulty", "dise_category"],
)
