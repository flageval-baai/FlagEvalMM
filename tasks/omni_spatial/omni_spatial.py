config = dict(
    dataset_path="qizekun/OmniSpatial",
    split="train",
    anno_file="data.json",
    processed_dataset_path="OmniSpatial",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate", post_prompt="Your answer must be clear and accurate."
    ),
    config=config,
    name="omni_spatial",
)

evaluator = dict(type="BaseEvaluator")
