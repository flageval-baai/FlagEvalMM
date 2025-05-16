config = dict(
    dataset_path="BAAI/Video-SafetyBench",
    split="harmful",
    processed_dataset_path="Video-SafetyBench",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt="",
        pre_prompt="These are frames of a video.\n",
    ),
    config=config,
    name="video_safety_harmful",
)
