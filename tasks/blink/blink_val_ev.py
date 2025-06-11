# Subset for EmbodiedVerse
task_name = "vqa"

config = dict(
    dataset_path="BLINK-Benchmark/BLINK",
    split="val",
    processed_dataset_path="BLINK",
    processor="process.py",
    dataset_names=[
        "Counting",
        "Relative_Depth",
        "Spatial_Relation",
        "Multi-view_Reasoning",
        "Visual_Correspondence",
    ],
    anno_file="data_ev.json",
)

dataset = dict(
    type="VqaBaseDataset",
    anno_file="data_ev.json",
    config=config,
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt="The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options.",
    ),
    name="blink_val_ev",
)

evaluator = dict(type="BaseEvaluator", detailed_keys=["sub_task"])
