task_name = "vqa"

config = dict(
    dataset_path="BLINK-Benchmark/BLINK",
    split="val",
    processed_dataset_path="BLINK",
    processor="process.py",
    dataset_names=[
        "Art_Style",
        "Counting",
        "Forensic_Detection",
        "Functional_Correspondence",
        "IQ_Test",
        "Jigsaw",
        "Multi-view_Reasoning",
        "Object_Localization",
        "Relative_Depth",
        "Relative_Reflectance",
        "Semantic_Correspondence",
        "Spatial_Relation",
        "Visual_Correspondence",
        "Visual_Similarity",
    ],
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt="The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options.",
    ),
    name="blink_val",
)

evaluator = dict(type="BaseEvaluator", detailed_keys=["sub_task"])
