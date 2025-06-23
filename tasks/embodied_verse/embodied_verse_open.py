config = dict(
    dataset_path="FlagEval/EmbodiedVerse-Bench",
    split="open",
    processed_dataset_path="EmbodiedVerse-Bench",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate", post_prompt=""),
    config=config,
    name="embodied_verse",
)

evaluator = dict(type="BaseEvaluator", detailed_keys=["level-1"])