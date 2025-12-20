config = dict(
    dataset_path="lmms-lab/MMMU",
    split="validation",
    processed_dataset_path="MMMU",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate", post_prompt="Let's think step by step and put the letter of your final choice after 'Answer: '"),
    config=config,
    name="mmmu_val",
)

evaluator = dict(type="MmmuEvaluator")
