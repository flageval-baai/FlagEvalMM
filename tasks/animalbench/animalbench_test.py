config = dict(
    dataset_path="jynkris1016/Animal-Bench",
    split="test",
    processed_dataset_path="AnimalBench",
    processor="process.py",
)


dataset = dict(
    type="VideoDataset",
    prompt_template=dict(
        type="PromptTemplate",
        pre_prompt="Carefully watch the video and pay attention to the animal's behavior, actions, and interactions. Based on your observations, select the best option that accurately addresses the question.",
    ),
    config=config,
    name="animalbench",
)

evaluator = dict(type="BaseEvaluator", detailed_keys=["sub_task"])
