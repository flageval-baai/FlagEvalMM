config = dict(
    dataset_path="OpenGVLab/MVBench",
    split="test",
    processed_dataset_path="MVBench",
    processor="process.py",
)


dataset = dict(
    type="VideoDataset",
    prompt_template=dict(
        type="PromptTemplate",
        pre_prompt="Carefully watch the video and pay attention to the cause and sequence of events, the detail and movement of objects, and the action and pose of persons. Based on your observations, select the best option that accurately addresses the question.",
    ),
    config=config,
    name="mvbench",
)

evaluator = dict(type="BaseEvaluator", detailed_keys=["sub_task"])
