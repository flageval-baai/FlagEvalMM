config = dict(
    dataset_path="BAAI/ShareRobot-Bench",
    split="test",
    processed_dataset_path="share_robot/affordance/data",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt="Only return the box in format: [x1, y1, x2, y2] with no other output.",
    ),
    name="share_robot_affordance",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
