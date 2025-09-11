config = dict(
    dataset_path="BAAI/ShareRobot-Bench",
    split="test",
    processed_dataset_path="share_robot/trajectory/data",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(
        type="PromptTemplate",
    ),
    name="share_robot_trajectory",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
