config = dict(
    dataset_path="AI4Math/MathVista",
    split="testmini",
    processed_dataset_path="MathVista",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    anno_file="testmini_converted.json",
    data_root="AI4Math/MathVista",
    config=config,
    prompt_template=dict(type="PromptTemplate"),
    name="math_vista",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
