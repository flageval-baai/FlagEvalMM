config = dict(
    dataset_path="/share/projset/mmdataset/MathVista",
    split="testmini",
    processed_dataset_path="MathVista",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    anno_file="testmini_converted.json",
    data_root="/share/projset/mmdataset/MathVista",
    config=config,
    prompt_template=dict(type="PromptTemplate", use_cot_math=True),
    # prompt_template=dict(type="PromptTemplate"),
    name="math_vista",
)

# evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
evaluator = dict(type="BoxEvaluator")