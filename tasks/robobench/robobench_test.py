config = dict(
    dataset_path="/home/guowei/rtx_frames_success_5/gemini-002-rtx-success-planning-anno-5_reformat.json",
    split="all",
    processed_dataset_path="robobench",
    processor="process.py",
)


dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(
        type="PromptTemplate", 
        pre_prompt="""You are a robot using the joint control. 
Your goal is to infer the sequence of steps needed to complete the task described below.  
You should reason step-by-step.\n Task: """,   # prompt 前缀
        post_prompt="""Please list the steps to complete this task."""   # prompt 后缀
    ),
    name="robobench_all",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")


