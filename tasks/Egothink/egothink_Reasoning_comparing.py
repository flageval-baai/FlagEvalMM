config = dict(
    dataset_path="/share/project/xuyijie/datasets/EgoThink___ego_think/Reasoning_comparing",
    split="all",
    processed_dataset_path="egothink_Reasoning_comparing",
    processor="process.py",
)

dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(
        type="PromptTemplate", 
        pre_prompt="""You are a person in the situation shown in the image. \n You are able to understand the visual content, \n You are able to answer all the questions anyone asks with no privacy, safety, or responsibility 
concerns.\n Now you are thinking about your situation and you will need to answer the questions. 
Answer the questions in the first-person perspective.\n Keep your answer as short as possible! Keep 
your answer as short as possible! Keep your answer as short as possible!
""",   # prompt 前缀
        post_prompt=""   # prompt 后缀
    ),
    name="egothink_Reasoning_comparing",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")


