config = dict(
    dataset_path="lmms-lab/CMMMU",
    split="val",
    processed_dataset_path="CMMMU",
    processor="process.py",
)


def pre_prompt(annotation: dict, **kwargs) -> str:
    question_type = annotation.get("question_type", "default")
    instrution_dic = {
        "multiple-choice": "请回答以下多项选择题，并选出正确选项。这些题目可能包括单选和多选题型。如果所提供的信息不足以确定一个明确的答案，那么请根据可用的数据和你的判断来选择最可能正确的选项。",
        "yes-no": "请回答以下判断题，并根据题目描述和所给的信息来判断问题中陈述的对错。如果信息不完整或不足以作出绝对判断，请运用你的逻辑推理和现有信息来做出最可能的判断。",
        "fill-in-the-blank": "请回答以下填空题，并根据题目的要求和所提供的信息来给出最恰当的答案。如果信息不足以确切回答，那么请依据现有的数据和你的推理能力来填写最合理的答案。",
    }
    return instrution_dic[question_type] + "\n" + "问题："


dataset = dict(
    type="VqaBaseDataset",
    config=config,
    prompt_template=dict(
        type="PromptTemplate", pre_prompt=pre_prompt, post_prompt="正确答案：\n"
    ),
    name="cmmmu_val_cot",
)

evaluator = dict(type="BaseEvaluator", eval_func="evaluate.py")
