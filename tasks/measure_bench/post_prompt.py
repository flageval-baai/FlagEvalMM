def is_chinese(text):
    """Check if the text contains Chinese characters"""
    for char in text:
        if "\u4e00" <= char <= "\u9fff":
            return True
    return False


def post_prompt_func(annotation: dict, **kwargs):
    question = annotation.get("question", "")

    if is_chinese(question):
        base_prompt = "请以以下格式结束回答：`最终答案：`, "
        return base_prompt + "后跟一个带着仪表读数和对应单位（如有必要）的字符串。"
    else:
        base_prompt = "Finalize your output with: `Final Answer:`, "
        return (
            base_prompt + "followed by a string representing the correct answer with"
            " the reading of the instrument and the corresponding unit"
            " (if necessary)."
        )


dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate", post_prompt=post_prompt_func),
)
