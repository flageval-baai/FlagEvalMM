def is_chinese(text):
    """Check if the text contains Chinese characters"""
    for char in text:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False

def post_prompt_func(question: str, question_type: str, evaluator: str, evaluator_kwargs: dict, **kwargs):
    # Check if input is Chinese
    is_chinese_input = is_chinese(question)
    
    if evaluator == "key_items_matching":
        return ""
    
    if is_chinese_input:
        # Chinese version
        base_prompt = "**请以以下格式结束回答：** `最终答案：`, "
        if evaluator == "choices_matching":
            return base_prompt + "后跟一个或多个用逗号分隔的表示正确答案选项的字母。\n**格式示例：** `最终答案：A`"
        elif evaluator == "ordered_list_matching":
            return base_prompt + "后跟一个用逗号分隔的列表。\n**格式示例：** `最终答案：A, B, C, D, E`"
        elif evaluator == "number_matching":
            return base_prompt + "后跟一个数字。\n**格式示例：** `最终答案：123`"
        else:
            return base_prompt + "后跟一个表示正确答案的字符串。"
    else:
        # English version (original)
        base_prompt = "**Finalize your output with:** `Final Answer:`, "
        if evaluator == "choices_matching":
            return base_prompt + "followed by a letter or a comma-separated list of letters which means the option of correct answer.\n**Format example:** `Final Answer: A`"
        elif evaluator == "ordered_list_matching" and all(len(item) == 1 for item in evaluator_kwargs["order"]): # only add uniform suffix for char items
            return base_prompt + "followed by a comma-separated list.\n**Format example:** `Final Answer: A, B, C, D, E`"
        elif evaluator == "number_matching":
            return base_prompt + "followed by a number.\n**Format example:** `Final Answer: 123`"
        else:
            return base_prompt + "followed by a string representing the correct answer."

dataset = dict(
    type="VqaBaseDataset",
    prompt_template=dict(type="PromptTemplate", post_prompt=post_prompt_func),
)
