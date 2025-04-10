from typing import Optional, Callable, Union
from flagevalmm.registry import PROMPTS

cot_math_prompt = (
    "Answer the preceding question. The last line of your response should follow this format: "
    "'Answer: \\boxed{$FINAL_ANSWER}' (without quotes), where 'FINAL_ANSWER' is your choice option or conclusion "
    "based on the reasoning provided.  If it is a multiple choice question, only one letter is allowed in the \\boxed{$FINAL_ANSWER} ."
    "Think step by step logically, considering all relevant information before answering."
)

@PROMPTS.register_module()
class PromptTemplate:
    def __init__(
        self,
        *,
        pre_prompt: Optional[Union[str, Callable]] = None,
        post_prompt: Optional[Union[str, Callable]] = None,
        examples: Optional[Union[str, Callable]] = None,
        prompt_func: Optional[Callable] = None,
        use_cot_math: bool = False,
    ) -> None:
        self.pre_prompt = pre_prompt
        self.post_prompt = post_prompt
        self.examples = examples
        self.use_cot_math = use_cot_math
        self.build_prompt = (
            self.default_prompt_func if prompt_func is None else prompt_func
        )

    def get_default_post_prompt(
        self, *, question: str, question_type: str, **kwargs
    ) -> str:
        is_cn = any("\u4e00" <= char <= "\u9fff" for char in question)
        cn_instructions = {
            "multiple-choice": "从给定的选项中直接选择答案的字母，不要多余的解释。",
            "multiple-response": "请直接回答正确选项的字母，正确选项可能有多个，不要多余的解释。",
            "fill-in-the-blank": "在横线或者空白处直接填上答案，如果有多个空需要填，使用分号(;)分隔。直接给出答案，不需要多余的解释",
            "yes-no": "直接回答'对'或'错'，不需要多余的解释",
            "cloze": "在所有横线上或者文章的空白处填上答案，不需要多余的解释",
            "default": "请认真读题并回答。",
        }

        en_instructions = {
            "multiple-choice": "Answer with the option's letter from the given choices directly.",
            "multiple-response": "Answer with the option's letters from the given choices directly. There might be multiple correct choices; indicate them by listing their letters together without spaces.",
            "fill-in-the-blank": "Complete each blank with a single word or phrase directly. If there is more than one blank, separate your answers with a semicolon (;).",
            "yes-no": "Answer with 'yes' or 'no'.",
            "cloze": "Fill in the answers directly on all the horizontal lines or in the blank spaces of the article.",
            "default": "Answer the question using a single word or phrase.",
            "cot-math": cot_math_prompt,
        }

        instructions = cn_instructions if is_cn else en_instructions

        if self.use_cot_math:
            instruct = cot_math_prompt
        else:
            instruct = instructions.get(question_type, instructions["default"])

        return instruct

    def infer_prompt(
        self,
        *,
        prompt: Optional[Union[str, Callable[..., str]]] = None,
        question: str,
        question_type: str,
        **kwargs,
    ) -> str:
        if prompt is None:
            return ""
        elif isinstance(prompt, str):
            return prompt
        return prompt(question=question, question_type=question_type, **kwargs)

    def default_prompt_func(self, question: str, question_type: str, **kwargs) -> str:
        pre_prompt = self.infer_prompt(
            prompt=self.pre_prompt,
            question=question,
            question_type=question_type,
            **kwargs,
        )
        examples = self.infer_prompt(
            prompt=self.examples,
            question=question,
            question_type=question_type,
            **kwargs,
        )
        if self.post_prompt is None:
            post_prompt = self.get_default_post_prompt(
                question=question, question_type=question_type
            )
        else:
            post_prompt = self.infer_prompt(
                prompt=self.post_prompt,
                question=question,
                question_type=question_type,
                **kwargs,
            )

        prompt = f"{examples}{pre_prompt}{question}\n{post_prompt}"
        return prompt.strip()
