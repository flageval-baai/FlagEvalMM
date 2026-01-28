_base_ = ["mmmu_pro_standard_test.py"]

dataset = dict(
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt="Answer the preceding multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering.",
    ),
    name="mmmu_pro_standard_test_cot",
)
