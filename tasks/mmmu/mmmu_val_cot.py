_base_ = ["mmmu_val.py"]

dataset = dict(
    prompt_template=dict(
        type="PromptTemplate",
        post_prompt="Write out the multiple-choice question in the image and then solve it. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of options. Think step by step before answering.",
    ),
    name="mmmu_val_cot",
)
