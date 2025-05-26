# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import traceback
from collections import defaultdict
import openai
from typing import Optional
import os
import json
from tenacity import retry, stop_after_attempt, wait_random_exponential
from pathlib import Path
from tqdm import tqdm
from typing import Dict, List

import requests


def set_openai_key(key: Optional[str] = None):
    if key is None:
        assert "OPENAI_API_KEY" in os.environ
        key = os.environ["OPENAI_API_KEY"]
    openai.api_key = key


def prepare_openai_messages(content: str):
    return [{"role": "user", "content": content}]



@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def call_openai_api(
    messages: list,
    model: str = "gpt-4",
    seed: Optional[int] = None,
    max_tokens: int = 32,
    temperature: float = 0.2,
    verbose: bool = False,
):
    client = openai.OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        seed=seed,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if verbose:
        print("openai api response: {}".format(completion))
    assert len(completion.choices) == 1
    return completion.choices[0].message.content



API_KEY = "869d966045f44db6ae0b8de02f7bf776"
ENDPOINT = "https://baai-emllm-eastus2.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview"

def parse_score(output: str, tag: str = "Your mark:") -> str:
    # import pdb;pdb.set_trace()
    try:
       score=eval(output.split('\n')[-1].split(':')[-1].strip())
       return score[0][0]
    except Exception as e: 
        return 0


def get_llm_match_score(
    question: str,
    answer: str,
    prediction: str,
    extra_answers: Optional[list] = None,
    openai_key: Optional[str] = None,
    openai_model: str = "gpt-4-1106-preview",
    openai_seed: int = 1234,
    openai_max_tokens: int = 256,
    openai_temperature: float = 0.2,
    openai_top_p: float = 0.95,
    verbose: bool = False,
    endpoint: bool = False,
):
    if prediction is None:
        return 0
    
    # prompt_name = "mmbench" if extra_answers is None else "mmbench-extra"
    # prompt = load_prompt(prompt_name)
    prompt="""
Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to
the user question displayed below. Your evaluation should consider correctness and helpfulness. You will be given
a reference answer and the assistant's answer. Begin your evaluation by comparing the assistant's answer with the
reference answer. Identify and correct any mistakes. The assistant has access to an image alongwith questions but
you will not be given images. Therefore, please consider only how the answer is close to the reference answer. If
the assistant' answer is not exactly same as or similar to the answer, then he must be wrong. Be as objective as
possible. Discourage uninformative answers. Also, equally treat short and long answers and focus on the correctness
of answers. After providing your explanation, you must rate the response with either 0, 0.5 or 1 by strictly following
this format: “[[rating]]”, for example: “Rating: [[0.5]]”.\n\n[Question]\n{question}\n\n[The Start of Reference
Answer]\n{refanswer}\n[The End of Reference Answer]\n\n[The Start of Assistant's Answer]\n{answer}\n[The
End of Assistant's Answer]

"""
    if endpoint:
        # use requesets to access API
        try:
            headers = {
                "Content-Type": "application/json",
                "api-key": API_KEY,
            }

            messages = prepare_openai_messages(
                prompt.format(
                    question=question,
                    refanswer=answer,
                    answer=prediction
                ),
            )
            payload = {
                "messages": messages,
                "temperature": openai_temperature,
                "top_p": openai_top_p,
                "max_tokens": openai_max_tokens,
            }

            response = requests.post(ENDPOINT, headers=headers, json=payload)
            response.raise_for_status()  # Raise an error for bad responses
            response = response.json()
            content = response['choices'][0]['message']['content']
            # import pdb;pdb.set_trace()
            return parse_score(content)

        except Exception as e:
            traceback.print_exc()
            raise e

    else:
        # use defualt method to access API
        try:
            set_openai_key(key=openai_key)
            messages = prepare_openai_messages(
                prompt.format(
                    question=question,
                    answer=answer,
                    prediction=prediction,
                    extra_answers=extra_answers,
                ),
            )
            output = call_openai_api(
                messages=messages,
                model=openai_model,
                seed=openai_seed,
                max_tokens=openai_max_tokens,
                temperature=openai_temperature,
                verbose=verbose,
            )
            return parse_score(output)
        except Exception as e:
            traceback.print_exc()
            raise e


def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    """评估模型预测结果，支持断点恢复与分批保存"""


    error_data = []
    acc = 0
    # 计算结果
    scores=0
    batch_count = 0
    dataset_name = ""

    for i, pred in enumerate(tqdm(predictions)):
        question_id = str(pred["question_id"])

        prediction = pred["answer"]
        sample = annotations[question_id]
        question = sample["question"].split('\n')[-1]

        if isinstance(sample["answer"], str):
            gt = sample["answer"].split("[SEG]")
        else:
            gt = str(sample['answer'])
        if not dataset_name:
            dataset_name = sample["dataset_name"]
        answer = gt[0]
        try:
            score = get_llm_match_score(question, answer, prediction, endpoint=True)
        except Exception as e:
            print(e)
            error_data.append(sample)
            continue
        acc += 1
        scores += score
        # score is digit        
        sample["llm_score"] = score
    
        print("*" * 40)
        print("Item ID:    {}".format(i))
        print("Example question:    {}".format(question))
        print("Ground-truth answer: {}".format(answer))
        print("Predicted answer:    {}".format(prediction))
        print("LLM-match score:     {}".format(score))
    

    # 计算
    print(f"✅ Accuracy: {dataset_name}: scores: {100*scores/acc}")
    result = {
        "dataset_name": dataset_name,
        "scores": 100*scores/acc,
        "count": acc,
        "error_data": error_data
    }
    return result