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

DEFAULT_DATA_DIR: Path = "/home/guowei/FlagEvalMM/tasks/openeqa/prompts"


import requests


PROMPT_NAME_TO_PATH = {
    "mmbench": DEFAULT_DATA_DIR / Path("mmbench.txt"),
    "mmbench-extra": DEFAULT_DATA_DIR / Path("mmbench-extra.txt"),
    "blind-llm": DEFAULT_DATA_DIR / Path("blind-llm.txt"),
    "gpt4v": DEFAULT_DATA_DIR / Path("gpt4v.txt"),
    "claude3-vision": DEFAULT_DATA_DIR / Path("claude3-vision.txt"),
    "gemini-pro-vision": DEFAULT_DATA_DIR / Path("gemini-pro-vision.txt"),
}

def set_openai_key(key: Optional[str] = None):
    if key is None:
        assert "OPENAI_API_KEY" in os.environ
        key = os.environ["OPENAI_API_KEY"]
    openai.api_key = key

def load_prompt(name: str):
    if name not in PROMPT_NAME_TO_PATH:
        raise ValueError("invalid prompt: {}".format(name))
    path = PROMPT_NAME_TO_PATH[name]
    with path.open("r") as f:
        return f.read().strip()



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
    if output.isdigit():
        return int(output)
    start_idx = output.find(tag)
    if start_idx == -1:
        return str(output)
    end_idx = output.find("\n", start_idx)
    if end_idx == -1:
        return int(output[start_idx:].replace(tag, "").strip())
    return int(output[start_idx:end_idx].replace(tag, "").strip())


def get_llm_match_score(
    question: str,
    answer: str,
    prediction: str,
    extra_answers: Optional[list] = None,
    openai_key: Optional[str] = None,
    openai_model: str = "gpt-4-1106-preview",
    openai_seed: int = 1234,
    openai_max_tokens: int = 32,
    openai_temperature: float = 0.2,
    openai_top_p: float = 0.95,
    verbose: bool = False,
    endpoint: bool = False,
):
    if prediction is None:
        return 0
    
    prompt_name = "mmbench" if extra_answers is None else "mmbench-extra"
    prompt = load_prompt(prompt_name)

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
                    answer=answer,
                    prediction=prediction,
                    extra_answers=extra_answers,
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



SAVE_INTERVAL = 30
SAVE_PATH = "/root/.cache/flagevalmm/results/eval_result_temp.json"

def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:
    """评估模型预测结果，支持断点恢复与分批保存"""

    # ============ 加载已有数据（恢复用） ============
    # if os.path.exists(SAVE_PATH):
    #     with open(SAVE_PATH, "r") as f:
    #         temp_data = json.load(f)
    #     processed_ids = set([item["question_id"] for item in temp_data])
    #     print(f"恢复执行：已处理 {len(processed_ids)} 条")
    # else:
    temp_data = []
    processed_ids = set()

    scores_openeqa_hm3d = []
    scores_openeqa_scannet = []
    error_data = []

    batch_count = 0

    for i, pred in enumerate(tqdm(predictions)):
        question_id = str(pred["question_id"])
        if question_id in processed_ids:
            continue

        prediction = pred["answer"]
        sample = annotations[question_id]
        question = sample["question"]

        if isinstance(sample["answer"], str):
            gt = sample["answer"].split("[SEG]")
        else:
            gt = str(sample['answer'])

        answer = gt[0]
        try:
            score = get_llm_match_score(question, answer, prediction, endpoint=True)
        except Exception as e:
            print(e)
            error_data.append(sample)
            continue

        if isinstance(score, str):
            print("*" * 40)
            print("Item ID:    {}".format(i))
            print("Example question:    {}".format(question))
            print("Ground-truth answer: {}".format(answer))
            print("Predicted answer:    {}".format(prediction))
            error_data.append(sample)
        else:
            llm_score = (score - 1) / 4
            sample["llm_score"] = llm_score
            sample["question_id"] = question_id  # 用于恢复
            print("*" * 40)
            print("Item ID:    {}".format(i))
            print("Example question:    {}".format(question))
            print("Ground-truth answer: {}".format(answer))
            print("Predicted answer:    {}".format(prediction))
            print("LLM-match score:     {}".format(llm_score))

            temp_data.append(sample)

            if sample["type_level_2"] == "openeqa_hm3d-v0":
                scores_openeqa_hm3d.append(llm_score)
            elif sample["type_level_2"] == "openeqa_scannet-v0":
                scores_openeqa_scannet.append(llm_score)

            batch_count += 1

            # ============ 每隔 SAVE_INTERVAL 保存一次 ============
            # if batch_count >= SAVE_INTERVAL:
            #     with open(SAVE_PATH, "w") as f:
            #         json.dump(temp_data, f, indent=2)
            #     print(f"已保存 {len(temp_data)} 条中间结果到 {SAVE_PATH}")
            #     batch_count = 0

    # ============ 全部完成后保存 ============
    with open(SAVE_PATH, "w") as f:
        json.dump(temp_data, f, indent=2)
    print(f"✅ 所有结果保存完成，共 {len(temp_data)} 条")

    # ============ 计算最终结果 ============
    scores_hm3d = sum(scores_openeqa_hm3d) / len(scores_openeqa_hm3d) if scores_openeqa_hm3d else 0
    scores_scannet = sum(scores_openeqa_scannet) / len(scores_openeqa_scannet) if scores_openeqa_scannet else 0

    result = {
        "openeqa_hm3d": scores_hm3d * 100,
        "openeqa_scannet": scores_scannet * 100
    }

    return result

