      
from openai import OpenAI
import random
import time
import os
import concurrent.futures
import json
from functools import partial
from typing import Dict, List, Tuple
import json
import re
from collections import defaultdict
from tqdm import tqdm


class Evaluator_Baseline():
    def __init__(self, base_url, api_key, model_type):
        # Path to benchmark
        self.base_url = base_url
        self.api_key = api_key
        self.model_type = model_type

        self.benchmark_json = "./task_subtask_test_200_per_type.json"
        self.output_json = f"./eval_{self.model_type.replace('/', '_')}_task_subtask_test_200_per_type.json"

    def run(self):
        print("==============================================================")
        print(f"Evaluating {self.model_type}")
        self.eval(self.benchmark_json, self.output_json)
        print(f"Saving evaluation result to {self.output_json}")
        print("==============================================================\n")

    def gpt4o_query(self, prompt, max_retries=5, initial_delay=3):

        base_url=self.base_url
        api_key=self.api_key
        model_version=self.model_type
        client = OpenAI(base_url=base_url, api_key=api_key)

        for attempt in range(max_retries):
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                        ],
                    },
                ]

                response = client.chat.completions.create(
                    model=model_version,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=512,
                    # extra_body={"enable_thinking": False},
                )
                return response.choices[0].message.content

            except Exception as e:
                if attempt == max_retries - 1:
                    raise Exception(
                        f"Failed after {max_retries} attempts. Last error: {str(e)}"
                    )
                delay = initial_delay * (2**attempt) + random.uniform(
                    0, 0.1 * initial_delay * (2**attempt)
                )
                time.sleep(delay)


    def process_item(self, item, progress_bar):
        try:
            formatted_prompt = item['conversations'][0]['value']
            response = self.gpt4o_query(formatted_prompt)
            new_item = {
                "id": item['id'],
                "question": item['conversations'][0]['value'],
                "answer": item['conversations'][1]['value'],
                "pred": response
            }
            progress_bar.update(1)
            return new_item, None  # Successful result
        except Exception as e:
            error_message = f"Error processing item: {item.get('id', 'Unknown')}, error: {e}\n"
            print(error_message.strip())

            with open("error.txt", "a", encoding="utf-8") as error_file:
                error_file.write(error_message)

            progress_bar.update(1)
            return None, error_message  # Return error message


    def eval(self, input_json, output_json):
        with open(input_json, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if os.path.exists(output_json):
            with open(output_json, 'r', encoding='utf-8') as f:
                data_with_pred = json.load(f)
        else:
            data_with_pred = []

        exist_id = []
        for exist_item in data_with_pred:
            exist_id.append(exist_item["id"])

        data_no_process = []
        for item_xx in data:
            if item_xx["id"] not in exist_id:
                data_no_process.append(item_xx)

        processed_count = len(data_with_pred)
        print(f"Resuming from item {processed_count}/{len(data)}...")

        total_items = len(data) - processed_count
        with tqdm(total=total_items) as progress_bar:
            # Prepare the partial function to use in ThreadPoolExecutor
            process_item_partial = partial(self.process_item, progress_bar=progress_bar)

            with concurrent.futures.ThreadPoolExecutor(max_workers=24) as executor:
                futures = []
                for i, item in enumerate(tqdm(data_no_process, total=len(data_no_process))):
                    futures.append(executor.submit(process_item_partial, item))

                for future in concurrent.futures.as_completed(futures):
                    result, error = future.result()

                    if result:
                        data_with_pred.append(result)
                    else:
                        # Log the error if needed
                        print(f"Error occurred for an item: {error}")

                    # Save checkpoint every 50 items
                    if len(data_with_pred) % 10 == 0:
                        with open(output_json, 'w', encoding='utf-8') as f:
                            json.dump(data_with_pred, f, ensure_ascii=False, indent=4)
                        # print(f"Checkpoint saved at {len(data_with_pred)} items.")

        # Final save after processing all items
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(data_with_pred, f, ensure_ascii=False, indent=4)

        print(f"Processed dataset saved to: {output_json}")




def extract_json_from_string(s):
    
    json_match = re.search(r'\{[\s\S]*\}', s)
    if json_match:
        json_str = json_match.group(0)
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            return None
    return None

def compare_answer_pred(item):
    
    answer_json = extract_json_from_string(item["answer"])
    pred_json = extract_json_from_string(item["pred"])
    
    if answer_json is None or pred_json is None:
        return 0  
    
    return 1 if answer_json == pred_json else 0

def categorize_items(data):

    categories = {
        'restaurant': [],
        'home': [],
        'supermarket': []
    }
    
    for item in data:
        item_id = str(item.get('id', '')).lower()
        if 'restaurant' in item_id:
            categories['restaurant'].append(item)
        elif 'home' in item_id:
            categories['home'].append(item)
        elif 'supermarket' in item_id:
            categories['supermarket'].append(item)
    
    return categories

def calculate_scores(data):

    categories = categorize_items(data)
    
    
    total_score = 0
    for item in data:
        total_score += compare_answer_pred(item)
    overall_avg = total_score / len(data) if data else 0.0
    

    category_scores = {}
    for category, items in categories.items():
        if not items:
            category_scores[category] = 0.0
            continue
        
        category_score = 0
        for item in items:
            category_score += compare_answer_pred(item)
        category_scores[category] = category_score / (len(items)-1)
    
    return {
        'overall_average': overall_avg,
        'category_averages': category_scores
    }




def get_result(annotations: Dict, predictions: List[Dict]) -> Dict:

    data = []
    for idx, pred in enumerate(predictions):
        question_id = str(pred["question_id"])
        gt_data = annotations[question_id]
        gt_answer = gt_data["answer"]
        data.append({
            "id": question_id,
            "question": pred["prompt"],
            "answer": gt_answer,
            "pred": pred["answer"]
        })

    with open("data.json", "w") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


    results = calculate_scores(data)
    print("Overall Average Score:", results['overall_average'])
    print("Category Averages:")
    for category, score in results['category_averages'].items():
        print(f"  {category}: {score}")
    return results

    