from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from flagevalmm.server import ServerDataset
from flagevalmm.models.base_model_adapter import BaseModelAdapter
from flagevalmm.models.claude import Claude
from flagevalmm.server.utils import parse_args


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info: Dict):
        model_name = task_info["model_name"]
        self.model = Claude(
            model_name=model_name,
            max_tokens=1024,
            api_key=task_info.get("api_key", None),
            use_cache=task_info.get("use_cache", False),
            max_image_size=task_info.get("max_image_size", 4 * 1024 * 1024),
            min_short_side=task_info.get("min_short_side", None),
        )

    def process_single_item(self, i):
        question_id, img_path, qs = self.dataset[i]
        print(qs)
        messages = self.model.build_message(qs, image_paths=img_path)
        try:
            result = self.model.infer(messages, max_tokens=1024, temperature=0.0)
        except Exception as e:
            result = "Error code " + str(e)
        return {"question_id": question_id, "question": qs, "answer": result}

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        self.dataset = ServerDataset(task_name, self.server_ip, self.server_port)
        results = []
        num_workers = self.task_info.get("num_workers", 1)
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_item = {
                executor.submit(self.process_single_item, i): i
                for i in range(len(self.dataset))
            }

            for future in as_completed(future_to_item):
                result = future.result()
                results.append(result)

        self.save_result(results, meta_info)


if __name__ == "__main__":
    args = parse_args()
    model_adapter = ModelAdapter(
        server_ip=args.server_ip,
        server_port=args.server_port,
        timeout=args.timeout,
        extra_cfg=args.cfg,
    )
    model_adapter.run()
