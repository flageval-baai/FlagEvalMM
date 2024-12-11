from typing import Dict, Any

from flagevalmm.server import ServerDataset
from flagevalmm.models.base_model_adapter import BaseModelAdapter
from flagevalmm.models.hunyuan import Hunyuan
from flagevalmm.server.utils import parse_args, default_collate_fn


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info: Dict):
        model_name = task_info["model_name"]
        self.model = Hunyuan(
            model_name=model_name,
            max_tokens=1024,
            url=task_info.get("url", None),
            use_cache=task_info.get("use_cache", False),
        )

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        dataloader = self.create_data_loader(
            ServerDataset,
            task_name,
            collate_fn=default_collate_fn,
            batch_size=1,
            num_workers=0,
        )
        results = []

        for question_id, img_path, qs in dataloader:
            question_id = question_id[0]
            img_path_flaten = [p[0] for p in img_path]
            qs = qs[0]
            print(qs)
            messages = self.model.build_message(qs, image_paths=img_path_flaten)
            result = self.model.infer(messages)

            results.append(
                {"question_id": question_id, "question": qs, "answer": result}
            )
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
