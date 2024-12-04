import argparse
import re
import os
from typing import Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed
import atexit
import signal

from flagevalmm.server import ServerDataset
from flagevalmm.models.base_model_adapter import BaseModelAdapter
from flagevalmm.models.http_client import HttpClient
from flagevalmm.server.model_server import ModelServer
from flagevalmm.server.utils import get_random_port
from flagevalmm.common.logger import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Model Adapter")
    parser.add_argument(
        "--server_ip", "--server-ip", type=str, default="http://localhost"
    )
    parser.add_argument("--server_port", "--server-port", type=int, default=5000)
    parser.add_argument("--timeout", type=int, default=1000)
    parser.add_argument("--cfg", "-c", type=str, default=None)
    parser.add_argument("--num-workers", "--num-workers", type=int)
    parser.add_argument("--backend", type=str)
    return parser.parse_args()


class ModelAdapter(BaseModelAdapter):
    def __init__(self, server_ip, server_port, timeout, extra_cfg=None):
        super().__init__(
            server_ip=server_ip,
            server_port=server_port,
            timeout=timeout,
            extra_cfg=extra_cfg,
        )
        atexit.register(self.cleanup)
        signal.signal(signal.SIGTERM, self._signal_handler)
        signal.signal(signal.SIGINT, self._signal_handler)

    def _signal_handler(self, signum, frame):
        logger.info(f"Received signal {signum}, cleaning up...")
        self.cleanup()
        os._exit(0)

    def model_init(self, task_info: Dict):
        model_name = task_info["model_name"]
        if task_info.get("backend", None):
            self.model_server = self.launch_model(task_info)
        self.model = HttpClient(
            model_name=model_name,
            max_tokens=1024,
            api_key=task_info.get("api_key", None),
            url=task_info.get("url", None),
            use_cache=task_info.get("use_cache", False),
            max_image_size=task_info.get("max_image_size", 4 * 1024 * 1024),
            min_short_side=task_info.get("min_short_side", None),
            max_long_side=task_info.get("max_long_side", None),
        )

    def launch_model(self, task_info: Dict):
        port = get_random_port()
        # replace port in url
        url = re.sub(
            r":(\d+)/",
            f":{port}/",
            task_info.get("url", "http://localhost:8000/v1/chat/completions"),
        )
        task_info["url"] = url

        model_name = task_info["model_name"]
        model_server = ModelServer(
            model_name, port=port, extra_args=task_info.get("extra_args", None)
        )
        return model_server

    def process_single_item(self, i):
        question_id, img_path, qs = self.dataset[i]
        logger.info(qs)
        messages = self.model.build_message(qs, image_paths=img_path)
        try:
            result = self.model.infer(messages, max_tokens=1024, temperature=0.0)
        except Exception as e:
            result = "Error code " + str(e)
        return {"question_id": question_id, "question": qs, "answer": result}

    def cleanup(self):
        if hasattr(self, "model_server") and self.model_server is not None:
            try:
                self.model_server.stop()
                self.model_server = None
            except Exception as e:
                logger.error(f"Error shutting down model server: {e}")

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
