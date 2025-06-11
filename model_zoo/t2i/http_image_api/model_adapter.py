import json
import os
import asyncio
import concurrent.futures
from typing import Dict, Any
from flagevalmm.server.utils import get_data, parse_args
from flagevalmm.models.http_image_client import HttpImageClient
from flagevalmm.models.base_model_adapter import BaseModelAdapter
from flagevalmm.common.logger import get_logger

logger = get_logger(__name__)


class ModelAdapter(BaseModelAdapter):
    def model_init(self, task_info: Dict):
        model_name = task_info["model_name"]
        self.model = HttpImageClient(
            model_name=model_name,
            height=task_info.get("height", 1024),
            width=task_info.get("width", 1024),
            api_key=task_info.get("api_key", None),
            url=task_info.get("url", None),
        )
        self.skip_exists = task_info.get("skip_exists", True)

    async def maybe_save_image(self, response, output_dir, image_out_name):
        image_url = response["img_url"]
        if image_url is None:
            logger.info(f"Failed to generate image {response['reason']}")
            return False
        try:
            await self.model.download_and_save(
                image_url, os.path.join(output_dir, image_out_name)
            )
            logger.info(f"Image: {image_out_name} saved")
        except Exception as e:
            logger.info(f"Failed to download image: {e}")
            return False
        return True

    async def generate_and_save(self, prompt, question_id, output_dir, image_out_name):
        if self.skip_exists and os.path.exists(
            os.path.join(output_dir, image_out_name)
        ):
            logger.info(f"Skipping: {prompt}, id: {question_id}")
            return True, prompt, question_id, image_out_name
        logger.info(f"Generating: {prompt}, id: {question_id}")
        response = await self.model.generate(prompt)
        is_saved = await self.maybe_save_image(response, output_dir, image_out_name)
        return is_saved, prompt, question_id, image_out_name

    async def process_single_item(self, i, task_name, output_dir):
        response = get_data(i, task_name, self.server_ip, self.server_port)
        prompt, question_id = response["prompt"], response["id"]
        image_out_name = f"{question_id}.png"

        return await self.generate_and_save(
            prompt, question_id, output_dir, image_out_name
        )

    def run_one_task(self, task_name: str, meta_info: Dict[str, Any]):
        text_num = meta_info["length"]
        output_dir = meta_info["output_dir"]

        output_info = []

        async def func():
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
                loop = asyncio.get_event_loop()
                futures = [
                    loop.run_in_executor(
                        executor,
                        asyncio.run,
                        self.process_single_item(i, task_name, output_dir),
                    )
                    for i in range(text_num)
                ]
                results = await asyncio.gather(*futures)

            for result in results:
                is_saved, prompt, question_id, image_out_name = result
                print("prompt", prompt)
                if is_saved:
                    output_info.append(
                        {"prompt": prompt, "id": question_id, "image": image_out_name}
                    )

            # Save the collected output information
            with open(
                f"{output_dir}/{task_name}.json", "w", encoding="utf-8"
            ) as outfile:
                json.dump(output_info, outfile, indent=2, ensure_ascii=False)

        asyncio.run(func())


if __name__ == "__main__":
    args = parse_args()
    model_adapter = ModelAdapter(
        server_ip=args.server_ip,
        server_port=args.server_port,
        timeout=args.timeout,
        extra_cfg=args.cfg,
        local_mode=args.local_mode,
        task_names=args.tasks,
        output_dir=args.output_dir,
        model_path=args.model,
        debug=args.debug,
        quiet=args.quiet,
    )
    model_adapter.run()
