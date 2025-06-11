import os
import os.path as osp
import json
import asyncio
from typing import Dict
from flagevalmm.models import Kolors, SenseMirage, HunyuanImage, DoubaoImage, Flux
from flagevalmm.common.logger import get_logger
from flagevalmm.server.utils import get_meta, submit, get_data, parse_args
from flagevalmm.models.base_model_adapter import BaseModelAdapter

logger = get_logger(__name__)


class ImageGenerator(BaseModelAdapter):
    MODEL_MAP = {
        "kolors": Kolors,
        "sense_mirage": SenseMirage,
        "hunyuan": HunyuanImage,
        "doubao": DoubaoImage,
        "flux": Flux,
    }

    def model_init(self, task_info: Dict):
        model_name = task_info["model_name"]

        img_height = task_info.get("height", 1024)
        img_width = task_info.get("width", 1024)
        self.skip_exists = task_info.get("skip_exists", True)
        self.model_client = self.MODEL_MAP[self.get_model_type(model_name)](
            model_name, img_height, img_width
        )

    def get_model_type(self, model_name: str):
        if "flux" in model_name:
            return "flux"
        elif "doubao" in model_name:
            return "doubao"
        else:
            return model_name

    def run(self) -> None:
        for task_name in self.tasks:
            asyncio.run(self.run_one_task(task_name))
            meta_info = get_meta(task_name, self.server_ip, self.server_port)
            output_dir = meta_info["output_dir"]
            submit(
                task_name,
                self.model_client.model_name,
                self.server_ip,
                self.server_port,
                output_dir=output_dir,
            )

    async def maybe_save_image(self, response, output_dir, image_out_name):
        if response.get("img_mode", False):
            if "image" not in response:
                logger.info(f"Failed to generate image {response['reason']}")
                return False
            response["image"].save(os.path.join(output_dir, image_out_name))
            logger.info(f"Image: {image_out_name} saved")
            return True

        image_url = response["img_url"]
        if image_url is None:
            logger.info(f"Failed to generate image {response['reason']}")
            return False
        try:
            await self.model_client.download_and_save(
                image_url, os.path.join(output_dir, image_out_name)
            )
            logger.info(f"Image: {image_out_name} saved")
        except Exception as e:
            logger.info(f"Failed to download image: {e}")
            return False
        return True

    async def run_one_task(self, task_name: str) -> None:
        meta_info = get_meta(task_name, self.server_ip, self.server_port)
        text_num = meta_info["length"]
        output_dir = meta_info["output_dir"]
        os.makedirs(output_dir, exist_ok=True)
        output_info = []
        for i in range(text_num):
            response = get_data(i, task_name, self.server_ip, self.server_port)
            prompt, question_id = response["prompt"], response["id"]
            image_out_name = f"{question_id}.png"
            if self.skip_exists and osp.exists(osp.join(output_dir, image_out_name)):
                logger.info(f"Skipping: {prompt}, id: {question_id}")
                is_saved = True
            else:
                logger.info(f"Generating: {prompt}, id: {question_id}")
                response = await self.model_client.generate(prompt)
                is_saved = await self.maybe_save_image(
                    response, output_dir, image_out_name
                )
            if is_saved:
                output_info.append(
                    {"prompt": prompt, "id": question_id, "image": image_out_name}
                )

        json.dump(
            output_info,
            open(f"{output_dir}/{task_name}.json", "w"),
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    args = parse_args()
    generator = ImageGenerator(
        server_ip=args.server_ip,
        server_port=args.server_port,
        extra_cfg=args.cfg,
        local_mode=args.local_mode,
        task_names=args.tasks,
        output_dir=args.output_dir,
        model_path=args.model,
        debug=args.debug,
    )
    generator.run()
