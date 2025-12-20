import re
import base64
import math
from typing import Union
from io import BytesIO
from PIL import Image
from flagevalmm.common.logger import get_logger

logger = get_logger(__name__)


def remove_images_symbol(text):
    pattern = r"<image\s*\d+\>"
    result = re.sub(pattern, "", text)
    return result


def encode_image_old(
    image_or_path: Union[str, Image.Image],
    max_size=None,
    min_short_side=None,
    max_long_side=None,
):
    if isinstance(image_or_path, str):
        img = Image.open(image_or_path)
    else:
        img = image_or_path
    if img.mode != "RGB":
        img = img.convert("RGB")

    format = "JPEG"
    if max_long_side is not None and (
        img.width > max_long_side or img.height > max_long_side
    ):
        scale = max_long_side / max(img.width, img.height)
        img = img.resize(
            (math.floor(img.width * scale), math.floor(img.height * scale))
        )
    if min_short_side is not None and (
        img.width < min_short_side or img.height < min_short_side
    ):
        # pad the image to the minimum size
        img_new = Image.new(
            "RGB",
            (max(min_short_side, img.width), max(min_short_side, img.height)),
            (255, 255, 255),
        )
        img_new.paste(img, (0, 0))
        img = img_new
        logger.warning(
            f"Image {image_or_path} is too small, padding to {img_new.width}x{img_new.height}"
        )
    with BytesIO() as output:
        img.save(output, format=format, quality=95)
        image_data = output.getvalue()
        base64_img = base64.b64encode(image_data).decode("utf-8")
        image_size = len(base64_img)
        if (
            max_size is not None and image_size > max_size
        ):  # Check if image size is greater than 5 MB
            scale = math.sqrt(image_size / max_size)
            new_size = (
                math.floor(img.width / scale),
                math.floor(img.height / scale),
            )
            img = img.resize(new_size)
            # Convert the image to bytes
            with BytesIO() as output2:
                img.save(output2, format=format, quality=95)
                image_data = output2.getvalue()
                base64_img = base64.b64encode(image_data).decode("utf-8")
    return base64_img


def encode_image(
    image_or_path: Union[str, Image.Image],
    max_size=None,
    min_short_side=None,
    max_long_side=None,
):
    """
    Encode image as base64 JPEG without padding or resizing.

    Notes:
    - This function intentionally does NOT apply padding (min_short_side) or resizing
      (max_long_side / max_size). The parameters are kept for backward compatibility.
    """
    if isinstance(image_or_path, str):
        img = Image.open(image_or_path)
    else:
        img = image_or_path
    if img.mode != "RGB":
        img = img.convert("RGB")

    format = "JPEG"
    with BytesIO() as output:
        img.save(output, format=format, quality=100)
        image_data = output.getvalue()
        base64_img = base64.b64encode(image_data).decode("utf-8")
    return base64_img
