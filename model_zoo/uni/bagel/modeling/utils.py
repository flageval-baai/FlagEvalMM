# Copyright (c) 2023 OpenGVLab
# Copyright (c) 2025 Bytedance Ltd. and/or its affiliates.
# SPDX-License-Identifier: MIT
#
# This file has been modified by ByteDance Ltd. and/or its affiliates. on 2025-05-20.
#
# Original file was released under MIT, with the full license text
# available at https://github.com/OpenGVLab/InternVL/blob/main/LICENSE.
#
# This modified file is released under the same license.

import os
from typing import TYPE_CHECKING

import yaml
from PIL import Image
from safetensors.torch import load_file
import torch
from torch.nn.attention.flex_attention import or_masks, and_masks

from modeling.autoencoder import load_ae
from modeling.bagel import SiglipVisionConfig, SiglipVisionModel, Bagel, BagelConfig, Qwen2Config, Qwen2ForCausalLM
from modeling.qwen2 import Qwen2Tokenizer

from .transforms import ImageTransform

def pil_img2rgb(image):
    if image.mode == "RGBA" or image.info.get("transparency", None) is not None:
        image = image.convert("RGBA")
        white = Image.new(mode="RGB", size=image.size, color=(255, 255, 255))
        white.paste(image, mask=image.split()[3])
        image = white
    else:
        image = image.convert("RGB")

    return image


def add_special_tokens(tokenizer):
    all_special_tokens = []
    for k, v in tokenizer.special_tokens_map.items():
        if isinstance(v, str):
            all_special_tokens.append(v)
        elif isinstance(v, list):
            all_special_tokens += v

    new_tokens = []

    if '<|im_start|>' not in all_special_tokens:
        new_tokens.append('<|im_start|>')

    if '<|im_end|>' not in all_special_tokens:
        new_tokens.append('<|im_end|>')

    if '<|vision_start|>' not in all_special_tokens:
        new_tokens.append('<|vision_start|>')

    if '<|vision_end|>' not in all_special_tokens:
        new_tokens.append('<|vision_end|>')

    num_new_tokens = tokenizer.add_tokens(new_tokens)
    bos_token_id = tokenizer.convert_tokens_to_ids('<|im_start|>')
    eos_token_id = tokenizer.convert_tokens_to_ids('<|im_end|>')
    start_of_image = tokenizer.convert_tokens_to_ids('<|vision_start|>')
    end_of_image = tokenizer.convert_tokens_to_ids('<|vision_end|>')

    new_token_ids = dict(
        bos_token_id=bos_token_id, 
        eos_token_id=eos_token_id, 
        start_of_image=start_of_image, 
        end_of_image=end_of_image, 
    )

    return tokenizer, new_token_ids, num_new_tokens


def _get_model_path(args_or_model_path) -> str:
    if isinstance(args_or_model_path, str):
        return args_or_model_path
    if hasattr(args_or_model_path, "model_path"):
        return args_or_model_path.model_path
    raise TypeError("Expected `str` model_path or an object with `.model_path`")


def _build_common_configs(model_path: str):
    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

    return llm_config, vit_config


def _build_tokenizer(model_path: str):
    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
    return tokenizer, new_token_ids


def _load_ema_weights(model, model_path: str):
    model_state_dict_path = os.path.join(model_path, "ema.safetensors")
    model_state_dict = load_file(model_state_dict_path, device="cpu")
    msg = model.load_state_dict(model_state_dict, strict=False)
    print(msg)
    del model_state_dict



def load_model_and_tokenizer(args):

    model_path = _get_model_path(args)
    llm_config, vit_config = _build_common_configs(model_path)

    config = BagelConfig(
        visual_gen=False,
        visual_und=True,
        llm_config=llm_config, 
        vit_config=vit_config,
        vit_max_num_patch_per_side=70,
        connector_act='gelu_pytorch_tanh',
    )
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)

    tokenizer, new_token_ids = _build_tokenizer(model_path)
    _load_ema_weights(model, model_path)
    model = model.cuda().eval()

    return model, tokenizer, new_token_ids


def load_gen_model_and_tokenizer(args, max_latent_size: int = 64):
    """
    Load BAGEL for generation (T2I): visual_gen=True and return VAE model as well.

    Returns:
        (gen_model, tokenizer, new_token_ids, vae_model)
    """

    model_path = _get_model_path(args)
    llm_config, vit_config = _build_common_configs(model_path)
    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))
    config = BagelConfig(
        visual_gen=True,
        visual_und=True,
        llm_config=llm_config,
        vit_config=vit_config,
        vae_config=vae_config,
        vit_max_num_patch_per_side=70,
        connector_act="gelu_pytorch_tanh",
        latent_patch_size=2,
        max_latent_size=max_latent_size,
    )

    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    gen_model = Bagel(language_model, vit_model, config)
    gen_model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config)
    tokenizer, new_token_ids = _build_tokenizer(model_path)
    _load_ema_weights(gen_model, model_path)
    gen_model = gen_model.cuda().eval()
    vae_model = vae_model.cuda().eval()

    return gen_model, tokenizer, new_token_ids, vae_model


def build_transform():
    # Prefer a stable, repo-local config file instead of depending on CWD.
    # Default path: <repo>/model_zoo/uni/bagel/example.yaml
    cfg_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "example.yaml")
    )

    data_config = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            data_config = yaml.safe_load(f) or {}

    # Backward-compatible defaults (match example.yaml)
    img_args = (data_config.get("vlm_sft", {}) or {}).get("image_transform_args", {}) or {}
    max_image_size = int(img_args.get("max_image_size", 980))
    min_image_size = int(img_args.get("min_image_size", 378))
    image_stride = int(img_args.get("image_stride", 14))
    max_pixels = int(img_args.get("max_pixels", 2_007_040))

    image_transform = ImageTransform(
        max_image_size=max_image_size,
        min_image_size=min_image_size,
        image_stride=image_stride,
        max_pixels=max_pixels,
    )

    return image_transform


def process_conversation(images, conversation):
    images = [pil_img2rgb(image) for image in images]
    return images, conversation

