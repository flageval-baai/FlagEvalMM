# API Models

This folder contains the API models like GPT, Claude, Gemini, etc. And it also supports OpenAI http API.

## Usage

You can quickly evaluate the API models by using the following command:

Example of evaluating gpt-style models:

```bash
flagevalmm --tasks tasks/mmmu/mmmu_val.py \
        --exec model_zoo/vlm/api_model/model_adapter.py \
        --model gpt-4o-mini \
        --num-workers 4 \
        --url https://api.openai.com/v1/chat/completions \
        --api-key $OPENAI_API_KEY \
        --output-dir ./results/gpt-4o-mini \
        --use-cache
```

Example of evaluating hunyuan-vision:

```bash
flagevalmm --tasks tasks/mmmu/mmmu_val.py \
        --exec model_zoo/vlm/api_model/model_adapter.py \
        --model hunyuan-vision \
        --model-type hunyuan \
        --num-workers 4 \
        --url hunyuan.tencentcloudapi.com \
        --output-dir ./results/hunyuan-vision \
        --use-cache \
```
