
# Instructions for running the evaluation command


```bash
flagevalmm --tasks tasks/retrieval/msrvtt/msrvtt_test.py \
  --exec model_zoo/retrieval/clip2video/model_adapter.py \ 
  --model openai/clip-vit-base-patch32 \
  --output-dir ./results/clip2video
```
#  Performance results for CLIP2Video

| Protocol | R@1 | R@5 | R@10 | Median Rank | Mean Rank |
|----------|-----|-----|------|-------------|-----------|
| MSRVTT | 45.6 | 72.6 | 81.7 | 2 | 14.6 |

