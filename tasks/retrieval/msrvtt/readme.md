
# Instructions for running the evaluation command


```bash
flagevalmm --tasks tasks/retrieval/msrvtt/msrvtt_test.py \
  --exec model_zoo/retrieval/clip2video/model_adapter.py \ 
  --model openai/clip-vit-base-patch32 \
  --output-dir ./results/clip2video
```
#  Performance results for CLIP2Video

| Protocol | R@1  | R@5  | R@10 |
|----------|------|------|------|
| v2t      | 10.1 | 20.8 | 27.8 |


| Protocol | R@1   | R@5   | R@10  |
|----------|-------|-------|-------|
| t2v      | 6.82  | 16.50 | 21.64 |


