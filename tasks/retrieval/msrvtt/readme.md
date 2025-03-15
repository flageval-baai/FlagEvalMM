# clip2video download path
https://drive.google.com/drive/folders/1a5Dcg8wNh88Z-bxb0ZMV3IJFjtSe7X2A
# Instructions for running the evaluation command


```bash
flagevalmm --tasks tasks/retrieval/msrvtt/msrvtt_test.py \
  --exec model_zoo/retrieval/clip2video/model_adapter.py \ 
  --model "your clip2video path" \
  --output-dir ./results/clip2video
```
#  Performance results for CLIP2Video

| Protocol | R@1 | R@5 | R@10 | Median Rank | Mean Rank |
|----------|-----|-----|------|-------------|-----------|
| MSRVTT | 45.6 | 72.6 | 81.7 | 2 | 14.6 |

