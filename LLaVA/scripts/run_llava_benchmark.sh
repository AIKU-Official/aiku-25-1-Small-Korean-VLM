#!/bin/bash

CUDA_VISIBLE_DEVICES=0 python llava/eval/run_llava_benchmark.py \
    --model-path "/home/aikusrv04/aiku/small_korean_vlm/checkpoints/lora_merged/llava-hyperclovax-korean-ocr-culture-augmented" \
    --model-name "llava-hyperclovax" \
    --image-file "/home/aikusrv04/aiku/small_korean_vlm/data/korean_food/images/Img_000_0234.jpg" \
    --query "이미지 최대한 자세하게 설명해줘" \
    --conv-mode "hyperclovax" \
    --sep "," \
    --temperature 0.0 \
    --max_new_tokens 256


#    --image-root "/home/aikusrv04/aiku/small_korean_vlm/data/korean_visualization_qa/val/images" \
#    --json-file "/home/aikusrv04/aiku/small_korean_vlm/data/korean_visualization_qa/val/json/annotation_reduced_final.json" \
#    --output-file "/home/aikusrv04/aiku/small_korean_vlm/data/korean_visualization_qa/val/json/annotation_reduced_final-llava-hyperclovax-korean-ocr-culture-augmented.jsonl" \