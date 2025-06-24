#!/bin/bash

# =========================================================================
# TinyLLaVA 모델을 사용하여 VQA 데이터셋에 대한 추론을 실행하는 쉘 스크립트
# =========================================================================


ROOT_DIR="/home/aikusrv04/aiku/small_korean_vlm" 

IMAGE_DIR="${ROOT_DIR}/data"
QUESTION_FILE="${ROOT_DIR}/data/korean_image/val/val_multichoice_conversation.json"
OUTPUT_FILE="${ROOT_DIR}/k_culture_eval/val_multichoice_conversation_pred_onevision.json"

MODEL_PATH="llava-hf/llava-onevision-qwen2-0.5b-ov-hf"

# Python 스크립트 실행
python inference_onevision.py \
    --model_path "$MODEL_PATH" \
    --image_dir "$IMAGE_DIR" \
    --question_file "$QUESTION_FILE" \
    --output_file "$OUTPUT_FILE"
