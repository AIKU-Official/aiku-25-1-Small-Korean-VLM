#!/bin/bash

export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export HUGGINGFACE_TOKEN="YOUR_HUGGINGFACE_TOKEN"

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed llava/train/train_xformers.py \
    --lora_enable True --lora_r 128 --lora_alpha 256 --mm_projector_lr 2e-5 \
    --deepspeed ./scripts/zero2.json \
    --model_name_or_path naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B \
    --version hyperclovax \
    --data_path /home/aikusrv04/aiku/small_korean_vlm/data/llava_instruction_tuning/final_ko_llava_instruction_tuning_korean_only.json \
    --image_folder /home/aikusrv04/aiku/small_korean_vlm/data \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --pretrain_mm_mlp_adapter /home/aikusrv04/aiku/small_korean_vlm/checkpoints/projectors/mlp_projector/mm_projector.bin \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio pad \
    --group_by_modality_length True \
    --bf16 False \
    --output_dir /home/aikusrv04/aiku/small_korean_vlm/checkpoints/llava-hyperclovax-korean-ocr-culture-augmented-korean-only \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 100 \
    --learning_rate 2e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 False \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb
