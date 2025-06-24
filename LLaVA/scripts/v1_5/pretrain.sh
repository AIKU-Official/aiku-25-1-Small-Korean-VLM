#!/bin/bash
export CUDA_HOME=/usr/local/cuda-12.1
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export HUGGINGFACE_TOKEN="YOUR_HUGGINGFACE_TOKEN"

CUDA_VISIBLE_DEVICES=0,1,2,3 deepspeed llava/train/train_xformers.py \
    --model_name_or_path naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B \
    --deepspeed ./scripts/zero2.json \
    --version plain \
    --data_path /home/aikusrv04/aiku/small_korean_vlm/data/LLaVA-CC3M-Pretrain-595K/chat.json \
    --image_folder /home/aikusrv04/aiku/small_korean_vlm/data/LLaVA-CC3M-Pretrain-595K/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type ldpnetv2 \
    --tune_mm_mlp_adapter True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --bf16 False \
    --output_dir /home/aikusrv04/aiku/small_korean_vlm/checkpoints/projectors/ldpnetv2 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 500 \
    --save_total_limit 10 \
    --learning_rate 1e-3 \
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
