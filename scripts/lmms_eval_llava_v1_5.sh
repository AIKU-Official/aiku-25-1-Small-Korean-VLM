#TASKS=("dtcbench" "mmbench_ko_dev" "seedbench_ko" "mmstar_ko" "llava_in_the_wild_ko" "pope" "gqa" "mme")
TASKS=("llava_in_the_wild_ko")

export OPENAI_API_KEY="YOUR_OPENAI_API_KEY_HERE"
for TASK in "${TASKS[@]}"; do
    echo "Running task: $TASK"
    CUDA_VISIBLE_DEVICES=2 accelerate launch \
        --num_processes=1 \
        -m lmms_eval \
        --model llava \
        --model_args pretrained="/home/aikusrv04/aiku/small_korean_vlm/checkpoints/lora_merged/llava-hyperclovax-korean-ocr-culture-augmented,model_name=llava_hyperclovax,conv_template=hyperclovax,device_map=auto" \
        --gen_kwargs max_new_tokens=512 \
        --tasks "$TASK" \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix reproduce \
        --output_path /home/aikusrv04/aiku/small_korean_vlm/outputs
done
