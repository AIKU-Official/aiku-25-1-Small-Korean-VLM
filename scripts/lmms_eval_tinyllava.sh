TASKS=("llava_in_the_wild_ko")

for TASK in "${TASKS[@]}"; do
    CUDA_VISIBLE_DEVICES=3 accelerate launch \
        --num_processes=1 \
        -m lmms_eval \
        --model llava_hf \
        --model_args pretrained="bczhou/tiny-llava-v1-hf,device_map=auto" \
        --gen_kwargs max_new_tokens=512 \
        --tasks "$TASK" \
        --batch_size 1 \
        --log_samples \
        --log_samples_suffix reproduce \
        --output_path /home/aikusrv04/aiku/small_korean_vlm/outputs
done
