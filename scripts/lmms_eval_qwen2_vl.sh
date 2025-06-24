#TASKS=("dtcbench" "mmbench_ko_dev" "seedbench_ko" "mmstar_ko")
TASKS=("llava_in_the_wild_ko")

for TASK in "${TASKS[@]}"; do
    CUDA_VISIBLE_DEVICES=6 accelerate launch --num_processes=4 --main_process_port 12399 -m lmms_eval \
        --model=llava_onevision \
        --model_args=pretrained=lmms-lab/llava-onevision-qwen2-0.5b-si,conv_template=qwen_1_5,device_map=auto,model_name=llava_qwen \
        --tasks="$TASK" \
        --gen_kwargs max_new_tokens=512 \
        --batch_size=1 \
        --log_samples \
        --log_samples_suffix reproduce \
        --output_path /home/aikusrv04/aiku/small_korean_vlm/outputs
done
