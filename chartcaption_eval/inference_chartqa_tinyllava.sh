export CUDA_VISIBLE_DEVICES=4
python inference_chartqa_tinyllava.py \
    --model_name_or_path bczhou/tiny-llava-v1-hf \
    --image_path /home/aikusrv04/aiku/small_korean_vlm/data/korean_caption_bench/val2014 \
    --json_path /home/aikusrv04/aiku/small_korean_vlm/data/korean_caption_bench/json/meta.json \
    --output_path /home/aikusrv04/aiku/small_korean_vlm/chartcaption_eval/output \
    --start_idx 0 