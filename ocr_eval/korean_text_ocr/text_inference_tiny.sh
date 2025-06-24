export CUDA_VISIBLE_DEVICES=3
python text_inference_tiny.py \
    --model_name_or_path bczhou/tiny-llava-v1-hf \
    --image_root /home/aikusrv04/aiku/small_korean_vlm/data/korean_text_in_the_wild_ocr/images \
    --json_path /home/aikusrv04/aiku/small_korean_vlm/data/korean_text_in_the_wild_ocr/text_bbox_output_final.json \
    --output_path /home/aikusrv04/aiku/small_korean_vlm/ocr_eval/output_bbox \
    --start_idx 0