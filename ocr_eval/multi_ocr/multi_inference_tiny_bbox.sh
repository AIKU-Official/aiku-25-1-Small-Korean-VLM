export CUDA_VISIBLE_DEVICES=3
python multi_inference_tiny_bbox.py \
    --model_name_or_path bczhou/tiny-llava-v1-hf \
    --image_root /home/aikusrv04/aiku/small_korean_vlm/data/korean_ocr_multilingual \
    --json_path /home/aikusrv04/aiku/small_korean_vlm/data/korean_ocr_multilingual/정리된_JSON/multi_ocr_val_bbox_final.json \
    --output_path /home/aikusrv04/aiku/small_korean_vlm/output_bbox \
    --start_idx 0
