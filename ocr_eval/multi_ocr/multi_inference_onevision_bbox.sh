export CUDA_VISIBLE_DEVICES=3
python multi_inference_onevision_bbox.py \
    --model_name_or_path llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
    --image_root /home/aikusrv04/aiku/small_korean_vlm/data/korean_ocr_multilingual \
    --json_path /home/aikusrv04/aiku/small_korean_vlm/data/korean_ocr_multilingual/정리된_JSON/multi_ocr_val_bbox_final.json \
    --output_path /home/aikusrv04/aiku/small_korean_vlm/ocr_eval/output_bbox \
    --start_idx 1570
