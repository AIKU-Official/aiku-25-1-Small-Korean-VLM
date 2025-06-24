export CUDA_VISIBLE_DEVICES=4
python inference_chartqa_onevision.py \
    --model_name_or_path llava-hf/llava-onevision-qwen2-0.5b-ov-hf \
    --image_path /home/aikusrv04/aiku/small_korean_vlm/data/korean_visualization_qa/val/images \
    --json_path /home/aikusrv04/aiku/small_korean_vlm/data/korean_visualization_qa/val/json/annotation_reduced.json \
    --output_path /home/aikusrv04/aiku/small_korean_vlm/chartqa_eval/output \
    --start_idx 0 