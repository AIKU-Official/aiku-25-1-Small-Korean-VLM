export CUDA_VISIBLE_DEVICES=4
python inference_chartqa_tinyllava.py \
    --model_name_or_path bczhou/tiny-llava-v1-hf \
    --image_path /home/aikusrv04/aiku/small_korean_vlm/data/korean_visualization_qa/val/images \
    --json_path /home/aikusrv04/aiku/small_korean_vlm/data/korean_visualization_qa/val/json/annotation_reduced.json \
    --output_path /home/aikusrv04/aiku/small_korean_vlm/chartqa_eval/output \
    --start_idx 0 