export CUDA_VISIBLE_DEVICES=5,6
python inference_chartqa_kollava.py \
    --model_name_or_path tabtoyou/KoLLaVA-v1.5-Synatra-7b \
    --image_path /home/aikusrv04/aiku/small_korean_vlm/data/korean_visualization_qa/val/images \
    --json_path /home/aikusrv04/aiku/small_korean_vlm/data/korean_visualization_qa/val/json/annotation_reduced.json \
    --output_path /home/aikusrv04/aiku/small_korean_vlm/chartqa_eval/output \
    --start_idx 0 