# python eval_metric.py val_multichoice_conversation_pred.json \
# --model_name "bczhou/tiny-llava-v1-hf" \
# --output_file "evaluation_results_tinyllava.json"



python eval_metric.py val_multichoice_conversation_pred_onevision.json \
--model_name "llava-hf/llava-onevision-qwen2-0.5b-ov-hf" \
--output_file "evaluation_results_onevision.json"
