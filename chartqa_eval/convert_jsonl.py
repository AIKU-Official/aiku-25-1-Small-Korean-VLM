import json

input_path = "/home/aikusrv04/aiku/small_korean_vlm/chartqa_eval/output/tinyllava_chartqa.json"
output_path = "/home/aikusrv04/aiku/small_korean_vlm/chartqa_eval/output/tinyllava_chartqa_cleaned.jsonl"

with open(input_path, "r", encoding="utf-8") as infile:
    data = json.load(infile)

with open(output_path, "w", encoding="utf-8") as outfile:
    for entry in data:
        conv = entry["conversations"]
        image_id = entry["id"]
        image_file = entry["image"]
        
        # 3개씩 묶어서 처리 (human, gpt, pred)
        for i in range(0, len(conv), 3):
            question = conv[i]["value"]
            gpt_answer = conv[i+1]["value"]
            pred_answer = conv[i+2]["value"]

            output_obj = {
                "id": image_id,
                "image": image_file,
                "question": question,
                "gpt": gpt_answer,
                "pred": pred_answer
            }

            outfile.write(json.dumps(output_obj, ensure_ascii=False) + "\n")
