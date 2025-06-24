import json
import os

folder_path = "/home/aikusrv04/aiku/small_korean_vlm/chartqa_eval/eval_result/tinyllava"
file_prefix = "tinyllava_eval_"
file_suffix = ".jsonl"
file_indices = range(68)  # 0 ~ 67

total = 6785
correct = 0

for i in file_indices:
    file_path = os.path.join(folder_path, f"{file_prefix}{i}{file_suffix}")
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data.get("gpt_eval") == "1":
                correct += 1

accuracy = correct / total
print(f"Accuracy: {accuracy:.4f} ({correct}/{total})")
