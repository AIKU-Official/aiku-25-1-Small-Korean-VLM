import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import LlavaForConditionalGeneration, LlavaProcessor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="bczhou/tiny-llava-v1-hf")
parser.add_argument("--image_root", type=str, required=True, help="데이터 루트 디렉토리 (korean_ocr_multilingual)")
parser.add_argument("--json_path", type=str, required=True, help="OCR instruction pair JSONL 파일 경로")
parser.add_argument("--output_path", type=str, required=True, help="결과 저장 경로")
parser.add_argument("--start_idx", type=int, default=0)
args = parser.parse_args()

# 모델과 프로세서 로드
model = LlavaForConditionalGeneration.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
).to(0)
processor = LlavaProcessor.from_pretrained(args.model_name_or_path)

# JSONL 데이터 로드
json_data = []
with open(args.json_path, "r", encoding="utf-8") as f:
    for line in f:
        json_data.append(json.loads(line))

# Inference
all_results = []
start = args.start_idx
pbar = tqdm(total=len(json_data[start:]), initial=start, desc="OCR Inference (Multilingual)")

for idx, entry in enumerate(json_data[start:], start=start):
    result = {
        "id": entry["id"],
        "image": entry["image"]
    }

    # 이미지 경로 구성
    image_name = os.path.basename(entry["image"])
    image_full_path = os.path.join(args.image_root, "Validation/01.원천데이터", image_name)
    
    if not os.path.exists(image_full_path):
        print(f"⚠️ 이미지 로드 실패: {image_full_path}")
        continue

    try:
        raw_image = Image.open(image_full_path).convert("RGB")
    except Exception as e:
        print(f"⚠️ 이미지 로드 실패: {image_full_path} - {e}")
        continue

    conversations = []
    for conv in entry["conversations"]:
        if conv["from"] == "human":
            question = conv["value"]
            # 이미지 프롬프트 추가
            if "<image>" not in question:
                prompt = f"<image>\n{question}"
            else:
                prompt = question

            # 모델 입력
            inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to(0, torch.float16)
            output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
            decoded_output = processor.decode(output[0][2:], skip_special_tokens=True).strip()

            conversations.append({
                "from": "human",
                "value": prompt
            })
            conversations.append({
                "from": "gpt",
                "value": decoded_output
            })

            print(f"질문: {question}")
            print(f"답변: {decoded_output}")

    result["conversations"] = conversations
    all_results.append(result)
    pbar.update(1)

pbar.close()

# JSONL로 저장
os.makedirs(args.output_path, exist_ok=True)
output_file = os.path.join(args.output_path, "ocr_multi_tiny_inference_results.jsonl")
with open(output_file, "w", encoding="utf-8") as f:
    for item in all_results:
        json.dump(item, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ OCR inference 결과가 저장되었습니다: {output_file}")
