import os
import json
import re
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name_or_path", type=str, default="llava-hf/llava-onevision-qwen2-0.5b-ov-hf")
parser.add_argument("--image_root", type=str, required=True, help="이미지가 저장된 최상위 폴더 경로")
parser.add_argument("--json_path", type=str, required=True, help="instruction pair JSONL 파일 경로")
parser.add_argument("--output_path", type=str, required=True, help="결과 저장 경로")
parser.add_argument("--start_idx", type=int, default=0)
args = parser.parse_args()

# 1️⃣ 모델과 프로세서 로드
model = LlavaOnevisionForConditionalGeneration.from_pretrained(
    args.model_name_or_path,
    torch_dtype=torch.float16,
).to(0)
processor = AutoProcessor.from_pretrained(args.model_name_or_path, trust_remote_code=True)

# 2️⃣ JSONL 로드
json_data = []
with open(args.json_path, "r", encoding="utf-8") as f:
    for line in f:
        json_data.append(json.loads(line))

# 3️⃣ 결과 파일 미리 열기 (append 모드)
os.makedirs(args.output_path, exist_ok=True)
output_file = os.path.join(args.output_path, "ocr_multi_onevision_inference_results.jsonl")
f_out = open(output_file, "a", encoding="utf-8")

# 4️⃣ Inference 루프
start = args.start_idx
pbar = tqdm(total=len(json_data[start:]), initial=start, desc="OCR Inference (Multilingual)")

for idx, entry in enumerate(json_data[start:], start=start):
    result = {
        "id": entry["id"],
        "image": entry["image"]
    }

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

            if question.startswith("<image>"):
                question = re.sub(r"^<image>\s*\n?", "", question).lstrip()

            conversations.append({
                "from": "human",
                "value": question
            })

            conversation = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image"}
                    ]
                }
            ]

            processed_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
            inputs = processor(text=processed_prompt, images=raw_image, return_tensors="pt").to(0, torch.float16)

            try:
                output = model.generate(
                    **inputs,
                    max_new_tokens=40,
                    do_sample=False
                )
                decoded_output = processor.decode(output[0][2:], skip_special_tokens=True).strip()
                if "assistant" in decoded_output:
                    decoded_output = decoded_output.split("assistant", 1)[-1].strip()
            except Exception as e:
                print(f"❌ 모델 추론 실패 (id: {entry['id']}) - {e}")
                break

            conversations.append({
                "from": "gpt",
                "value": decoded_output
            })

            print(f"질문: {question}")
            print(f"답변: {decoded_output}")
            break  # 이미지당 질문 한 쌍만

    result["conversations"] = conversations

    # 결과 즉시 저장
    json.dump(result, f_out, ensure_ascii=False)
    f_out.write("\n")
    f_out.flush()
    pbar.update(1)

pbar.close()
f_out.close()

print(f"✅ OCR inference 결과가 저장되었습니다: {output_file}")
