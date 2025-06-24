# inference_onevision.py (쉘 스크립트 호환 최종 버전)

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
import re
import json
import os
from tqdm import tqdm
import argparse

def find_human_prompt(conversation):
    """대화 목록에서 'human'의 프롬프트를 찾습니다."""
    for turn in conversation:
        if turn.get('from') == 'human':
            return turn.get('value')
    return None

def main():
    # 쉘 스크립트의 인자 이름과 정확히 일치하도록 수정
    parser = argparse.ArgumentParser(description="Run inference with Llava-Onevision model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained model or model identifier from Hugging Face.")
    parser.add_argument("--image_dir", type=str, required=True, help="Path to the root folder containing images.")
    parser.add_argument("--question_file", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--output_file", type=str, required=True, help="Full path to the output JSON file where results will be saved.")
    args = parser.parse_args()

    # 1. 모델 및 프로세서 설정
    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    print(f"사용 디바이스: {device}")    
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    ).to(device)
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    print(f"Model loaded on device: {device}")

    # 2. 데이터 로드
    with open(args.question_file, 'r', encoding='utf-8') as f:
        json_data = json.load(f)

    # 3. 추론 실행
    all_results = []
    for entry in tqdm(json_data, desc="Processing entries"):
        # 이미지 경로 처리 로직 수정
        # JSON 안의 상대 경로 (예: /korean_image/...)와 기본 이미지 디렉토리를 올바르게 조합
        relative_image_path = entry['image'].lstrip('/')
        image_path = os.path.join(args.image_dir, relative_image_path)
        
        try:
            raw_image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Warning: Image not found, skipping. Path: {image_path}")
            continue

        human_prompt_full = find_human_prompt(entry['conversations'])
        if not human_prompt_full:
            continue


        # --- 최종 수정된 입력 생성 로직 ---

        # 1. messages 객체 구성 (이전과 동일)
        question_text = re.sub(r"<image>\s*\n?", "", human_prompt_full).strip()
        messages = [
            {"role": "user", "content": [{"type": "text", "text": question_text}, {"type": "image"}]}
        ]

        # 2. 템플릿을 적용하여 '텍스트 프롬프트' 문자열 생성
        prompt_text_string = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # 3. 생성된 텍스트 프롬프트와 이미지를 함께 processor에 전달
        inputs = processor(
            text=prompt_text_string,  # 'messages='가 아닌 'text=' 인자를 사용
            images=raw_image,
            return_tensors="pt",
            padding=True
        ).to(device, torch.float16)

        output = model.generate(**inputs, max_new_tokens=100, do_sample=False)
        decoded_output_full = processor.batch_decode(output, skip_special_tokens=True)[0]
        
        try:
            answer = decoded_output_full.split("assistant\n")[-1].replace("<|im_end|>", "").strip()
        except IndexError:
            answer = decoded_output_full

        new_conversations = entry['conversations'].copy()
        new_conversations.append({"from": "pred", "value": answer})
        
        result = {
            "id": entry['id'],
            "image": entry['image'],
            "conversations": new_conversations
        }
        all_results.append(result)

    # 4. 최종 결과 파일로 저장 로직 수정
    # --output_file 인자로 받은 전체 경로에 바로 저장
    output_folder = os.path.dirname(args.output_file)
    if not os.path.exists(output_folder):
        os.makedirs(output_folder, exist_ok=True)
        
    with open(args.output_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    main()