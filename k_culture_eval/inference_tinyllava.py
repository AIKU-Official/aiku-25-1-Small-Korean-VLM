# inference_tinyllava.py

import argparse
import torch
import json
import os
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration, LlavaProcessor
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="TinyLLaVA 모델 추론을 실행합니다.")
    parser.add_argument("--model_path", type=str, default="bczhou/tiny-llava-v1-hf", help="Hugging Face 모델 경로")
    parser.add_argument("--image_dir", type=str, required=True, help="이미지가 저장된 기본 디렉토리 경로")
    parser.add_argument("--question_file", type=str, required=True, help="질문이 담긴 입력 JSON 파일 경로")
    parser.add_argument("--output_file", type=str, required=True, help="예측 결과가 저장될 출력 JSON 파일 경로")
    parser.add_argument("--max_new_tokens", type=int, default=20, help="생성할 최대 새 토큰 수")
    return parser.parse_args()

def find_human_prompt(conversation):
    """대화 목록에서 'human'의 프롬프트를 찾습니다."""
    for turn in conversation:
        if turn['from'] == 'human':
            # 프롬프트는 <image> 플레이스홀더를 이미 포함하고 있습니다.
            return turn['value']
    return None

def main():
    """추론 프로세스를 실행하는 메인 함수입니다."""
    args = parse_args()
    model_id = args.model_path

    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    print(f"사용 가능한 장치: {device}")
    
    model = LlavaForConditionalGeneration.from_pretrained(
         model_id,
         torch_dtype=torch.float16,
         low_cpu_mem_usage=True,
    ).to(5)
    processor = LlavaProcessor.from_pretrained(model_id)


    # --- 2. 데이터 로드 ---
    print(f"질문 데이터를 다음 파일에서 로드합니다: {args.question_file}")
    with open(args.question_file, 'r', encoding='utf-8') as f:
        questions_data = json.load(f)

    # --- 3. 추론 루프 ---
    results = []
    for item in tqdm(questions_data, desc="질문 처리 중"):
        human_prompt_full = find_human_prompt(item['conversations'])
        if not human_prompt_full:
            results.append(item)
            continue

        # LLaVA 모델의 채팅 템플릿에 맞게 프롬프트를 구성합니다.
        prompt_text = human_prompt_full.replace('<image>\n', '')
        prompt = f"USER: <image>\n{prompt_text}\nASSISTANT:"

        # 이미지 전체 경로를 구성합니다.
        # JSON 내의 경로가 '/'로 시작하더라도 올바르게 처리합니다.
        image_path = item['image'].lstrip('/')
        full_image_path = os.path.join(args.image_dir, image_path)

        if not os.path.exists(full_image_path):
            print(f"경고: {full_image_path} 에서 이미지를 찾을 수 없습니다. 이 항목을 건너뜁니다.")
            results.append(item) # 예측 없이 원본 항목 추가
            continue

        raw_image = Image.open(full_image_path).convert('RGB')

        inputs = processor(text=prompt, images=raw_image, return_tensors="pt").to(device, torch.float16)
        print("input 완료")
        output = model.generate(**inputs, 
                                max_new_tokens=args.max_new_tokens, 
                                do_sample=False)
        print("output 완료")
        decoded_output = processor.decode(output[0], skip_special_tokens=True)
        
        try:
            # "ASSISTANT:" 뒤에 오는 텍스트를 답변으로 추출합니다.
            answer = decoded_output.split('ASSISTANT:')[-1].strip()
        except IndexError:
            answer = "" # 예기치 않은 출력 형식 처리

        pred_turn = {
            "from": "pred",
            "value": answer
        }

        item['conversations'].append(pred_turn)
        results.append(item)

    # --- 4. 결과 저장 ---
    print(f"결과를 다음 파일에 저장합니다: {args.output_file}")
    with open(args.output_file, 'w', encoding='utf-8') as f:
        # indent=2와 ensure_ascii=False를 사용하여 가독성이 좋고 한글이 깨지지 않는 JSON 파일을 생성합니다.
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("추론이 완료되었습니다.")

if __name__ == "__main__":
    main()