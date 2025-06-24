import json
from openai import OpenAI
from tqdm import tqdm
import os
import math

client = OpenAI(api_key="YOUR_OPENAI_API_KEY_STRING")

# 평가 프롬프트 구성
def build_prompt(question, gt_answer, prediction):
    return f"""다음은 시각 질문 응답 데이터입니다.

질문: {question}

정답 응답: {gt_answer}
모델의 예측 응답: {prediction}

다음 기준에 따라 모델 응답을 평가하세요:
- 질문을 그대로 반복하거나, 아무런 정보도 추가하지 않은 경우는 오답입니다.
- 예측이 정답과 **동일한 정보 또는 의미를 정확하게 전달**해야만 정답입니다.

정답이면 **1**, 오답이면 **0**만 출력하세요. 다른 설명은 하지 마세요.

답:"""

# GPT 호출 함수
def ask_gpt_accuracy(prompt, model="gpt-4.1-nano"):
    try:
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=5,
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"

# 데이터 불러오기
data = []
with open("/home/aikusrv04/aiku/small_korean_vlm/chartqa_eval/output/tinyllava_chartqa_cleaned.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

output_dir = "/home/aikusrv04/aiku/small_korean_vlm/chartqa_eval/eval_result/tinyllava"
os.makedirs(output_dir, exist_ok=True)

chunk_size = 100
num_chunks = math.ceil(len(data) / chunk_size)

for chunk_idx in range(14, num_chunks):
    chunk = data[chunk_idx * chunk_size : (chunk_idx + 1) * chunk_size]
    evaluated_chunk = []

    print(f"🔄 Processing chunk {chunk_idx + 1} / {num_chunks}...")

    for item in tqdm(chunk):
        question = item["question"]
        gt_answer = item["gpt"]
        prediction = item["pred"]

        prompt = build_prompt(question, gt_answer, prediction)
        score = ask_gpt_accuracy(prompt)

        item["gpt_eval"] = score
        evaluated_chunk.append(item)

    output_path = os.path.join(output_dir, f"tinyllava_eval_{chunk_idx}.jsonl")
    with open(output_path, "w", encoding="utf-8") as f:
        for r in evaluated_chunk:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ Chunk {chunk_idx} 저장 완료 → {output_path}")

print("🎉 전체 평가 완료")
