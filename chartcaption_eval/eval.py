import json
from konlpy.tag import Mecab
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# 형태소 분석기
mecab = Mecab()

# smoothing function to avoid zero scores for short outputs
smoothie = SmoothingFunction().method4

# 파일 경로
jsonl_path = r"/home/aikusrv04/aiku/small_korean_vlm/data/korean_caption_bench/json/filtered_ours.jsonl"

# BLEU score 저장 리스트
bleu_scores = []

# JSONL 파일 읽기
with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        data = json.loads(line)

        # 형태소 단위로 tokenizing
        references = [mecab.morphs(ref.strip()) for ref in data["caption_ko"]]
        hypothesis = mecab.morphs(data["output"].strip())

        # BLEU 계산
        score = sentence_bleu(references, hypothesis, smoothing_function=smoothie)
        bleu_scores.append(score)

# 평균 BLEU score 출력
avg_bleu = sum(bleu_scores) / len(bleu_scores)
print(f"Average BLEU score (Mecab 기반): {avg_bleu:.4f}")
