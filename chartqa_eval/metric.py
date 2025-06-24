import json
import nltk
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from collections import defaultdict
from tqdm import tqdm

nltk.download('wordnet')  # for METEOR
nltk.download('omw-1.4')

def compute_metrics(reference, hypothesis):
    # 문자열 토큰화
    ref_tokens = reference.strip().split()
    hyp_tokens = hypothesis.strip().split()

    # BLEU-1 ~ BLEU-4
    ref = [ref_tokens]
    smooth_fn = SmoothingFunction().method1
    bleu_scores = [
        sentence_bleu(ref, hyp_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth_fn),
        sentence_bleu(ref, hyp_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth_fn),
        sentence_bleu(ref, hyp_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth_fn),
        sentence_bleu(ref, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth_fn)
    ]

    # METEOR (token list 기반)
    meteor = nltk.translate.meteor_score.single_meteor_score(ref_tokens, hyp_tokens)

    # ROUGE
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_l = rouge.score(reference, hypothesis)['rougeL'].fmeasure

    return {
        "BLEU-1": bleu_scores[0],
        "BLEU-2": bleu_scores[1],
        "BLEU-3": bleu_scores[2],
        "BLEU-4": bleu_scores[3],
        "METEOR": meteor,
        "ROUGE-L": rouge_l
    }


with open('/home/aikusrv04/aiku/small_korean_vlm/chartqa_eval/output/tinyllava_chartqa.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# metric 저장
scores = defaultdict(list)

for item in tqdm(data):
    conversations = item['conversations']
    for i in range(len(conversations)):
        if conversations[i]['from'] == 'gpt':
            gt_answer = conversations[i]['value'].strip()
            pred = conversations[i + 1]['value'].strip().split("\n")[0]  # 첫 번째 pred만 사용
            metrics = compute_metrics(gt_answer, pred)
            for key, val in metrics.items():
                scores[key].append(val)

# 평균 결과 출력
avg_scores = {key: sum(vals)/len(vals) for key, vals in scores.items()}
print("=== 평균 메트릭 ===")
for k, v in avg_scores.items():
    print(f"{k}: {v:.4f}")
