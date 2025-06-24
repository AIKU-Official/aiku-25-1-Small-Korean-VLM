# metric.py (JSON 저장 기능 추가 버전)

import json
import argparse
import re
import datetime
from typing import Dict, Any, Optional

def find_answer_from_conversation(item: Dict[str, Any], source: str) -> Optional[str]:
    """대화 목록에서 특정 출처('gpt' 또는 'pred')의 답변을 찾습니다."""
    for turn in item.get('conversations', []):
        if turn.get('from') == source:
            return turn.get('value')
    return None

def extract_choice(text: str) -> Optional[str]:
    """주어진 텍스트에서 알파벳 선택지(A, B, C, D)를 추출합니다."""
    if not text:
        return None
    
    text_upper = text.upper()
    match = re.search(r'\b([A-D])\b', text_upper)
    if match:
        return match.group(1)
        
    return None

def main():
    """메인 평가 함수"""
    parser = argparse.ArgumentParser(
        description="Evaluate the model's predictions and save the results to a JSON file."
    )
    parser.add_argument(
        "prediction_file",
        help="Path to the JSON file containing model predictions ('pred') and ground truth ('gpt')."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Unknown/Not Specified",
        help="Name of the model being evaluated. This will be saved in the output JSON."
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional path to save the evaluation results as a JSON file."
    )
    parser.add_argument(
        "--show_errors",
        type=int,
        default=3,
        help="Number of incorrect prediction examples to display. Set to 0 to hide."
    )
    args = parser.parse_args()

    try:
        with open(args.prediction_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다 -> {args.prediction_file}")
        return
    except json.JSONDecodeError:
        print(f"오류: JSON 파일 형식이 올바르지 않습니다 -> {args.prediction_file}")
        return

    correct_count = 0
    total_count = 0
    incorrect_examples = []

    for item in data:
        truth_answer_raw = find_answer_from_conversation(item, 'gpt')
        pred_answer_raw = find_answer_from_conversation(item, 'pred')

        if truth_answer_raw is None or pred_answer_raw is None:
            continue

        total_count += 1

        truth_choice = extract_choice(truth_answer_raw)
        pred_choice = extract_choice(pred_answer_raw)

        if truth_choice is not None and truth_choice == pred_choice:
            correct_count += 1
        else:
            incorrect_info = {
                "id": item.get('id', 'N/A'),
                "ground_truth": truth_answer_raw,
                "prediction": pred_answer_raw,
                "parsed_truth": truth_choice,
                "parsed_pred": pred_choice,
            }
            incorrect_examples.append(incorrect_info)

    # --- 평가 지표 계산 ---
    if total_count > 0:
        accuracy = (correct_count / total_count) * 100
        incorrect_count = total_count - correct_count
    else:
        accuracy = 0
        incorrect_count = 0

    # --- 결과 요약 데이터 생성 ---
    results_summary = {
        "model_name": args.model_name,
        "evaluation_file": args.prediction_file,
        "evaluation_timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "metrics": {
            "total_items": total_count,
            "correct_items": correct_count,
            "incorrect_items": incorrect_count,
            "accuracy": round(accuracy, 2)
        },
        "incorrect_examples": incorrect_examples
    }

    # --- 최종 결과 콘솔 출력 ---
    print("\n" + "="*25)
    print("      평가 결과")
    print("="*25 + "\n")

    if total_count == 0:
        print("평가할 항목을 찾을 수 없습니다. JSON 파일 내용을 확인해 주세요.")
    else:
        print(f"사용된 모델: {results_summary['model_name']}")
        print(f"전체 항목: {results_summary['metrics']['total_items']}개")
        print(f"정답 개수: {results_summary['metrics']['correct_items']}개")
        print(f"오답 개수: {results_summary['metrics']['incorrect_items']}개")
        print(f"정 확 도: {results_summary['metrics']['accuracy']}%")

    if args.show_errors > 0 and incorrect_examples:
        print("\n" + "-"*25)
        print(f"      오답 예시 (상위 {min(args.show_errors, len(incorrect_examples))}개)")
        print("-"*25 + "\n")
        
        for i, ex in enumerate(incorrect_examples[:args.show_errors]):
            print(f"예시 {i+1} (ID: {ex['id']})")
            print(f"  - 정답: '{ex['ground_truth']}' (추출: {ex['parsed_truth']})")
            print(f"  - 예측: '{ex['prediction']}' (추출: {ex['parsed_pred']})")
            print("-" * 15)
            
    # --- 결과 파일 저장 ---
    if args.output_file:
        try:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(results_summary, f, ensure_ascii=False, indent=4)
            print(f"\n 평가 결과를 다음 파일에 저장했습니다: {args.output_file}")
        except IOError as e:
            print(f"\n오류: 결과를 파일에 저장하는 데 실패했습니다. {e}")

if __name__ == "__main__":
    main()