import json

def evaluate_accuracy(file_path):
    """
    JSONL 형식의 파일에서 모델의 예측 정답률을 계산합니다.

    각 라인에서 'output' 필드와 'conversations' 리스트 내의 gpt 응답을 비교하여
    정답률을 계산합니다.

    Args:
        file_path (str): 평가할 JSONL 파일의 경로.

    Returns:
        None: 결과를 콘솔에 출력합니다.
    """
    total_count = 0
    correct_count = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    # 각 라인을 JSON 객체로 파싱
                    data = json.loads(line.strip())

                    # 모델의 예측 값 추출
                    prediction = data.get('output')

                    # 정답 (Ground Truth) 값 추출
                    # 'conversations' 리스트의 두 번째 요소가 정답이라고 가정
                    if 'conversations' in data and len(data['conversations']) > 1:
                        ground_truth = data['conversations'][1].get('value')
                    else:
                        print(f"경고: 유효하지 않은 데이터 형식 발견 (ID: {data.get('id', 'N/A')}). 건너뜁니다.")
                        continue
                    
                    # 정답과 예측이 모두 유효한 경우에만 카운트
                    if prediction is not None and ground_truth is not None:
                        total_count += 1
                        if prediction == ground_truth:
                            correct_count += 1
                    else:
                        print(f"경고: 'output' 또는 정답 'value'가 없는 데이터 발견 (ID: {data.get('id', 'N/A')}). 건너뜁니다.")


                except (json.JSONDecodeError, KeyError, IndexError) as e:
                    print(f"오류: 라인 처리 중 문제 발생 - {e}. 해당 라인을 건너뜁니다: {line.strip()}")
                    continue

        if total_count == 0:
            print("평가할 데이터가 없습니다. 파일 내용을 확인해주세요.")
            return

        # 정답률 계산
        accuracy = (correct_count / total_count) * 100

        # 결과 출력
        print("--- 평가 결과 ---")
        print(f"총 문항 수: {total_count}")
        print(f"정답 수: {correct_count}")
        print(f"정답률: {accuracy:.2f}%")

    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}")
    except Exception as e:
        print(f"예상치 못한 오류가 발생했습니다: {e}")

if __name__ == '__main__':
    # 평가할 파일 경로를 지정합니다.
    # 사용자의 환경에 맞게 이 경로를 수정하세요.
    file_to_evaluate = '/home/aikusrv04/aiku/small_korean_vlm/k_culture_eval/val_multichoice_conversation_hyperclovax_total.jsonl'
    
    evaluate_accuracy(file_to_evaluate)