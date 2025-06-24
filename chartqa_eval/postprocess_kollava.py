import json

input_path = "/home/aikusrv04/aiku/small_korean_vlm/chartqa_eval/output/kollava_chartqa.jsonl"   # 원본 파일 경로
output_path = "/home/aikusrv04/aiku/small_korean_vlm/chartqa_eval/output/kollava_chartqa_cleaned.jsonl"  # 저장할 파일 경로

with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)
        if "output" in data:
            # '\n' 앞부분만 남기고 잘라냄
            data["output"] = data["output"].split('\n')[0]
        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
