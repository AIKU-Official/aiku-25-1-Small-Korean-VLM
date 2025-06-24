#!/bin/bash

SRC_DIR="/home/aikusrv04/aiku/small_korean_vlm/data/시각화 자료 질의응답 데이터"
DST_DIR="/home/aikusrv04/aiku/small_korean_vlm/data/시각화 자료 질의응답 데이터_unzipped"

mkdir -p "$DST_DIR"

# Training과 Validation 폴더를 대상으로 반복
for SPLIT in Training Validation; do
    echo "📂 [$SPLIT] 압축 해제 중..."

    # 하위 디렉토리까지 포함하여 .zip 파일 찾기
    find "$SRC_DIR/$SPLIT" -type f -name "*.zip" | while read ZIPFILE; do
        FILENAME=$(basename "$ZIPFILE".zip)
        
        # 압축을 풀 폴더 생성
        UNZIP_DIR="$DST_DIR/$SPLIT/$FILENAME"
        mkdir -p "$UNZIP_DIR"

        # 압축 해제
        unzip -q "$ZIPFILE" -d "$UNZIP_DIR"
        echo "✅ 완료: $FILENAME"
    done
done

echo "🎉 모든 압축 해제 완료!"
