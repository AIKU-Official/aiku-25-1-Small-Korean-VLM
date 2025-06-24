#!/bin/bash

# AIHub 한 개씩 다운로드 및 처리 자동화 스크립트
# - filekey로 하나씩 다운로드
# - 내부 폴더 구조 파악 후 *_0001.jpg/json/png 만 남기고 삭제
# - zip 파일은 자동 삭제되므로 따로 처리하지 않음

APIKEY='25EC46A3-85F1-4658-94E0-181312551A84'
DATASETKEY=81
WORKDIR=$(pwd)

# filekeys 리스트 (순차 처리)
filekeys=(
    67939 67940 
)

for key in "${filekeys[@]}"; do
    echo "🔽 filekey $key 다운로드 중..."
    # zip 다운밖에 안되는 듯.
    aihubshell -mode d -datasetkey $DATASETKEY -filekey $key -aihubapikey "$APIKEY"

    echo "✅ filekey $key 완료"
done
