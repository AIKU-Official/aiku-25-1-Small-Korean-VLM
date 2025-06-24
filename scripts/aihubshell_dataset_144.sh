#!/bin/bash

# AIHub 한 개씩 다운로드 및 처리 자동화 스크립트
# - filekey로 하나씩 다운로드
# - 내부 폴더 구조 파악 후 *_0001.jpg/json/png 만 남기고 삭제
# - zip 파일은 자동 삭제되므로 따로 처리하지 않음

APIKEY='EC263BEC-B54B-4B75-8383-5F48F73FA3EE'
DATASETKEY=144
WORKDIR=$(pwd)

# filekeys 리스트 (순차 처리)
filekeys=(
    # 랜드마크
#   완 49994 49995 49996 50105 50106 50107 50108

    # 유적
  # 완 50206 50207 50208 
  # 완 50111 50112 50113 50114 50115 50116 50117 50118 50119 50120 50121 50122 50123 50124 50125 50126 50127 50128 50129 50130 50131 50132 50133 50134 50135 50136 50137 50138 50139 50140 50141 50142 50143 50144 50145 50146 50147 50148 50149 50150 50151 50152 50153 50154 50155 50156 50157 50158 50159 50160 50161 50162 50163 50164 
  # 완 50052 50053 50054 50055 50056 50057 50058 50059 50060 50061 50062 50063 50064 50065 50066 50067 50068 50069 50070 50071 50072 50073 50074 
  # 나중에 50075 50076 50077 50078 50079 50080 50081 50082

    # 상품
  #  완 50083 50084 50085 50086 50087 50088 50089 
  # 나중에 다운 50090 50091 50092 50093 50094 50095 50096 50097 50098 50099 50100 50101 50102 50103 50104
 
  # 삭제 파일 깨짐 이슈 63613
)

for key in "${filekeys[@]}"; do
    echo "🔽 filekey $key 다운로드 중..."
    # zip 다운밖에 안되는 듯.
    aihubshell -mode d -datasetkey $DATASETKEY -filekey $key -aihubapikey "$APIKEY"
    
    # .zip 파일 탐색
    echo "🔍 [$key] 새로 생성된 zip 탐색 중..."
    zip_path=$(find "$WORKDIR" -maxdepth 5 -type f -name '*.zip' -newermt "$(date -d '5 minutes ago' '+%Y-%m-%d %H:%M:%S')" | head -n 1)
    # 못찾은 경우
    if [[ -z "$zip_path" ]]; then
        echo "⚠️ [$key] zip 파일 탐색 실패. 스킵합니다."
        continue
    fi

    # unzip
    unzip_dir="${zip_path%.zip}_unzipped"
    mkdir -p "$unzip_dir"
    echo "📂 [$key] unzip 중... → $unzip_dir"
    unzip -q "$zip_path" -d "$unzip_dir"

    # zip 파일 삭제.
    rm -f "$zip_path"

    # HF로 시작하는 하위폴더 순회
    find "$unzip_dir" -type d -name "HF*" | while read -r hffolder; do
        find "$hffolder" -type f \( ! -iname '*_0001.jpg' ! -iname '*_0001.json' ! -iname '*0001.png' \) -delete
    done

    echo "✅ filekey $key 완료"
done
