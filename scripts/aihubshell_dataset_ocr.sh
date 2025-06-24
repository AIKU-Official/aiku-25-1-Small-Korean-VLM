#!/bin/bash

APIKEY_1='91345F61-E82D-4972-B25F-4A876C64D362'  # 발급받은 API 키
DATASETKEY_1=71300                             # 데이터셋 키
WORKDIR=$(pwd)                                 # 현재 디렉토리

# 다운로드할 filekey 목록
filekeys_1=(
    #525981 525982 #525983 525984 525985
    #525986 525987 525988 
    #525989 525990
    #525993 525994 525997 525998 526001 526002
    #526005 526006 526009 526010 526013 526014
    #526017 526018 526021 526022 526025 526026
    #463808 463809 463810 463811
    #463820 463821 463822 463823
    #463832 463833 463834 463835
    #463844 463845 463846 463847
    463812 463813 463814 463815 463816 463817 463818 463819
    463824 463825 463826 463827 463828 463829 463830 463831
    463836 463837 463838 463839 463840 463841 463842 463843
    463848 463849 463850 463851 463852 463853 463854 463855
)

for key in "${filekeys_1[@]}"; do
    echo "🔽 [$key] 다운로드 시작"
    
    # 파일 다운로드
    aihubshell -mode d -datasetkey 71300 -filekey $key -aihubapikey "$APIKEY_1"
    
    # zip 파일 탐색 (10분 이내에 생성된 것 중 가장 최신)
    zip_path=$(find "$WORKDIR" -maxdepth 5 -type f -name "*$key*.zip" -printf "%T@ %p\n" | sort -n | tail -n 1 | cut -d' ' -f2-)

    
    if [[ -z "$zip_path" ]]; then
        echo "⚠️ [$key] zip 파일을 찾을 수 없습니다. 스킵."
        continue
    fi

    # 압축 해제할 디렉토리 설정
    unzip_dir="${zip_path%.zip}_unzipped"
    mkdir -p "$unzip_dir"

    echo "📂 [$key] 압축 해제 중... → $unzip_dir"
    unzip -q "$zip_path" -d "$unzip_dir"

    # 압축 파일 삭제
    rm -f "$zip_path"
    echo "🗑️ [$key] zip 파일 삭제 완료"

    echo "✅ [$key] 완료"
done