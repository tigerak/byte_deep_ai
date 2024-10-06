#!/bin/bash

# 작업 모드 입력 받기 (train 또는 inference)
read -p "Enter mode (train|inference|title_train|title_inference): " MODE

# 입력값 검증
if [[ "$MODE" != "train" && "$MODE" != "inference" && "$MODE" != "title_train" && "$MODE" != "title_inference" ]]; then
    echo "Invalid mode. Please enter 'train' or 'inference' or title_train or title_inference."
    exit 1
fi

# 추가 옵션 설정
ADD_OPTION=""

if [[ "$MODE" == "train" || "$MODE" == "title_train" ]]; then
    read -p "Enter add (or press Enter to skip): " ADD
    if [[ ! -z "$ADD" ]]; then
        ADD_OPTION="--add $ADD"
    fi
elif [[ "$MODE" == "inference" || "$MODE" != "title_inference" ]]; then
    read -p "Enter additional option (merge|rt or press Enter to skip): " ADD
    if [[ "$ADD" == "merge" || "$ADD" == "rt" ]]; then
        ADD_OPTION="--work $ADD"
    fi
fi

# 가상환경 경로 설정
VENV_PATH="/home/deep_ai/Project/venv"

# main.py 경로 설정
MAIN_PY_PATH="/home/deep_ai/Project/function/main.py"

# 현재 날짜와 시간 설정
CURRENT_DATE=$(date '+%y_%m_%d')
CURRENT_TIME=$(date '+%H_%M_%S')

# 로그를 저장할 디렉토리 경로 설정
LOG_DIR="/home/deep_ai/Project/data/output/sft/$CURRENT_DATE"

# 디렉토리 생성 (이미 존재하면 무시)
mkdir -p $LOG_DIR

# 로그 파일 경로 설정
LOG_FILE="$LOG_DIR/$CURRENT_TIME.log"

# nohup을 사용하여 main.py 실행 (출력을 nohup.out에 저장) / 즉시 출력 : -u
nohup $VENV_PATH/bin/python -u $MAIN_PY_PATH --mode $MODE $ADD_OPTION --date $CURRENT_DATE --time $CURRENT_TIME > $LOG_FILE 2>&1 &

echo "main.py is running in the background with nohup"