#!/bin/bash

# 로그 파일이 저장되는 디렉토리 설정
LOG_DIR="app"
CURRENT_LOG="$LOG_DIR/gunicorn.log"

# # 어제 날짜를 가져옴
# YESTERDAY=$(date -d "yesterday" '+%Y-%m-%d')

# # 어제 날짜를 붙인 파일 이름 설정
# YESTERDAY_LOG="$LOG_DIR/$YESTERDAY_gunicorn.log"

# # 현재 로그 파일을 어제 날짜의 파일로 이동
# mv $CURRENT_LOG $YESTERDAY_LOG

# # Nginx 재시작
# sudo service nginx restart

# Gunicorn 재시작 (gunicorn 프로세스 ID를 얻어 재시작)
# pkill gunicorn
gunicorn -c app/gunicorn_config.py app.run:app > $CURRENT_LOG 2>&1 &

# 필요한 경우 백업 및 정리 작업 추가 가능

# sudo service nginx restart
# gunicorn -c app/gunicorn_config.py app.run:app > app/gunicorn.log 2>&1 & 
# nohup redis-server > app/redis.log 2>&1 &
# nohup python -u app/rq_worker.py > app/rq_worker.log 2>&1 &
# nohup python -u rss/main.py > /dev/null 2>&1 &