#!/usr/bin/env bash
# 발표/데모용 백엔드 실행 스크립트.
#
# --reload 없이 실행한다 — 개발 중 파일 저장으로 인한 자동 재기동은 src/backend/main.py의
# lifespan이 애써 예열해둔 캐시(@lru_cache)를 통째로 날려버려서, 발표 도중 저장 한 번에
# 다시 수십 초짜리 콜드 응답으로 돌아간다. --workers도 1로 고정한다 — 캐시가 프로세스
# 메모리 안에 있는 구조라, worker를 늘리면 같은 parquet을 worker 수만큼 중복으로 메모리에
# 올릴 뿐이고(데모 노트북 메모리 낭비), 발표 상황에 필요 없는 동시접속 처리량만 늘어난다.
#
# 콘솔에 "Application startup complete"와 "[warmup] ... 사전 계산 완료" 로그가 다 뜬
# 뒤에만 화면 공유를 시작할 것 — 그 전에 A/B센터를 클릭하면 그 첫 클릭만 다시 느리다
# (기동 자체는 parquet 로딩 때문에 실측 약 50~65초 걸린다. 이건 서버 기동 시간이지 발표
# 중 응답 시간이 아니다 — 예열이 끝난 뒤에는 A/B센터 전환이 0.1~0.3초대로 응답한다).
set -euo pipefail
cd "$(dirname "$0")/../src/backend"

# 이 프로젝트는 별도 venv/conda 환경을 쓰지 않고 시스템 기본 python 인터프리터에
# fastapi/uvicorn/pandas가 직접 설치돼 있다(팀원별 경로가 다를 수 있어 PYTHON_BIN으로
# 오버라이드 가능하게 해뒀다).
PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  PYTHON_BIN=python3
fi

export PYTHONIOENCODING=utf-8

echo "[run_demo] ${PYTHON_BIN} 로 백엔드 기동 (--reload 없음, --workers 1)"
echo "[run_demo] 'Application startup complete' + '[warmup] ... 사전 계산 완료' 로그가 뜰 때까지 기다린 뒤 발표를 시작하세요."
exec "$PYTHON_BIN" -m uvicorn main:app --host 0.0.0.0 --port 8000 --workers 1
