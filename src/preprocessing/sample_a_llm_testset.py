"""
sample_a_llm_testset.py

설명:
A센터 상품 마스터 데이터에서

- 정상 분류 상품 175개
- 결측 포함 상품 200개

를 랜덤 추출하여
LLM 성능 평가용 테스트셋을 생성합니다.

입력 파일:
- a_final_product_master_complete.xlsx
- a_final_product_master_missing.xlsx

출력 파일:
- a_llm_testset_375.xlsx

추가 생성 컬럼:
- sample_id
- 테스트케이스
- 정답_대분류
- 정답_중분류
- 정답_소분류

모델 결과 입력 컬럼:
- GPT_대분류
- GPT_중분류
- GPT_소분류

- Claude_대분류
- Claude_중분류
- Claude_소분류

- Gemini_대분류
- Gemini_중분류
- Gemini_소분류

- CLOVA_대분류
- CLOVA_중분류
- CLOVA_소분류

샘플 구성:
- 정상 데이터 175건
- 결측 데이터 200건
- 총 375건

용도:
- Ground Truth 작성
- GPT-4o 성능 평가
- Claude Sonnet 성능 평가
- Gemini 성능 평가
- HyperCLOVA X 성능 평가
- 모델 간 정확도 비교
"""

import pandas as pd
from pathlib import Path

# ==========================================================
# 경로 설정
# ==========================================================

BASE_DIR = Path(__file__).resolve().parents[2]

MASTER_DIR = BASE_DIR / "data" / "master"

COMPLETE_FILE = (
    MASTER_DIR /
    "a_final_product_master_complete.xlsx"
)

MISSING_FILE = (
    MASTER_DIR /
    "a_final_product_master_missing.xlsx"
)

# ==========================================================
# 샘플 설정
# ==========================================================

NORMAL_SAMPLE_SIZE = 175
MISSING_SAMPLE_SIZE = 200

RANDOM_SEED = 42

# ==========================================================
# 데이터 로드
# ==========================================================

print("데이터 로드 중...")

complete_df = pd.read_excel(
    COMPLETE_FILE
)

missing_df = pd.read_excel(
    MISSING_FILE
)

print(
    f"정상 데이터 수 : {len(complete_df):,}"
)

print(
    f"결측 데이터 수 : {len(missing_df):,}"
)

# ==========================================================
# 랜덤 샘플 추출
# ==========================================================

normal_sample = complete_df.sample(
    n=NORMAL_SAMPLE_SIZE,
    random_state=RANDOM_SEED
)

missing_sample = missing_df.sample(
    n=MISSING_SAMPLE_SIZE,
    random_state=RANDOM_SEED
)

# ==========================================================
# 테스트 케이스 구분
# ==========================================================

normal_sample["테스트케이스"] = "정상"

missing_sample["테스트케이스"] = "결측"

# ==========================================================
# 통합
# ==========================================================

testset_df = pd.concat(
    [
        normal_sample,
        missing_sample
    ],
    ignore_index=True
)

# ==========================================================
# Sample ID 생성
# ==========================================================

testset_df.insert(
    0,
    "sample_id",
    range(
        1,
        len(testset_df) + 1
    )
)

# ==========================================================
# Ground Truth 컬럼
# ==========================================================

testset_df["정답_대분류"] = ""
testset_df["정답_중분류"] = ""
testset_df["정답_소분류"] = ""

# ==========================================================
# GPT 결과 컬럼
# ==========================================================

testset_df["GPT_대분류"] = ""
testset_df["GPT_중분류"] = ""
testset_df["GPT_소분류"] = ""

# ==========================================================
# Claude 결과 컬럼
# ==========================================================

testset_df["Claude_대분류"] = ""
testset_df["Claude_중분류"] = ""
testset_df["Claude_소분류"] = ""

# ==========================================================
# Gemini 결과 컬럼
# ==========================================================

testset_df["Gemini_대분류"] = ""
testset_df["Gemini_중분류"] = ""
testset_df["Gemini_소분류"] = ""

# ==========================================================
# HyperCLOVA X 결과 컬럼
# ==========================================================

testset_df["CLOVA_대분류"] = ""
testset_df["CLOVA_중분류"] = ""
testset_df["CLOVA_소분류"] = ""

# ==========================================================
# 컬럼 순서 정리
# ==========================================================

priority_cols = [

    "sample_id",
    "테스트케이스",

    "상품명",

    "대분류",
    "중분류",
    "소분류",

    "정답_대분류",
    "정답_중분류",
    "정답_소분류",

    "GPT_대분류",
    "GPT_중분류",
    "GPT_소분류",

    "Claude_대분류",
    "Claude_중분류",
    "Claude_소분류",

    "Gemini_대분류",
    "Gemini_중분류",
    "Gemini_소분류",

    "CLOVA_대분류",
    "CLOVA_중분류",
    "CLOVA_소분류"
]

existing_priority_cols = [

    col
    for col in priority_cols
    if col in testset_df.columns

]

remaining_cols = [

    col
    for col in testset_df.columns
    if col not in existing_priority_cols

]

testset_df = testset_df[
    existing_priority_cols +
    remaining_cols
]

# ==========================================================
# 저장
# ==========================================================

OUTPUT_FILE = (
    MASTER_DIR /
    "a_llm_testset_375.xlsx"
)

testset_df.to_excel(
    OUTPUT_FILE,
    index=False
)

# ==========================================================
# 결과 출력
# ==========================================================

print("\n========== 추출 결과 ==========")

print(
    f"정상 샘플 수 : {NORMAL_SAMPLE_SIZE}"
)

print(
    f"결측 샘플 수 : {MISSING_SAMPLE_SIZE}"
)

print(
    f"전체 샘플 수 : {len(testset_df)}"
)

print(
    f"\n저장 완료 : {OUTPUT_FILE.name}"
)