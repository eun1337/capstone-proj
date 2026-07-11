"""
clean_cpi_monthly.py
------------------------------------------------------------------
cpi_monthly.xlsx 정제 스크립트.
- header=0 기본 그대로 (헤더 1줄이라 문제없음)
- 시도별 공백(전각공백 포함) 정규화
- wide format(날짜가 컬럼)을 melt로 long format 변환
- "전국" 행은 전국 집계값이므로 별도 파일로 분리 저장
- 시도명은 2024년 이후 공식 행정구역명으로 표준화
  (강원도->강원특별자치도, 전라북도->전북특별자치도, 제주도->제주특별자치도)
- 최종 컬럼:
  - 시도별(전국 제외): 시도,년,월,소비자물가지수
  - 전국: 년,월,소비자물가지수 (시도 구분이 의미 없으므로 시도별 컬럼 자체를 제거)
- 결과를 같은 폴더에 cpi_monthly_clean.csv (+ cpi_monthly_national.csv) 로 저장
"""

import re
from pathlib import Path

import pandas as pd

BASE_DIR     = Path(__file__).resolve().parents[2]          
DATA_DIR     = BASE_DIR / "data"
ECONOMIC_DIR = DATA_DIR / "external" / "economic"
SRC_FILE     = ECONOMIC_DIR / "cpi_monthly.xlsx"
OUT_FILE     = ECONOMIC_DIR / "cpi_monthly_clean.csv"
OUT_NATIONAL = ECONOMIC_DIR / "cpi_monthly_national.csv"


def norm_region(s):
    if pd.isna(s):
        return s
    return re.sub(r"[\s\u3000]+", "", str(s))



SIDO_STANDARD = {
    "강원도": "강원특별자치도",
    "전라북도": "전북특별자치도",
    "제주도": "제주특별자치도",
    "제주": "제주특별자치도",
}

df = pd.read_excel(SRC_FILE)
print(f"원본 shape: {df.shape}")

df["시도별"] = df["시도별"].apply(norm_region)


df_long = pd.melt(df, id_vars=["시도별"], var_name="연월", value_name="CPI")
df_long["연월"] = pd.to_datetime(df_long["연월"], format="%Y.%m")


national = df_long[df_long["시도별"] == "전국"].copy()
regional = df_long[df_long["시도별"] != "전국"].copy()


regional = regional.rename(columns={"시도별": "시도", "CPI": "소비자물가지수"})
before_unique = set(regional["시도"].unique())
regional["시도"] = regional["시도"].replace(SIDO_STANDARD)
after_unique = set(regional["시도"].unique())
if before_unique - after_unique:
    print(f"\n표준화로 바뀐 시도명: {before_unique - after_unique}")
regional["년"] = regional["연월"].dt.year
regional["월"] = regional["연월"].dt.month
regional = regional.drop(columns=["연월"])
regional = regional[["시도", "년", "월", "소비자물가지수"]]


national = national.rename(columns={"CPI": "소비자물가지수"})
national["년"] = national["연월"].dt.year
national["월"] = national["연월"].dt.month
national = national.drop(columns=["연월", "시도별"])
national = national[["년", "월", "소비자물가지수"]]

print(f"\n시도별(전국 제외) 최종 shape: {regional.shape}")
print(f"전국 최종 shape: {national.shape}")
print()
print(regional.head())
print()
print("시도별(전국 제외) 고유값:", sorted(regional["시도"].unique()))
print("중복행:", regional.duplicated(subset=["시도", "년", "월"]).sum(), "건")
print("결측치:")
print(regional.isna().sum())

regional.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
national.to_csv(OUT_NATIONAL, index=False, encoding="utf-8-sig")
print(f"저장 완료: {OUT_FILE}")
print(f"저장 완료: {OUT_NATIONAL}")
