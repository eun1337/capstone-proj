"""
clean_retail_sales_index.py
------------------------------------------------------------------
retail_sales_index.xlsx 정제 스크립트.
- header=[0,1] 로 2단 헤더 읽기 (0행=연월, 1행=경상지수/불변지수 구분)
- MultiIndex 컬럼을 (연월, 지수유형) 형태로 유지한 채 melt
- 지수유형 값을 "경상지수"/"불변지수"로 축약
- 시도명 공백(전각공백) 정규화
- 시도명은 2024년 이후 공식 행정구역명으로 표준화
  (강원도->강원특별자치도, 전라북도->전북특별자치도, 제주도->제주특별자치도)
- 지수유형에 따라 경상지수/불변지수 두 개의 파일로 분리 저장
  (시도별->시도, 연월->년/월, 값 컬럼명은 경상지수/불변지수로 변경)
- 결과를 같은 폴더에 retail_sales_index_current_clean.csv,
  retail_sales_index_constant_clean.csv 로 저장
"""

import re
from pathlib import Path

import pandas as pd

BASE_DIR     = Path(__file__).resolve().parents[2]           
DATA_DIR     = BASE_DIR / "data"
ECONOMIC_DIR = DATA_DIR / "external" / "economic"
SRC_FILE     = ECONOMIC_DIR / "retail_sales_index.xlsx"
OUT_CURRENT  = ECONOMIC_DIR / "retail_sales_index_current_clean.csv"
OUT_REAL     = ECONOMIC_DIR / "retail_sales_index_constant_clean.csv"


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

df = pd.read_excel(SRC_FILE, header=[0, 1])
print(f"원본 shape (2단 헤더 읽기 후): {df.shape}")
print("컬럼 예시:", df.columns.tolist()[:5])


new_cols = list(df.columns)
new_cols[0] = ("시도별", "")
df.columns = pd.MultiIndex.from_tuples(new_cols)
df[("시도별", "")] = df[("시도별", "")].apply(norm_region)


df_indexed = df.set_index(("시도별", ""))
df_indexed.index.name = "시도별"


stacked = df_indexed.stack(level=[0, 1], future_stack=True)
stacked.name = "값"
df_long = stacked.reset_index()
df_long.columns = ["시도별", "연월", "지수유형", "값"]


df_long["지수유형"] = df_long["지수유형"].str.replace("대형소매점 ", "", regex=False)


df_long["연월"] = pd.to_datetime(df_long["연월"], format="%Y.%m")
df_long["값"] = pd.to_numeric(df_long["값"], errors="coerce")

print(f"\nlong 변환 후 shape: {df_long.shape}")
print("지수유형 고유값:", df_long["지수유형"].unique())
print("숫자 변환 실패(결측):", df_long["값"].isna().sum(), "건")
print("중복행:", df_long.duplicated().sum(), "건")


df_long = df_long.rename(columns={"시도별": "시도"})
before_unique = set(df_long["시도"].unique())
df_long["시도"] = df_long["시도"].replace(SIDO_STANDARD)
after_unique = set(df_long["시도"].unique())
if before_unique - after_unique:
    print(f"\n표준화로 바뀐 시도명: {before_unique - after_unique}")
df_long["년"] = df_long["연월"].dt.year
df_long["월"] = df_long["연월"].dt.month
df_long = df_long.drop(columns=["연월"])


current = (
    df_long[df_long["지수유형"] == "경상지수"]
    .drop(columns=["지수유형"])
    .rename(columns={"값": "경상지수"})[["시도", "년", "월", "경상지수"]]
)
real = (
    df_long[df_long["지수유형"] == "불변지수"]
    .drop(columns=["지수유형"])
    .rename(columns={"값": "불변지수"})[["시도", "년", "월", "불변지수"]]
)

print(f"\n경상지수 shape: {current.shape}")
print(current.head())
print(f"\n불변지수 shape: {real.shape}")
print(real.head())
print("\n시도 고유값:", sorted(df_long["시도"].unique()))

current.to_csv(OUT_CURRENT, index=False, encoding="utf-8-sig")
real.to_csv(OUT_REAL, index=False, encoding="utf-8-sig")
print(f"저장 완료: {OUT_CURRENT}")
print(f"저장 완료: {OUT_REAL}")
