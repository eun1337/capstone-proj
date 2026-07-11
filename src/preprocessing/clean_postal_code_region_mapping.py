"""
clean_postal_code_region_mapping.py
------------------------------------------------------------------
postal_code_region_mapping.xlsx의 "전체_우편번호" 탭에서 데이터를 읽어와 정제.
- 원본 우편번호가 int64로 읽혀서 "06236" -> 6236 처럼 앞자리 0이 사라진 상태
  -> str로 변환 후 zfill(5)로 5자리 우편번호 형식 복원
- 읍면 컬럼은 사용하지 않으므로 제거
- 시도명은 2024년 이후 공식 행정구역명으로 표준화
  (강원도->강원특별자치도, 전라북도->전북특별자치도, 제주도->제주특별자치도)
- 최종 컬럼: 우편번호,시도,시군구
- 결과를 같은 폴더에 postal_code_region_mapping_clean.csv 로 저장
(합계 행은 이미 수동으로 제거된 상태를 전제로 함)
"""

from pathlib import Path

import pandas as pd

BASE_DIR     = Path(__file__).resolve().parents[2]          
DATA_DIR     = BASE_DIR / "data"
REGIONAL_DIR = DATA_DIR / "external" / "regional"
MAPPING_FILE = REGIONAL_DIR / "postal_code_region_mapping.xlsx"
OUT_FILE     = REGIONAL_DIR / "postal_code_region_mapping_clean.csv"


SIDO_STANDARD = {
    "강원도": "강원특별자치도",
    "전라북도": "전북특별자치도",
    "제주도": "제주특별자치도",
    "제주": "제주특별자치도",
}

df = pd.read_excel(MAPPING_FILE, sheet_name="전체_우편번호")
print(f"원본 shape: {df.shape}")
print("원본 컬럼:", df.columns.tolist())


df["우편번호"] = df["우편번호"].astype(str).str.zfill(5)


df = df.drop(columns=["읍면"])


before_unique = set(df["시도"].unique())
df["시도"] = df["시도"].replace(SIDO_STANDARD)
after_unique = set(df["시도"].unique())
if before_unique - after_unique:
    print(f"\n표준화로 바뀐 시도명: {before_unique - after_unique}")

df = df[["우편번호", "시도", "시군구"]]

print(f"\n최종 shape: {df.shape}")
print(df.head())
print()
print("5자리 아닌 값 있는지 확인:", (df["우편번호"].str.len() != 5).sum(), "건")
print("시도 고유값:", sorted(df["시도"].unique()))
print("중복행:", df.duplicated().sum(), "건")

df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
print(f"저장 완료: {OUT_FILE}")
