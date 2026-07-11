"""
clean_public_holidays.py
------------------------------------------------------------------
public_holidays_2021_2024.csv 정제 스크립트 (data.go.kr 특일 API 포맷 가정:
locdate, dateName, isHoliday, dateKind, seq).
- dateKind, isHoliday, seq 컬럼 제거
- dateName -> 공휴일 로 컬럼명 변경
- locdate -> 년/월/일 컬럼으로 분리
- 공휴일 중 "설날", "추석"에 해당하는 행만 추출 (연휴 포함, contains 매칭)
- "대체공휴일(설날)" 등 변형된 명칭은 모두 "설날"/"추석"으로 통일
- 결과를 같은 폴더에 public_holidays_2021_2024_clean.csv 로 저장
"""

from pathlib import Path

import pandas as pd

BASE_DIR    = Path(__file__).resolve().parents[2]           # capstone-proj/
DATA_DIR    = BASE_DIR / "data"
HOLIDAY_DIR = DATA_DIR / "external" / "holiday"
SRC_FILE    = HOLIDAY_DIR / "public_holidays_2021_2024.csv"
OUT_FILE    = HOLIDAY_DIR / "public_holidays_2021_2024_clean.csv"

try:
    df = pd.read_csv(SRC_FILE, dtype={"locdate": str})
except UnicodeDecodeError:
    df = pd.read_csv(SRC_FILE, dtype={"locdate": str}, encoding="cp949")

print(f"원본 shape: {df.shape}")
print("원본 컬럼:", df.columns.tolist())

drop_cols = [c for c in ["dateKind", "isHoliday", "seq"] if c in df.columns]
df = df.drop(columns=drop_cols)
df = df.rename(columns={"dateName": "공휴일"})

df["locdate"] = pd.to_datetime(df["locdate"], format="%Y%m%d")
df["년"] = df["locdate"].dt.year
df["월"] = df["locdate"].dt.month
df["일"] = df["locdate"].dt.day
df = df.drop(columns=["locdate"])

print("\n공휴일 원본 고유값:", sorted(df["공휴일"].unique()))

mask = df["공휴일"].str.contains("설날") | df["공휴일"].str.contains("추석")
df = df[mask].copy()


df["공휴일"] = df["공휴일"].apply(lambda s: "설날" if "설날" in s else "추석")

df = df[["공휴일", "년", "월", "일"]]

print(f"\n설날/추석 통일 후 shape: {df.shape}")
print("통일 후 공휴일 고유값:", sorted(df["공휴일"].unique()))
print(df)

df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
print(f"저장 완료: {OUT_FILE}")