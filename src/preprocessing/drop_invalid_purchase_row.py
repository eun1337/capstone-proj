"""
drop_invalid_purchase_row.py

설명: 매입 데이터에서 작업유형/금액부호/수량부호 조합이 이상한 행을 삭제합니다.
"""

import os
import pandas as pd
from config import PARQUET_DIR

print("[ 매입 데이터 이상 데이터 삭제 작업 시작 ]")

# A센터 매입
fname = "a_center_purchase_2021_2024.parquet"
path  = os.path.join(PARQUET_DIR, fname)
df    = pd.read_parquet(path)
before = len(df)

df["_금액"] = pd.to_numeric(df["판매금액"], errors="coerce")
df["_수량"] = pd.to_numeric(df["수량"],     errors="coerce")

drop_mask = (
    ((df["작업유형"] == "반출") & (df["_금액"] == 0) & (df["_수량"] < 0)) |
    ((df["작업유형"] == "입고") & (df["_금액"] == 0) & (df["_수량"] > 0))
)

df = df[~drop_mask].drop(columns=["_금액", "_수량"])
after = len(df)
df.to_parquet(path, index=False, engine="pyarrow")
print(f"\n파일명: {fname}")
print(f"    삭제 전: {before:,}행  →  삭제 후: {after:,}행  (제거: {before - after:,}행)")

# B센터 매입
fname = "b_center_purchase_2021_2024.parquet"
path  = os.path.join(PARQUET_DIR, fname)
df    = pd.read_parquet(path)
before = len(df)

df["_금액"] = pd.to_numeric(df["판매금액"], errors="coerce")
df["_수량"] = pd.to_numeric(df["수량"],     errors="coerce")

drop_mask = (
    ((df["작업유형"] == "반출") & (df["_금액"] > 0)  & (df["_수량"] > 0)) |
    ((df["작업유형"] == "반출") & (df["_금액"] == 0) & (df["_수량"] < 0)) |
    ((df["작업유형"] == "반출") & (df["_금액"] == 0) & (df["_수량"] > 0)) |
    ((df["작업유형"] == "입고") & (df["_금액"] < 0)  & (df["_수량"] > 0)) |
    ((df["작업유형"] == "입고") & (df["_금액"] == 0) & (df["_수량"] < 0))
)

df = df[~drop_mask].drop(columns=["_금액", "_수량"])
after = len(df)
df.to_parquet(path, index=False, engine="pyarrow")
print(f"\n파일명: {fname}")
print(f"    삭제 전: {before:,}행  →  삭제 후: {after:,}행  (제거: {before - after:,}행)")

print("이상 데이터 삭제 완료!")