"""
drop_invalid_barcode.py

설명: 바코드 값이 null이거나 0인 행을 삭제합니다.
"""

import os
import pandas as pd
from config import PARQUET_DIR

FILES = [
    "a_center_purchase_2021_2024.parquet",
    "a_center_sales_2024.parquet",
    "a_center_sales_2021_2023.parquet",
    "b_center_purchase_2021_2024.parquet",
    "b_center_sales_2024.parquet",
    "b_center_sales_2021_2023.parquet",
]

print("[ 바코드 결측치 제거 작업 시작 ]")

for fname in FILES:
    path = os.path.join(PARQUET_DIR, fname)
    df = pd.read_parquet(path)

    if "바코드" not in df.columns:
        print(f"\n- {fname}  (바코드 컬럼 없음, 스킵)")
        continue

    before = len(df)
    df = df[df["바코드"].notna()]
    df = df[df["바코드"].str.strip() != ""]
    df = df[df["바코드"].str.strip() != "0"]
    after = len(df)

    df.to_parquet(path, index=False, engine="pyarrow")
    print(f"\n파일명: {fname}")
    print(f"    삭제 전: {before:,}행  →  삭제 후: {after:,}행  (제거: {before - after:,}행)")

print("바코드 결측치 제거 작업 완료")