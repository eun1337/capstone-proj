"""
drop_optioncode_col.py

설명: 옵션 코드만 남기고 옵션 열을 삭제합니다.
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

print("[ 옵션 열 삭제 작업 시작 ]")

for fname in FILES:
    path = os.path.join(PARQUET_DIR, fname)
    df = pd.read_parquet(path)

    if "옵션" not in df.columns:
        print(f"\n- {fname}  (옵션 컬럼 없음, 스킵)")
        continue

    before_cols = df.shape[1]
    df = df.drop(columns=["옵션"])
    after_cols = df.shape[1]

    df.to_parquet(path, index=False, engine="pyarrow")
    print(f"\n파일명: {fname}")
    print(f"    컬럼 수: {before_cols}개  →  {after_cols}개")

print("옵션 열 삭제 작업 완료")