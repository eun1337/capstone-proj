"""
convert_date_col.py

설명: 날짜 컬럼의 타입을 datetime으로 통일하고,
      a_center_sales_2024의 시간 데이터를 제거하여 날짜만 남깁니다.
      쉼표가 포함된 경우 제거합니다.
"""

import os
import pandas as pd
from config import PARQUET_DIR

FILE_DATE_COL = {
    "a_center_purchase_2021_2024.parquet": "일자",
    "a_center_sales_2024.parquet":         "판매일",
    "a_center_sales_2021_2023.parquet":    "판매일",
    "b_center_purchase_2021_2024.parquet": "일자",
    "b_center_sales_2024.parquet":         "판매일",
    "b_center_sales_2021_2023.parquet":    "판매일",
}

print("[ 날짜 컬럼 타입 변환 및 통일 작업 시작 ]")

for fname, date_col in FILE_DATE_COL.items():
    path = os.path.join(PARQUET_DIR, fname)
    df = pd.read_parquet(path)

    if date_col not in df.columns:
        print(f"\n- {fname}  ('{date_col}' 컬럼 없음, 스킵)")
        continue

    df[date_col] = df[date_col].astype(str).str.replace(",", "", regex=False).str.strip()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[date_col] = df[date_col].dt.date
    df[date_col] = pd.to_datetime(df[date_col])

    df.to_parquet(path, index=False, engine="pyarrow")
    print(f"\n파일명: {fname}")
    print(f"    컬럼: '{date_col}'")
    print(f"    샘플: {df[date_col].dropna().head(3).tolist()}")

print("\n날짜 컬럼 변환 작업 완료")