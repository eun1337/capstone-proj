"""
rename_columns.py

설명: 파일별로 불일치하는 컬럼명을 통일합니다.
      data/parquet/ 의 원본을 읽어 동일 위치에 덮어씁니다.
"""

import os
import pandas as pd
from config import PARQUET_DIR

RENAME_MAP = {
    "공급금액":            "공급가액",
    "우편번호":            "매출처 우편번호",
    "상품 바코드(대한상의)": "바코드",
    "부가세(과세)":         "부가세",
    "옵션 코드":           "옵션코드",
}

FILES = [
    "a_center_purchase_2021_2024.parquet",
    "a_center_sales_2024.parquet",
    "a_center_sales_2021_2023.parquet",
    "b_center_purchase_2021_2024.parquet",
    "b_center_sales_2024.parquet",
    "b_center_sales_2021_2023.parquet",
]

print("[ 컬럼명 통일 작업 시작 ]")

for fname in FILES:
    path = os.path.join(PARQUET_DIR, fname)
    df = pd.read_parquet(path)
    applied = {k: v for k, v in RENAME_MAP.items() if k in df.columns}
    if applied:
        df = df.rename(columns=applied)
        df.to_parquet(path, index=False, engine="pyarrow")
        print(f"\n파일명: {fname}")
        for old, new in applied.items():
            print(f"    '{old}'  →  '{new}'")
    else:
        print(f"\n- {fname}  (변경 없음)")

print("컬럼명 통일 작업 완료")