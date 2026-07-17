"""
drop_invalid_zipcode_barcode.py

설명: 우편번호 및 바코드 이상값 행을 제거합니다.
      - 우편번호: 5자리 패딩 후 정수값 1000 미만인 행 제거
        (0패딩 복원은 convert_zipcode_dtype.py 에서 완료된 상태)
      - 바코드: 8, 12, 13, 14자리 외 행 제거
"""

import os
import pandas as pd
from config import PARQUET_DIR

VALID_BARCODE_DIGITS = {8, 12, 13, 14}

files = {
    "a_center_purchase_2021_2024" : ("공급업체 우편번호", "바코드"),
    "a_center_sales_2021_2023"    : ("매출처 우편번호",   "바코드"),
    "a_center_sales_2024"         : ("매출처 우편번호",   "바코드"),
    "b_center_purchase_2021_2024" : ("공급업체 우편번호", "바코드"),
    "b_center_sales_2021_2023"    : ("매출처 우편번호",   "바코드"),
    "b_center_sales_2024"         : ("매출처 우편번호",   "바코드"),
}

print("[ 우편번호 및 바코드 이상값 제거 작업 시작 ]")

for fname, (zip_col, bar_col) in files.items():
    fpath = os.path.join(PARQUET_DIR, f"{fname}.parquet")
    if not os.path.exists(fpath):
        print(f"파일 없음: {fname}")
        continue

    df = pd.read_parquet(fpath)
    original_len = len(df)

    # 우편번호: 1000 미만 제거 (이미 5자리 문자열 상태)
    zip_as_int    = pd.to_numeric(df[zip_col], errors="coerce")
    zip_drop_mask = zip_as_int < 1000
    zip_drop_cnt  = zip_drop_mask.sum()
    df = df[~zip_drop_mask].copy()

    # 바코드: 8, 12, 13, 14자리 외 제거
    bar_len       = df[bar_col].astype(str).str.len()
    bar_drop_mask = ~bar_len.isin(VALID_BARCODE_DIGITS)
    bar_drop_cnt  = bar_drop_mask.sum()
    df = df[~bar_drop_mask].copy()

    df.to_parquet(fpath, index=False, engine="pyarrow")

    print(f"\n파일명: {fname}")
    print(f"    우편번호 제거: {zip_drop_cnt:,}행 / 바코드 제거: {bar_drop_cnt:,}행")
    print(f"    {original_len:,}행 → {len(df):,}행")

print("우편번호 및 바코드 이상값 제거 완료")