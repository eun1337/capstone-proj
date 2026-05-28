"""
convert_to_parquet.py

설명: data/ 폴더 내 원본 엑셀 파일을 데이터 처리 효율화를 위해 parquet 형식으로 변환합니다.
      멀티시트 파일(2021-2023 매출 데이터)의 경우 모든 시트를 하나로 병합하여 저장합니다.
      저장 시 파일명을 영문으로 변환합니다.
"""

import pandas as pd
from config import BASE_DIR, PARQUET_DIR

SINGLE_SHEET_FILES = {
    "A센터_매입_2021-2024.xlsx": "a_center_purchase_2021_2024.parquet",
    "A센터_매출_2024.xlsx":      "a_center_sales_2024.parquet",
    "B센터_매입_2021-2024.xlsx": "b_center_purchase_2021_2024.parquet",
    "B센터_매출_2024.xlsx":      "b_center_sales_2024.parquet",
}

MULTI_SHEET_FILES = {
    "A센터_매출_2021-2023.xlsx": "a_center_sales_2021_2023.parquet",
    "B센터_매출_2021-2023.xlsx": "b_center_sales_2021_2023.parquet",
}


def convert_single_sheets():
    import os
    print("--- 단일 시트 파일 변환 시작 ---")
    for fname, dest_name in SINGLE_SHEET_FILES.items():
        src  = os.path.join(BASE_DIR, fname)
        dest = os.path.join(PARQUET_DIR, dest_name)
        print(f"Processing: {fname}")
        df = pd.read_excel(src, sheet_name=0, dtype=str)
        df.to_parquet(dest, index=False, engine="pyarrow")
        print(f"Saved: {dest} (shape: {df.shape})")


def convert_multi_sheets():
    import os
    print("\n--- 멀티 시트 파일 병합 및 변환 시작 ---")
    for fname, dest_name in MULTI_SHEET_FILES.items():
        src  = os.path.join(BASE_DIR, fname)
        dest = os.path.join(PARQUET_DIR, dest_name)
        print(f"Processing: {fname}")
        xl = pd.ExcelFile(src)
        dfs = []
        for sheet in xl.sheet_names:
            df = pd.read_excel(src, sheet_name=sheet, dtype=str)
            df["_source_sheet"] = sheet
            print(f"  - Sheet '{sheet}': {len(df):,} rows")
            dfs.append(df)
        combined = pd.concat(dfs, ignore_index=True)
        combined.to_parquet(dest, index=False, engine="pyarrow")
        print(f"Saved: {dest} (total rows: {len(combined):,})")


if __name__ == "__main__":
    convert_single_sheets()
    convert_multi_sheets()
    import os
    print("\n변환 프로세스 작업 완료")
    print(f"Output directory: {os.path.abspath(PARQUET_DIR)}")