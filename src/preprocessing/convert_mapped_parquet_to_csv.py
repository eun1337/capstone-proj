"""
convert_mapped_parquet_to_csv.py

설명: data/parquet/ 폴더 내 파일명이 'mapped.parquet'로 끝나는 파일을 CSV 형식으로 변환하여 data/csv/ 폴더에 저장하는 코드
"""

import os
import glob
import pandas as pd
from config import PARQUET_DIR, BASE_DIR

CSV_DIR = os.path.join(BASE_DIR, "csv")
os.makedirs(CSV_DIR, exist_ok=True)


def convert_mapped_parquet_to_csv():
    pattern = os.path.join(PARQUET_DIR, "*mapped.parquet")
    files = glob.glob(pattern)

    if not files:
        print("변환 대상 파일 없음: '*mapped.parquet' 파일이 존재하지 않습니다.")
        return

    print(f"변환 대상 파일 수: {len(files)}\n")
    for src in files:
        base_name = os.path.basename(src).replace(".parquet", ".csv")
        dest = os.path.join(CSV_DIR, base_name)

        print(f"Processing: {os.path.basename(src)}")
        df = pd.read_parquet(src, engine="pyarrow")
        df.to_csv(dest, index=False, encoding="utf-8-sig")
        print(f"Saved: {dest} (shape: {df.shape})")


if __name__ == "__main__":
    convert_mapped_parquet_to_csv()
    print(f"\n변환 완료. 출력 폴더: {os.path.abspath(CSV_DIR)}")
