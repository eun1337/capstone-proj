"""
check_files.py

설명: data/csv/ 폴더 내 CSV 파일의 컬럼명, dtype, row 수, 결측치 여부를 출력합니다.
    load_to_mysql.py 실행 전 데이터 상태를 확인하는 용도입니다.
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_DIR = os.path.join(BASE_DIR, "data", "csv")

CSV_FILES = {
    "a_purchase":       "a_center_purchase_2021_2024_mapped.csv",
    "a_sales_2021_2023": "a_center_sales_2021_2023_mapped.csv",
    "a_sales_2024":     "a_center_sales_2024_mapped.csv",
    "b_purchase":       "b_center_purchase_2021_2024_mapped.csv",
    "b_sales_2021_2023": "b_center_sales_2021_2023_mapped.csv",
    "b_sales_2024":     "b_center_sales_2024_mapped.csv",
}


def check_file(table_name: str, file_name: str):
    path = os.path.join(CSV_DIR, file_name)
    if not os.path.exists(path):
        print(f"[MISSING] {file_name} — 파일을 찾을 수 없습니다: {path}")
        return

    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)

    print(f"\n{'=' * 60}")
    print(f"테이블명 (예정): {table_name}")
    print(f"파일명          : {file_name}")
    print(f"행 수           : {len(df):,}")
    print(f"열 수           : {len(df.columns)}")
    print(f"\n[컬럼 정보]")
    print(f"{'컬럼명':<35} {'dtype':<15} {'결측치 수':>10} {'결측 비율':>10}")
    print("-" * 75)
    for col in df.columns:
        missing = df[col].isna().sum()
        ratio = missing / len(df) * 100 if len(df) > 0 else 0.0
        print(f"{col:<35} {str(df[col].dtype):<15} {missing:>10,} {ratio:>9.1f}%")


def main():
    print(f"CSV 폴더: {CSV_DIR}\n")
    for table_name, file_name in CSV_FILES.items():
        check_file(table_name, file_name)
    print(f"\n{'=' * 60}")
    print("점검 완료.")


if __name__ == "__main__":
    main()
