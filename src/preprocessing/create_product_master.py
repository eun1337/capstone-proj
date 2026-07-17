"""
create_product_master.py

설명: 각 parquet 파일에서 바코드 + 상품명 기준으로
      중복을 제거한 상품 마스터 파일을 생성합니다.
      상품 마스터에는 대분류, 중분류, 소분류 컬럼도 함께 포함됩니다.
      파일별 원본 행 개수, 중복 행 개수,
      중복 제거 후 최종 행 개수를 출력합니다.
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/parquet")
OUTPUT_DIR = Path("data/master")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

files = [
    "a_center_purchase_2021_2024.parquet",
    "a_center_sales_2021_2023.parquet",
    "a_center_sales_2024.parquet",
    "b_center_purchase_2021_2024.parquet",
    "b_center_sales_2021_2023.parquet",
    "b_center_sales_2024.parquet"
]

columns_to_use = [
    "바코드",
    "상품명",
    "대분류",
    "중분류",
    "소분류"
]

for file_name in files:

    print("\n" + "=" * 60)
    print(f"파일 처리 중: {file_name}")

    file_path = DATA_DIR / file_name

    df = pd.read_parquet(
        file_path,
        columns=columns_to_use
    )

    original_rows = len(df)

    print(f"원본 행 개수: {original_rows:,}")

    df["바코드"] = (
        df["바코드"]
        .astype(str)
        .str.strip()
    )

    df["상품명"] = (
        df["상품명"]
        .astype(str)
        .str.strip()
    )

    master_df = df.drop_duplicates(
        subset=["바코드", "상품명"]
    )

    final_rows = len(master_df)

    duplicate_rows = original_rows - final_rows

    print(f"중복 행 개수: {duplicate_rows:,}")
    print(f"중복 제거 후 최종 행 개수: {final_rows:,}")

    master_df = master_df.sort_values(
        by=[
            "대분류",
            "중분류",
            "소분류",
            "상품명",
            "바코드"
        ]
    )

    master_df = master_df.reset_index(drop=True)

    output_name = (
        file_name
        .replace(".parquet", "_product_master.xlsx")
    )

    output_path = OUTPUT_DIR / output_name

    master_df.to_excel(
        output_path,
        index=False
    )

    print(f"저장 위치: {output_path}")

print("\n모든 상품 마스터 생성 완료")