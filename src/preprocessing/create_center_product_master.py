"""
create_center_product_master.py

설명: A센터와 B센터의 상품 마스터 파일들을 각각 통합하여
      최종 품목 마스터 파일을 생성합니다.
      바코드 + 상품명 조합 기준으로 중복 제거를 수행하며,
      대분류, 중분류, 소분류 컬럼을 함께 포함합니다.
      A센터 최종 마스터와 B센터 최종 마스터를 각각 생성합니다.
"""

import pandas as pd
from pathlib import Path

MASTER_DIR = Path("data/master")
OUTPUT_DIR = Path("data/master")

a_files = [
    "a_center_purchase_2021_2024_product_master.xlsx",
    "a_center_sales_2021_2023_product_master.xlsx",
    "a_center_sales_2024_product_master.xlsx"
]

b_files = [
    "b_center_purchase_2021_2024_product_master.xlsx",
    "b_center_sales_2021_2023_product_master.xlsx",
    "b_center_sales_2024_product_master.xlsx"
]

columns_to_use = [
    "바코드",
    "상품명",
    "대분류",
    "중분류",
    "소분류"
]

center_configs = [
    {
        "center_name": "A센터",
        "files": a_files,
        "output_name": "a_final_product_master.xlsx"
    },
    {
        "center_name": "B센터",
        "files": b_files,
        "output_name": "b_final_product_master.xlsx"
    }
]

for config in center_configs:

    print("\n" + "=" * 60)
    print(f"{config['center_name']} 최종 품목 마스터 생성 시작")

    df_list = []

    for file_name in config["files"]:

        file_path = MASTER_DIR / file_name

        print(f"파일 읽는 중: {file_name}")

        df = pd.read_excel(
            file_path,
            usecols=columns_to_use
        )

        df_list.append(df)

    all_df = pd.concat(
        df_list,
        ignore_index=True
    )

    total_rows = len(all_df)

    print(f"\n총 합 행 개수: {total_rows:,}")

    final_df = all_df.drop_duplicates(
        subset=["바코드", "상품명"]
    )

    final_rows = len(final_df)

    duplicate_rows = total_rows - final_rows

    print(f"중복 행 개수: {duplicate_rows:,}")
    print(f"중복 제거 후 최종 행 개수: {final_rows:,}")

    final_df = final_df.sort_values(
        by=[
            "대분류",
            "중분류",
            "소분류",
            "상품명",
            "바코드"
        ]
    )

    final_df = final_df.reset_index(drop=True)

    output_path = OUTPUT_DIR / config["output_name"]

    final_df.to_excel(
        output_path,
        index=False
    )

    print(f"저장 위치: {output_path}")

print("\nA/B센터 최종 품목 마스터 생성 완료")