"""
split_complete_and_missing_classification.py

설명:
상품 마스터 데이터의 분류 품질 검증을 위해
대분류, 중분류, 소분류 컬럼의 결측 여부를 확인하고

- 분류 체계가 완전하게 입력된 상품
- 분류 체계에 결측이 존재하는 상품

을 각각 별도의 파일로 분리 저장합니다.

검증 대상 컬럼:
- 대분류
- 중분류
- 소분류

생성 파일:
- a_final_product_master_complete.xlsx
- a_final_product_master_missing.xlsx
- b_final_product_master_complete.xlsx
- b_final_product_master_missing.xlsx
"""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

MASTER_DIR = BASE_DIR / "data" / "master"

A_FILE = MASTER_DIR / "a_final_product_master.xlsx"
B_FILE = MASTER_DIR / "b_final_product_master.xlsx"


def split_classification(file_path):

    print(f"\n파일 처리 중: {file_path.name}")

    df = pd.read_excel(file_path)

    original_count = len(df)

    df.columns = df.columns.str.strip()

    required_cols = ["대분류", "중분류", "소분류"]


    for col in required_cols:

        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
        )

        df[col] = df[col].replace(
            [
                "",
                " ",
                "nan",
                "NaN",
                "None",
                "NULL",
                "N/A",
                "-"
            ],
            pd.NA
        )


    complete_df = df.dropna(
        subset=required_cols
    )


    missing_df = df[
        df[required_cols]
        .isna()
        .any(axis=1)
    ]


    complete_path = file_path.with_name(
        file_path.stem + "_complete.xlsx"
    )

    missing_path = file_path.with_name(
        file_path.stem + "_missing.xlsx"
    )

    complete_df.to_excel(
        complete_path,
        index=False
    )

    missing_df.to_excel(
        missing_path,
        index=False
    )

    print(f"원본 행 수      : {original_count:,}")
    print(f"완전 분류 상품  : {len(complete_df):,}")
    print(f"결측 포함 상품  : {len(missing_df):,}")
    print(
        f"검증 합계       : "
        f"{len(complete_df) + len(missing_df):,}"
    )

    print(f"저장 완료 : {complete_path.name}")
    print(f"저장 완료 : {missing_path.name}")



split_classification(A_FILE)
split_classification(B_FILE)

print("\n모든 작업 완료")