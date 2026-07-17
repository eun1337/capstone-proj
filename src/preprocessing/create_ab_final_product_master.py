"""
create_ab_final_product_master.py

설명:
    A센터와 B센터의 최종 품목 마스터를 불러와
    하나의 통합 품목 마스터를 생성합니다.

    각 데이터에 '센터' 컬럼을 추가하여
    A센터 데이터는 'A',
    B센터 데이터는 'B'로 표시합니다.

입력:
    - data/master/a_final_product_master.xlsx
    - data/master/b_final_product_master.xlsx

출력:
    - data/master/ab_final_product_master.xlsx
"""

import pandas as pd
from pathlib import Path

MASTER_DIR = Path("data/master")
OUTPUT_DIR = Path("data/master")

print("=" * 60)
print("A+B 센터 통합 최종 품목 마스터 생성 시작")
print("=" * 60)

df_a = pd.read_excel(MASTER_DIR / "a_final_product_master.xlsx")
df_b = pd.read_excel(MASTER_DIR / "b_final_product_master.xlsx")

print(f"A센터 데이터 수 : {len(df_a):,}")
print(f"B센터 데이터 수 : {len(df_b):,}")


df_a["센터"] = "A"
df_b["센터"] = "B"


df_ab = pd.concat(
    [df_a, df_b],
    ignore_index=True
)

print(f"통합 데이터 수 : {len(df_ab):,}")

output_file = OUTPUT_DIR / "ab_final_product_master.xlsx"

df_ab.to_excel(
    output_file,
    index=False
)

print(f"통합 파일 저장 완료: {output_file}")

chunk_size = 16270

for i, start_idx in enumerate(range(0, len(df_ab), chunk_size), start=1):
    chunk_df = df_ab.iloc[start_idx:start_idx + chunk_size]

    chunk_file = (
        OUTPUT_DIR
        / f"ab_final_product_master_{i}.xlsx"
    )

    chunk_df.to_excel(
        chunk_file,
        index=False
    )

    print(
        f"분할 파일 저장 완료: "
        f"{chunk_file.name} "
        f"({len(chunk_df):,}행)"
    )