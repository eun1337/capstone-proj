"""
drop_purchase_retail_col.py

설명: 매입 데이터에서 결측치가 매우 높은 매출처 관련 컬럼을 삭제합니다.
      삭제 대상: '매출처코드', '매출처 우편번호'
"""

import os
import pandas as pd
from config import PARQUET_DIR

FILES = [
    "a_center_purchase_2021_2024.parquet",
    "b_center_purchase_2021_2024.parquet",
]

DROP_COLS = ["매출처코드", "매출처 우편번호"]

print("[ 매입 데이터 매출처 컬럼 삭제 작업 시작 ]")

for fname in FILES:
    path = os.path.join(PARQUET_DIR, fname)
    df = pd.read_parquet(path)
    to_drop = [col for col in DROP_COLS if col in df.columns]

    if not to_drop:
        print(f"\n- {fname}  (삭제 대상 컬럼 없음, 스킵)")
        continue

    before_cols = df.shape[1]
    df = df.drop(columns=to_drop)
    after_cols = df.shape[1]

    df.to_parquet(path, index=False, engine="pyarrow")
    print(f"\n파일명: {fname}")
    print(f"    삭제 컬럼: {to_drop}")
    print(f"    컬럼 수: {before_cols}개  →  {after_cols}개")

print("\n매출처 컬럼 삭제 작업 완료")