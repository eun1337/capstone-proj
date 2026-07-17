"""
convert_zipcode_dtype.py

설명: 우편번호 컬럼을 문자열로 변환하여 앞자리 0 손실을 방지합니다.
      숫자로 저장되어 앞자리 0이 소실된 경우 zfill(5)로 복원합니다.
      매입 테이블: '공급업체 우편번호'
      매출 테이블: '매출처 우편번호'
"""

import os
import pandas as pd
from config import PARQUET_DIR

FILE_ZIPCODE_COL = {
    "a_center_purchase_2021_2024.parquet": "공급업체 우편번호",
    "a_center_sales_2024.parquet":         "매출처 우편번호",
    "a_center_sales_2021_2023.parquet":    "매출처 우편번호",
    "b_center_purchase_2021_2024.parquet": "공급업체 우편번호",
    "b_center_sales_2024.parquet":         "매출처 우편번호",
    "b_center_sales_2021_2023.parquet":    "매출처 우편번호",
}

print("[ 우편번호 타입 변환 작업 시작 ]")

for fname, zip_col in FILE_ZIPCODE_COL.items():
    path = os.path.join(PARQUET_DIR, fname)
    df = pd.read_parquet(path)

    if zip_col not in df.columns:
        print(f"\n- {fname}  ('{zip_col}' 컬럼 없음, 스킵)")
        continue

    def to_zipcode_str(x):
        if pd.isna(x) or str(x).strip() in ("", "nan"):
            return ""
        try:
            return str(int(float(x))).zfill(5)  # 앞에 0붙여 5자리로 복원
        except Exception:
            return str(x).strip().zfill(5)

    df[zip_col] = df[zip_col].apply(to_zipcode_str)
    df.to_parquet(path, index=False, engine="pyarrow")

    print(f"\n파일명: {fname}")
    print(f"    컬럼: '{zip_col}'")
    print(f"    샘플: {df[zip_col].dropna().head(3).tolist()}")

print("우편번호 타입 변환 완료")