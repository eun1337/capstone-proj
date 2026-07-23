"""
normalize_option_code.py

설명: data/final/cleaned_main_joined.parquet의 '옵션코드' 컬럼을
     EA / BX / CS 세 값으로 통일합니다.
     csv 동기화는 별도로 처리하므로 이 스크립트는 parquet만 갱신합니다.

[매핑 근거 - analyze_option_code.py 분석 결과]
- EA, ea                              -> EA (낱개 표기 차이)
- BX, box, BOX, bx, 박스               -> BX (박스 표기 차이)
- BX1, BX2, BX3, BX4                  -> BX
    * BX2/BX3는 BX와 바코드가 겹치는 상품에서 단가가 거의 동일(비율 0.7~1.1)해
      실제로는 BX와 같은 뜻으로 표기만 다르게 입력된 것으로 판단
    * BX1/BX4는 겹치는 바코드가 적어 직접 검증은 못했으나 동일 계열로 간주해 병합
- CS, cs, C/S                         -> CS (케이스 표기 차이)
- CS1, CS2, C/S1                      -> CS
    * CS와 겹치는 바코드는 없었지만(표본 1~2건) 동일 계열로 간주해 병합
"""

import os

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FINAL_DIR = os.path.join(BASE_DIR, "data", "final")
PARQUET_FILE = os.path.join(FINAL_DIR, "cleaned_main_joined.parquet")

COL = "옵션코드"

OPTION_MAP = {
    "EA": "EA",
    "ea": "EA",
    "BX": "BX",
    "box": "BX",
    "BOX": "BX",
    "bx": "BX",
    "박스": "BX",
    "BX1": "BX",
    "BX2": "BX",
    "BX3": "BX",
    "BX4": "BX",
    "CS": "CS",
    "cs": "CS",
    "C/S": "CS",
    "CS1": "CS",
    "CS2": "CS",
    "C/S1": "CS",
}


def main():
    print(f"[ 옵션코드 정규화 시작 ] {PARQUET_FILE}")
    df = pd.read_parquet(PARQUET_FILE)

    before_counts = df[COL].value_counts(dropna=False)

    unmapped = set(df[COL].dropna().unique()) - set(OPTION_MAP.keys())
    if unmapped:
        raise ValueError(
            f"OPTION_MAP에 없는 값이 발견되었습니다: {sorted(unmapped)}\n"
            "OPTION_MAP에 매핑을 추가한 뒤 다시 실행하세요."
        )

    df[COL] = df[COL].map(OPTION_MAP).astype(df[COL].dtype)

    after_counts = df[COL].value_counts(dropna=False)

    print("\n[ 변경 전 ]")
    print(before_counts)
    print("\n[ 변경 후 ]")
    print(after_counts)

    df.to_parquet(PARQUET_FILE, index=False, engine="pyarrow")
    print(f"\n저장 완료: {PARQUET_FILE}")


if __name__ == "__main__":
    main()
