"""
add_rollstd_log_features.py

data/final_feature_table.parquet에 신규 파생 변수 4개를 추가한다.

[신규 파생 변수 정의서]
| 새 컬럼명                        | 원본/기준 컬럼             | 생성 방식                          |
|-----------------------------------|-----------------------------|-------------------------------------|
| qty_lag1_filled_log1p             | qty_lag1_filled (기존)      | log1p                               |
| qty_rollmean_4_filled_log1p       | qty_rollmean_4_filled (기존)| log1p                               |
| qty_rollstd_4_filled              | qty_rollstd_4 (원본)        | 기존 causal fill(계단식 카테고리 fallback) 신규 적용 |
| qty_rollstd_4_filled_log1p        | qty_rollstd_4_filled (신규) | log1p                               |

qty_rollstd_4_filled은 archive/feature_engineering/feature_lag_rolling_calendar.py의
add_category_fallback() 원본 로직을 category_fallback.py로 분리하여 그대로 재사용한다
(소분류->중분류->대분류->센터전체_같은주->센터전체_widen->누적평균 6단계 causal
cascade). 이 함수는 원래 파이프라인에서 센터별로 개별 호출되므로(WEEK_COL 기준 그룹
통계가 센터 간 섞이면 안 됨), 여기서도 A/B를 분리해서 각각 적용한 뒤 다시 합친다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.feature_engineering.category_fallback import add_category_fallback

BASE_DIR = Path(__file__).resolve().parents[3]
TARGET_PATH = BASE_DIR / "data" / "final_feature_table.parquet"

CENTER_COL = "center_id"


def main():
    df = pd.read_parquet(TARGET_PATH)
    print(f"[로드] {TARGET_PATH} : {len(df):,}행 x {df.shape[1]}컬럼")

    parts = []
    for center in df[CENTER_COL].unique():
        part = df[df[CENTER_COL] == center].copy()
        part = add_category_fallback(part, ["qty_rollstd_4"])
        print(f"  [{center}] qty_rollstd_4_filled 생성 완료 "
              f"(잔여 결측 {part['qty_rollstd_4_filled'].isna().sum():,}건 — "
              "과거 참고 데이터 없는 시작 시점, 정상)")
        parts.append(part)

    df = pd.concat(parts, axis=0).sort_index()

    df = df.assign(
        qty_lag1_filled_log1p=np.log1p(df["qty_lag1_filled"]),
        qty_rollmean_4_filled_log1p=np.log1p(df["qty_rollmean_4_filled"]),
        qty_rollstd_4_filled_log1p=np.log1p(df["qty_rollstd_4_filled"]),
    )

    print(f"[저장] {TARGET_PATH} : {len(df):,}행 x {df.shape[1]}컬럼")
    df.to_parquet(TARGET_PATH, index=False)


if __name__ == "__main__":
    main()
