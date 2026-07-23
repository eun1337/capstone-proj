"""
최종 전처리 마스터 데이터셋 생성.

- 원본 정제 파일(cleaned_main_joined_cleaned_for_pred.parquet)에서 316개 (센터,상품키)
  조합(수량 부호반전 의심 상품, step_return_treatment_comparison.py에서 확정된 합집합 -
  기존 175개 센터무관 리스트를 실제 존재 센터로 확장 + 신규 244개 센터별 리스트)을
  drop한다.
- aggregate_demand_data.py의 주 단위 집계 로직을 그대로 재사용한다.
- 센터별로 완전히 분리해서 저장한다(B6: 완전분리 아키텍처 원칙).
- B센터는 레짐 컬럼 유무 두 버전으로 저장한다(컷오프 2023-07-01, changepoint
  재검증에서 반품률 기준 07-18로 확인됐으나 기존 분석과의 일관성을 위해 07-01 유지).
  A센터는 레짐 개념이 없어 한 버전만 저장한다.
- 이상치 주차(Step5, A 5건/B 4건)는 플래그하지 않는다(사용자 결정).
"""

import sys

import pandas as pd

sys.path.insert(0, "analysis-results")
sys.path.insert(0, "analysis-results/preprocessing")

from aggregate_demand_data import aggregate_weekly  # noqa: E402
from step_return_treatment_comparison import (  # noqa: E402
    PRODUCT_KEY_COLS, compare_with_old, identify_daily_candidates, load_daily,
)

OUT_A = "data/final/master_demand_weekly_A.parquet"
OUT_B_NO_REGIME = "data/final/master_demand_weekly_B_no_regime.parquet"
OUT_B_WITH_REGIME = "data/final/master_demand_weekly_B_with_regime.parquet"

REGIME_SHIFT_DATE = pd.Timestamp("2023-07-01")


def main() -> None:
    print("Loading raw daily data ...")
    df = load_daily()
    print(f"원본 전체 행 수: {len(df):,}")

    new_cand, _ = identify_daily_candidates(df)
    union_keys = compare_with_old(df, new_cand)

    mask = df.set_index(["센터"] + PRODUCT_KEY_COLS).index.isin(union_keys)
    before = len(df)
    df = df[~mask].copy()
    print(f"\n316개 상품×센터 조합 drop: {before - len(df):,}행 제거 (남은 행: {len(df):,})")

    print("\n주 단위 재집계 중 ...")
    weekly = aggregate_weekly(df)
    print(f"재집계 결과 행 수: {len(weekly):,}")

    a = weekly[weekly["센터"] == "A"].reset_index(drop=True)
    b = weekly[weekly["센터"] == "B"].reset_index(drop=True)

    a.to_parquet(OUT_A, index=False)
    print(f"\n저장 완료 -> {OUT_A} ({len(a):,}행)")

    b.to_parquet(OUT_B_NO_REGIME, index=False)
    print(f"저장 완료 -> {OUT_B_NO_REGIME} ({len(b):,}행)")

    b_with_regime = b.copy()
    b_with_regime["레짐"] = b_with_regime["주시작일"].apply(
        lambda d: "pre" if d < REGIME_SHIFT_DATE else "post"
    )
    b_with_regime.to_parquet(OUT_B_WITH_REGIME, index=False)
    n_pre = (b_with_regime["레짐"] == "pre").sum()
    n_post = (b_with_regime["레짐"] == "post").sum()
    print(f"저장 완료 -> {OUT_B_WITH_REGIME} ({len(b_with_regime):,}행, pre={n_pre:,}/post={n_post:,})")

    print("\n=== 요약 ===")
    print(f"A센터: {len(a):,}행, 기간 {a['주시작일'].min().date()} ~ {a['주시작일'].max().date()}")
    print(f"B센터: {len(b):,}행, 기간 {b['주시작일'].min().date()} ~ {b['주시작일'].max().date()}")


if __name__ == "__main__":
    main()
