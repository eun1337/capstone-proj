"""
build_b_full_history.py
Option3(A+B Pooling + 24개월 Rolling Window CV) 전용 B센터 전체이력 feature 생성.

기존 프로덕션 파이프라인(split_train_val_test.py)은 B의 `pre` 레짐(2023-07 이전,
데이터 품질 이슈로 팀이 제외 결정)을 학습 대상에서 뺀다. 하지만 이 CV 실험의
Option3 Fold1 Val(2023-01~06)은 그 pre 레짐 구간과 겹치고, 사용자가 "pre 레짐
포함해서 원안 그대로" 진행하기로 확정했다(plan 문서 참고).

master_demand_weekly_B_with_regime.parquet의 pre(2020-12-28~2023-06-26)/
post(2023-07-03~2024-12-30) 레짐은 실제로는 공백 없이 이어지는 연속 주간
시계열이라, 레짐 필터를 걷지 않고 그대로 build_center_df(regime_filter=None)를
호출하면 자연스럽게 연속 그리드가 만들어진다.

split_train_val_test.py / feature_lag_rolling_calendar.py /
feature_external_interaction_concat.py의 함수를 그대로 import해서 재사용하며,
이 세 파일은 전혀 수정하지 않는다. 산출물은 기존 data/ml/splits/가 아니라
data/ml/cv_experiment/B_full_feat.parquet에 별도로 저장한다(기존 B_train_feat/
B_pool_feat.parquet는 건드리지 않음).

이 실험은 반품(returns) 데이터를 전혀 쓰지 않고 출고(총판매수량) 데이터만으로
진행한다. split_train_val_test.build_center_df()는 내부적으로 raw를 읽어
총판매수량과 반품수량을 함께 집계하고 save_returns_table()로 반품수량을 별도
파일에 저장하는데, 이 실험에서는 그 흐름 자체를 타지 않도록 raw 로드 직후
'반품수량' 컬럼을 제거한다(load_raw()를 얇게 감싸서 monkeypatch). 그러면
aggregate_region_to_sku()의 집계 대상에서 반품수량이 자동으로 빠지고,
save_returns_table()도 자체 가드(대상 컬럼이 없으면 return)에 걸려 아무
파일도 만들지 않는다 — split_train_val_test.py 자체는 수정하지 않으면서
반품 관련 로직이 전혀 실행되지 않게 하는 방식.
"""

from pathlib import Path
import sys

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[4]
FEATURE_ENG_DIR = BASE_DIR / "src" / "ml" / "feature_engineering"
sys.path.insert(0, str(FEATURE_ENG_DIR))

import split_train_val_test as svt  # noqa: E402
from split_train_val_test import load_first_stock_map, build_center_df, FILE_B  # noqa: E402
from feature_lag_rolling_calendar import (  # noqa: E402
    add_lag_features,
    add_rolling_features,
    add_calendar_features,
    add_weeks_since_active,
    add_warmup_coldstart_flags,
    add_sbc_features,
    add_category_fallback,
    fallback_summary,
    B_LAG_WEEKS,
    ROLL_WINDOW,
    COLDSTART_THRESHOLD_WEEKS,
)
from feature_external_interaction_concat import (  # noqa: E402
    add_holiday_calendar_features,
    add_climate_extreme_flags,
    add_covid_flag,
    add_interaction_features,
    add_weeks_since_active_filled,
)

OUT_DIR = BASE_DIR / "data" / "ml" / "cv_experiment"
OUT_PATH = OUT_DIR / "B_full_feat.parquet"

RETURNS_COL = "반품수량"

_original_load_raw = svt.load_raw


def _load_raw_demand_only(path):
    """svt.load_raw()를 감싸서 반품수량 컬럼을 즉시 제거 — 이후 build_center_df 내부의
    모든 집계/저장 로직(aggregate_region_to_sku, save_returns_table)이 반품수량을
    아예 보지 못하게 해 반품 관련 흐름을 완전히 비활성화한다. 출고(총판매수량)만 남는다."""
    df = _original_load_raw(path)
    if RETURNS_COL in df.columns:
        df = df.drop(columns=[RETURNS_COL])
    return df


svt.load_raw = _load_raw_demand_only

# common.TARGET_COL이 target_h1이므로 이 horizon만 있으면 충분(프로덕션 B_HORIZONS=[1,4]와
# 달리 h4는 이 실험에서 쓰지 않아 생략 — 계산량 절감).
B_HORIZONS_FULL = [1]

REGIME_CUTOFF = pd.Timestamp("2023-07-01")


def build_b_full_history() -> pd.DataFrame:
    print("=== B센터 전체이력(pre+post 레짐 통합, regime_filter=None) feature 생성 ===")
    first_stock_map = load_first_stock_map()

    df = build_center_df(
        FILE_B, B_HORIZONS_FULL, first_stock_map,
        center_label="B_full", regime_filter=None,
    )
    print(
        f"  build_center_df(regime_filter=None): {len(df):,}행, "
        f"{df['week_st'].min().date()} ~ {df['week_st'].max().date()}"
    )

    df = add_lag_features(df, B_LAG_WEEKS)
    df = add_rolling_features(df, window=ROLL_WINDOW)
    df = add_calendar_features(df)
    df = add_weeks_since_active(df)

    lookback = max(B_LAG_WEEKS + [ROLL_WINDOW])
    df = add_warmup_coldstart_flags(
        df, lookback_weeks=lookback, coldstart_threshold=COLDSTART_THRESHOLD_WEEKS
    )
    df = add_sbc_features(df)

    fallback_target_cols = [f"qty_lag{n}" for n in B_LAG_WEEKS] + [
        f"qty_rollmean_{ROLL_WINDOW}", "adi_expanding", "cv2_expanding",
    ]
    df = add_category_fallback(df, fallback_target_cols)

    n_warmup = int(df["is_warmup"].sum())
    n_coldstart_rows = int(df["coldstart_flag"].sum())
    print(
        f"  is_warmup {n_warmup:,}행 ({n_warmup / len(df):.1%}) / "
        f"coldstart {n_coldstart_rows:,}행"
    )
    print(fallback_summary(df, fallback_target_cols).to_string(index=False))

    df = add_holiday_calendar_features(df)
    df = add_climate_extreme_flags(df)
    df = add_covid_flag(df)
    df = add_interaction_features(df)
    df = add_weeks_since_active_filled(df)

    df = df.drop(columns=["공휴일"], errors="ignore")

    n_pre_regime_rows = int((df["week_st"] < REGIME_CUTOFF).sum())
    print(
        f"  pre 레짐 구간(2023-07 이전) 포함 행수: {n_pre_regime_rows:,} "
        f"(프로덕션 파이프라인은 이 구간을 전부 제외함 — Option3 CV 실험 전용으로만 포함)"
    )

    return df


def main():
    df = build_b_full_history()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print()
    print(f"저장 완료 -> {OUT_PATH} ({len(df):,}행, {len(df.columns)}개 컬럼)")
    print(f"날짜 범위: {df['week_st'].min().date()} ~ {df['week_st'].max().date()}")


if __name__ == "__main__":
    main()
