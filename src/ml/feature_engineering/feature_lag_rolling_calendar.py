"""
feature_lag_rolling_calendar.py
Day2 - Lag / Rolling / 캘린더 feature 생성

A/B 센터의 split 파일들(train/val/test 또는 train/pool)을 SKU 단위로 합쳐서
lag, rolling, 캘린더 feature를 한 번에 계산한 뒤, 원래 분할 기준으로 다시 나눠 저장한다.
(각 split 파일을 따로따로 계산하면 경계에서 가짜 웜업 NaN이 생기기 때문에 반드시 합쳐서 계산 후 재분할한다)

핵심 설계:
    1) Lag: A는 qty_lag1/2/4, B는 qty_lag1만(원본+log1p 버전 둘 다 생성)
    2) Rolling: 공통 qty_rollmean_4/qty_rollstd_4, 현재 주 포함 트레일링 윈도우
       (horizon이 t+h, h>=1이므로 리키지 없음). 창 안에 qty NaN(입고 전 구조적 결측)이
       섞여 min_periods 미달이면 자연히 NaN이 되고, 이 NaN이 웜업/콜드스타트 판정 신호로 쓰임
    3) 캘린더: 주차/월/분기만 생성 (week_st가 항상 월요일이라 요일 컬럼은 만들지 않음)
    4) is_warmup: SKU 자기 grid_start(입고 기준)로부터 lag/rolling 계산에 필요한 최소 주 수를
       아직 못 채운 행(cumcount 기반, reindex/lag 계산 목적이라 입고 기준이 맞음)
    5) coldstart_flag: 입고가 아니라 실제 첫 판매(출고, sold_flag==1) 시점부터 누적 관측
       주수 < threshold(8주)로 판정 — 콜드스타트의 본질은 "수요 패턴 관측량"이지 "재고 보유
       기간"이 아니기 때문. 아직 한 번도 안 팔린 행은 항상 True. B는 레짐 필터로 과거
       판매이력이 가려지므로, Day1에서 넘어온 existed_before_regime=True(레짐 이전부터
       팔리던 베테랑 SKU)인 행은 강제로 False 처리해 신상품 오판을 보정함
    6) 카테고리 fallback(계단식): 소분류(같은주) -> 중분류(같은주) -> 대분류(같은주) ->
       센터전체(같은주) -> 센터전체(과거 방향으로만 최대 max_widen_weeks주 확장) ->
       센터전체 누적(expanding, 그 시점까지) 평균. 5·6단계 모두 미래 시점 데이터를
       참조하지 않도록 과거 방향으로만 확장/누적함. fallback 적용 대상은 is_warmup/
       coldstart_flag 플래그가 아니라 "그 feature 컬럼 값이 실제로 NaN인가"로 컬럼별
       직접 판정(플래그만으로는 SKU 중간에 생기는 구조적 NaN을 못 잡기 때문).
       채우는 값은 항상 해당 feature 컬럼 자체의 다른 SKU 평균(원본 qty 평균이 아님)
    7) 원본(NaN 유지) 컬럼과 `_filled`(fallback 채움) 컬럼을 둘 다 보관.
       LightGBM 등 트리 기반은 원본+플래그 사용을 권장(NaN 자체가 신호),
       SVM/ARIMA 등 NaN 미지원 모델은 `_filled` 사용을 권장. 단 `_filled`도 데이터셋/센터
       시작 시점처럼 과거 참고 데이터가 전혀 없는 극소수 행엔 NaN이 그대로 남을 수 있음
       (의도된 동작 — qty 원본의 NaN=모름/0=앎·안팔림 구분 원칙을 feature 레벨에서도
       지키기 위해 억지로 0을 채우지 않음). 이런 모델은 자체 파이프라인에서 SimpleImputer나
       행 제거로 마지막 처리 필요
    8) fallback_summary(): 컬럼별 원본 NaN 개수를 분모로 fallback 단계별 적용 비율을 집계.
       process_center() 실행 시 자동 출력되며, 단계별로 못 채운 행은 "미채움"으로 표시됨
    9) weeks_since_last_active: sku_last_active_week(datetime, point-in-time ffill)을
       (week_st - sku_last_active_week)주 로 숫자 변환한 feature. 한 번도 안 팔린 행은
       NaN 유지(coldstart_flag/is_warmup이 이미 그 상태를 표시하므로 억지로 채우지 않음)
   10) get_excluded_cols(): target_* 컬럼 + ID 컬럼(center_id/sku_id/week_st) +
       진단용 `_fill_source` 컬럼 + 변환 필요 컬럼(sku_last_active_week 원본, datetime 타입이라
       모델 입력 불가 - 숫자로 변환된 weeks_since_last_active는 제외 대상 아님)을 합쳐 반환.
       Day11+ 학습 코드에서 import해서 feature_cols 구성에 사용. LightGBM/SVM 트랙별
       원본 vs `_filled` 선택은 이 함수에 포함되지 않으며 트랙별로 별도 처리 필요
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
SPLIT_DIR = BASE_DIR / "data" / "ml" / "splits"
OUT_DIR = SPLIT_DIR

SKU_COL = "sku_id"
WEEK_COL = "week_st"
QTY_COL = "qty"

A_LAG_WEEKS = [1, 2, 4]
B_LAG_WEEKS = [1]
ROLL_WINDOW = 4

CATEGORY_HIERARCHY = ["KAN_소분류", "KAN_중분류", "KAN_대분류"]

COLDSTART_THRESHOLD_WEEKS = 8

A_SPLIT_FILES = {
    "train": SPLIT_DIR / "A_train.parquet",
    "val": SPLIT_DIR / "A_val.parquet",
    "test": SPLIT_DIR / "A_test.parquet",
}
B_SPLIT_FILES = {
    "train": SPLIT_DIR / "B_train.parquet",
    "pool": SPLIT_DIR / "B_2024_pool.parquet",
}


def load_and_tag(split_files: dict) -> pd.DataFrame:
    frames = []
    for label, path in split_files.items():
        df = pd.read_parquet(path)
        df["_split"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_lag_features(df: pd.DataFrame, lag_weeks: list[int]) -> pd.DataFrame:
    df = df.sort_values([SKU_COL, WEEK_COL]).reset_index(drop=True)
    grouped = df.groupby(SKU_COL, sort=False)[QTY_COL]
    for n in lag_weeks:
        raw = grouped.shift(n)
        df[f"qty_lag{n}"] = raw
        df[f"qty_lag{n}_log1p"] = np.log1p(raw.clip(lower=0))
    return df


def add_rolling_features(df: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    df = df.sort_values([SKU_COL, WEEK_COL]).reset_index(drop=True)
    g = df.groupby(SKU_COL, sort=False)[QTY_COL]
    df[f"qty_rollmean_{window}"] = g.transform(
        lambda s: s.rolling(window, min_periods=window).mean()
    )
    df[f"qty_rollstd_{window}"] = g.transform(
        lambda s: s.rolling(window, min_periods=window).std()
    )
    df[f"qty_rollmean_{window}_log1p"] = np.log1p(
        df[f"qty_rollmean_{window}"].clip(lower=0)
    )
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    iso = df[WEEK_COL].dt.isocalendar()
    df["주차"] = iso["week"].astype(int)
    df["월"] = df[WEEK_COL].dt.month
    df["분기"] = df[WEEK_COL].dt.quarter
    return df


def add_weeks_since_active(df: pd.DataFrame) -> pd.DataFrame:
    """sku_last_active_week(point-in-time ffill, datetime)을 숫자형 feature로 변환.
    아직 한 번도 안 팔린 행(sku_last_active_week가 NaT)은 NaN 유지
    -> coldstart_flag/is_warmup이 이미 이 상태를 표시하므로 억지로 0/큰값을 채우지 않음."""
    weeks = (df[WEEK_COL] - df["sku_last_active_week"]).dt.days // 7
    df["weeks_since_last_active"] = weeks
    return df


def add_warmup_coldstart_flags(
    df: pd.DataFrame, lookback_weeks: int, coldstart_threshold: int
) -> pd.DataFrame:
    df = df.sort_values([SKU_COL, WEEK_COL]).reset_index(drop=True)
    position_in_series = df.groupby(SKU_COL, sort=False).cumcount()
    df["is_warmup"] = position_in_series < lookback_weeks

    sold_mask = df["sold_flag"] == 1
    first_sale_week = (
        df.loc[sold_mask].groupby(SKU_COL)[WEEK_COL].min().rename("_first_sale_week")
    )
    df = df.merge(first_sale_week, on=SKU_COL, how="left")

    never_sold_yet = df["_first_sale_week"].isna() | (df[WEEK_COL] < df["_first_sale_week"])
    weeks_since_first_sale = (df[WEEK_COL] - df["_first_sale_week"]).dt.days // 7 + 1
    df["coldstart_flag"] = never_sold_yet | (weeks_since_first_sale < coldstart_threshold)
    df = df.drop(columns=["_first_sale_week"])

    if "existed_before_regime" in df.columns:
        df.loc[df["existed_before_regime"], "coldstart_flag"] = False
    return df


def add_category_fallback(
    df: pd.DataFrame, feature_cols: list[str], max_widen_weeks: int = 8
) -> pd.DataFrame:
    missing_hierarchy_cols = [c for c in CATEGORY_HIERARCHY if c not in df.columns]
    if missing_hierarchy_cols:
        raise KeyError(
            f"{missing_hierarchy_cols} 컬럼이 split 파일에 없음 — 카테고리평균 fallback 불가. "
            "master_demand_weekly 스키마에서 KAN 분류 컬럼이 최종 출력에 남아있는지 먼저 확인 필요."
        )

    for col in feature_cols:
        filled = df[col].copy()
        source = pd.Series(pd.NA, index=df.index, dtype="object")
        needs_fallback = filled.isna()

        for cat_col in CATEGORY_HIERARCHY:
            mask = needs_fallback & filled.isna()
            if not mask.any():
                break
            cat_mean = df.groupby([cat_col, WEEK_COL])[col].transform("mean")
            newly = mask & cat_mean.notna()
            filled.loc[newly] = cat_mean.loc[newly]
            source.loc[newly] = cat_col

        mask = needs_fallback & filled.isna()
        if mask.any():
            week_mean = df.groupby(WEEK_COL)[col].transform("mean")
            newly = mask & week_mean.notna()
            filled.loc[newly] = week_mean.loc[newly]
            source.loc[newly] = "센터전체_같은주"

        mask = needs_fallback & filled.isna()
        if mask.any():
            week_sum = df.groupby(WEEK_COL)[col].sum()
            week_count = df.groupby(WEEK_COL)[col].count()
            for k in range(1, max_widen_weeks + 1):
                mask = needs_fallback & filled.isna()
                if not mask.any():
                    break
                for idx in df.index[mask]:
                    center_week = df.at[idx, WEEK_COL]
                    lo = center_week - pd.Timedelta(weeks=k)
                    hi = center_week
                    window = (week_sum.index >= lo) & (week_sum.index <= hi)
                    total_count = week_count[window].sum()
                    if total_count > 0:
                        filled.at[idx] = week_sum[window].sum() / total_count
                        source.at[idx] = f"센터전체_widen{k}주"

        mask = needs_fallback & filled.isna()
        if mask.any():
            weekly_sum = df.groupby(WEEK_COL)[col].sum().sort_index()
            weekly_count = df.groupby(WEEK_COL)[col].count().sort_index()
            cum_sum = weekly_sum.cumsum()
            cum_count = weekly_count.cumsum()
            expanding_mean = (cum_sum / cum_count).rename("_expanding_mean")
            row_expanding_mean = df.loc[mask, WEEK_COL].map(expanding_mean)
            filled.loc[mask] = row_expanding_mean
            source.loc[mask & filled.notna()] = "누적평균(그 시점까지)"

        df[f"{col}_filled"] = filled
        df[f"{col}_fill_source"] = source
    return df


def fallback_summary(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    total = len(df)
    rows = []
    for col in feature_cols:
        source_col = f"{col}_fill_source"
        original_na = df[col].isna()
        n_needs_fallback = int(original_na.sum())
        row = {
            "column": col,
            "total_rows": total,
            "n_needs_fallback": n_needs_fallback,
            "pct_needs_fallback": round(n_needs_fallback / total * 100, 2) if total else 0.0,
        }
        if n_needs_fallback:
            stage_labels = df.loc[original_na, source_col].fillna("미채움")
            filled_by_stage = stage_labels.value_counts()
            for stage, cnt in filled_by_stage.items():
                row[f"pct_{stage}"] = round(cnt / n_needs_fallback * 100, 2)
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


def process_center(
    label: str, split_files: dict, lag_weeks: list[int], out_dir: Path = OUT_DIR
) -> None:
    df = load_and_tag(split_files)

    df = add_lag_features(df, lag_weeks)
    df = add_rolling_features(df)
    df = add_calendar_features(df)
    df = add_weeks_since_active(df)

    lookback = max(lag_weeks + [ROLL_WINDOW])
    df = add_warmup_coldstart_flags(
        df, lookback_weeks=lookback, coldstart_threshold=COLDSTART_THRESHOLD_WEEKS
    )

    fallback_target_cols = [f"qty_lag{n}" for n in lag_weeks] + [f"qty_rollmean_{ROLL_WINDOW}"]
    df = add_category_fallback(df, fallback_target_cols)

    n_warmup = int(df["is_warmup"].sum())
    n_coldstart_rows = int(df["coldstart_flag"].sum())
    n_coldstart_sku = df.loc[df["coldstart_flag"], SKU_COL].nunique()
    print(
        f"[{label}] 전체 {len(df):,}행 / is_warmup {n_warmup:,}행 "
        f"({n_warmup / len(df):.1%}) / coldstart {n_coldstart_rows:,}행, "
        f"{n_coldstart_sku:,}개 SKU"
    )
    summary = fallback_summary(df, fallback_target_cols)
    print(f"[{label}] fallback 단계별 비율:")
    print(summary.to_string(index=False))

    any_remaining_nan = False
    for col in fallback_target_cols:
        n_remaining_nan = int(df[f"{col}_filled"].isna().sum())
        if n_remaining_nan > 0:
            any_remaining_nan = True
            print(
                f"  ℹ [{label}] {col}_filled 에 결측 {n_remaining_nan:,}건 — "
                "과거 참고 데이터가 전혀 없는 시점(데이터셋/센터 시작 시점 등)으로 예상됨, 정상"
            )
    if not any_remaining_nan:
        print(f"[{label}] 모든 fallback 대상 컬럼에 잔여 결측 없음 확인")

    for split_label in df["_split"].unique():
        part = df[df["_split"] == split_label].drop(columns=["_split"])
        out_path = out_dir / f"{label}_{split_label}_feat.parquet"
        part.to_parquet(out_path, index=False)
        print(f"  -> {out_path} ({len(part):,}행)")


def process_center_a() -> None:
    process_center("A", A_SPLIT_FILES, A_LAG_WEEKS)


def process_center_b() -> None:
    process_center("B", B_SPLIT_FILES, B_LAG_WEEKS)


ID_COLS = ["center_id", SKU_COL, WEEK_COL]

DIAGNOSTIC_COL_SUFFIXES = ("_fill_source",)

NEEDS_TRANSFORM_COLS = ["sku_last_active_week"]


def get_excluded_cols(df: pd.DataFrame) -> list[str]:
    target_cols = [c for c in df.columns if c.startswith("target_")]
    diagnostic_cols = [c for c in df.columns if c.endswith(DIAGNOSTIC_COL_SUFFIXES)]
    return ID_COLS + target_cols + diagnostic_cols + NEEDS_TRANSFORM_COLS


if __name__ == "__main__":
    process_center_a()
    process_center_b()