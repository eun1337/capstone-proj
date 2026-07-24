"""
feature_lag_rolling_calendar.py
Day2 - Lag / Rolling / 캘린더 feature 생성

A/B 센터의 split 파일들(train/val/test 또는 train/pool)을 SKU 단위로 합쳐서
lag, rolling, 캘린더 feature를 한 번에 계산한 뒤, 원래 분할 기준으로 다시 나눠 저장한다.
(각 split 파일을 따로따로 계산하면 경계에서 가짜 웜업 NaN이 생기기 때문에 반드시 합쳐서 계산 후 재분할한다)
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

# 콜드스타트 판정 기준: SKU 전체 관측 기간(grid 상 행 수, qty 값 유무와 무관)이
# 이 값 미만이면 콜드스타트로 간주. A/B 공통 단일값이되, 데이터가 더 적은 B(train 26주) 기준으로
# 과도하게 플래그되지 않도록 보수적으로 시작 — 일단 feature 계산 최소선(ROLL_WINDOW)으로 잡고,
# 실제 B 데이터로 한번 돌려 아래 print되는 coldstart 비율을 보고 필요시에만 올릴 것(예: 8주).
# A는 데이터가 훨씬 많아 이 값을 올려도 영향이 작으므로, B가 감당 가능한 값 위주로 조정.
COLDSTART_THRESHOLD_WEEKS = ROLL_WINDOW

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
    """split 파일들을 읽어 origin 표시 컬럼(_split)을 붙여 하나로 합친다."""
    frames = []
    for label, path in split_files.items():
        df = pd.read_parquet(path)
        df["_split"] = label
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_lag_features(df: pd.DataFrame, lag_weeks: list[int]) -> pd.DataFrame:
    """sku_id 그룹 내에서만 shift (다른 SKU 값이 섞이지 않도록)."""
    df = df.sort_values([SKU_COL, WEEK_COL]).reset_index(drop=True)
    grouped = df.groupby(SKU_COL, sort=False)[QTY_COL]
    for n in lag_weeks:
        raw = grouped.shift(n)
        df[f"qty_lag{n}"] = raw
        df[f"qty_lag{n}_log1p"] = np.log1p(raw.clip(lower=0))
    return df


def add_rolling_features(df: pd.DataFrame, window: int = ROLL_WINDOW) -> pd.DataFrame:
    """최근 window주 이동평균/표준편차. 현재 주 포함 트레일링 윈도우.
    창 안에 qty NaN(입고 전 구조적 결측)이 섞여 min_periods 미달이면 자연히 NaN이 됨
    -> 이 NaN이 바로 '웜업/콜드스타트' 판정에 쓰이는 신호."""
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
    """주차/월/분기. week_st가 항상 월요일이라 요일 컬럼은 만들지 않음."""
    iso = df[WEEK_COL].dt.isocalendar()
    df["주차"] = iso["week"].astype(int)
    df["월"] = df[WEEK_COL].dt.month
    df["분기"] = df[WEEK_COL].dt.quarter
    return df


def add_warmup_coldstart_flags(
    df: pd.DataFrame, lookback_weeks: int, coldstart_threshold: int
) -> pd.DataFrame:
    """
    is_warmup: SKU 자기 grid_start로부터 lag/rolling 계산에 필요한 최소 주 수를 아직 못 채운 행
               (df가 이미 grid_start부터 시작하는 reindex된 캘린더라고 가정)
    coldstart_flag: SKU 전체 관측 기간(행 수)이 임계값 미만인 SKU 전체
    두 플래그 모두 '카테고리평균 fallback' 대상 표시용.
    """
    df = df.sort_values([SKU_COL, WEEK_COL]).reset_index(drop=True)
    position_in_series = df.groupby(SKU_COL, sort=False).cumcount()
    df["is_warmup"] = position_in_series < lookback_weeks

    sku_series_length = df.groupby(SKU_COL, sort=False)[WEEK_COL].transform("count")
    df["coldstart_flag"] = sku_series_length < coldstart_threshold
    return df


def add_category_fallback(
    df: pd.DataFrame, feature_cols: list[str], max_widen_weeks: int = 8
) -> pd.DataFrame:
    """
    is_warmup/coldstart_flag과 무관하게, feature 컬럼 값이 실제로 NaN인 행이면 전부 fallback 대상으로 삼음.
    채우는 값은 항상 '해당 feature 컬럼 자체'의 다른 SKU 평균.

    단계:
      1~3. 소분류 -> 중분류 -> 대분류, 같은 week_st 기준 카테고리 평균
      4. 센터 전체(같은 파일 내 모든 SKU), 같은 week_st 기준 평균
      5. 그래도 없으면(그 주에 센터 전체가 결측인 극단 케이스) 센터 전체를 기준으로
         week_st ±1주 -> ±2주 -> ... -> ±max_widen_weeks까지 점점 넓혀가며 평균
      6. 그래도 안 채워지면(사실상 발생 안 함) feature 전체 기간 평균으로 최종 보정
    원본 feature 컬럼은 그대로 두고 <col>_filled 컬럼을 별도로 만듦
    <col>_fill_source 컬럼에 어느 단계에서 채워졌는지 기록(원래 값 있던 행은 NaN).
    fallback_summary()로 컬럼별/단계별 비율을 집계할 수 있음.

    사용 가이드(강제 아님, 다운스트림에서 선택):
      - Day11 LightGBM 등 트리 기반: 원본 컬럼(qty_lag4 등) + is_warmup/coldstart_flag 그대로 사용 추천
        (NaN 자체가 정보이므로 억지로 채우지 않는 편이 나을 수 있음)
      - Day13 Kernel SVM, 통계 트랙(ARIMA 등) 등 NaN을 못 받는 모델: <col>_filled 사용
    """
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

        # 1~3단계: 소분류 -> 중분류 -> 대분류, 같은 week_st 기준
        for cat_col in CATEGORY_HIERARCHY:
            mask = needs_fallback & filled.isna()
            if not mask.any():
                break
            cat_mean = df.groupby([cat_col, WEEK_COL])[col].transform("mean")
            newly = mask & cat_mean.notna()
            filled.loc[newly] = cat_mean.loc[newly]
            source.loc[newly] = cat_col

        # 4단계: 센터 전체, 같은 week_st 기준
        mask = needs_fallback & filled.isna()
        if mask.any():
            week_mean = df.groupby(WEEK_COL)[col].transform("mean")
            newly = mask & week_mean.notna()
            filled.loc[newly] = week_mean.loc[newly]
            source.loc[newly] = "센터전체_같은주"

        # 5단계: 센터 전체 기준으로 앞뒤 주를 점점 넓혀가며 평균 (극단적으로 그 주 전체가 결측일 때만 작동)
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
                    hi = center_week + pd.Timedelta(weeks=k)
                    window = (week_sum.index >= lo) & (week_sum.index <= hi)
                    total_count = week_count[window].sum()
                    if total_count > 0:
                        filled.at[idx] = week_sum[window].sum() / total_count
                        source.at[idx] = f"센터전체_widen{k}주"

        # 6단계(최종 보정): 그래도 안 채워지면 feature 전체 기간 평균
        mask = needs_fallback & filled.isna()
        if mask.any():
            filled.loc[mask] = df[col].mean()
            source.loc[mask] = "전체기간평균"

        df[f"{col}_filled"] = filled
        df[f"{col}_fill_source"] = source
    return df


def fallback_summary(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    컬럼별로: 전체 행 수 대비 fallback 대상(is_warmup|coldstart) 비율,
    그리고 fallback 대상 중 어느 단계에서 채워졌는지 비율을 정리한 표를 반환.
    """
    total = len(df)
    rows = []
    for col in feature_cols:
        source_col = f"{col}_fill_source"
        filled_by_stage = df[source_col].value_counts(dropna=True)
        n_needs_fallback = int(df[source_col].notna().sum())
        row = {
            "column": col,
            "total_rows": total,
            "n_needs_fallback": n_needs_fallback,
            "pct_needs_fallback": round(n_needs_fallback / total * 100, 2) if total else 0.0,
        }
        for stage, cnt in filled_by_stage.items():
            row[f"pct_{stage}"] = round(cnt / n_needs_fallback * 100, 2) if n_needs_fallback else 0.0
        rows.append(row)
    return pd.DataFrame(rows).fillna(0.0)


def process_center(
    label: str, split_files: dict, lag_weeks: list[int], out_dir: Path = OUT_DIR
) -> None:
    df = load_and_tag(split_files)

    df = add_lag_features(df, lag_weeks)
    df = add_rolling_features(df)
    df = add_calendar_features(df)

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
            print(f"  ⚠ [{label}] {col}_filled 에 결측 {n_remaining_nan:,}건 남음 — cascade 재점검 필요")
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


if __name__ == "__main__":
    process_center_a()
    process_center_b()