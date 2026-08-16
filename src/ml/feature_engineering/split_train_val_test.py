"""
split_train_val_test.py

설명: A/B센터 주간 수요 데이터를 Train/Val/Test로 분할하고 horizon 타겟을 생성합니다.
    A_HORIZONS=[1,2,4,8], B_HORIZONS=[1,2,4] (h2는 target_h*_공휴일_W0/-1/+1 horizon별
    피처 도입을 위해 추가됨 — feature_external_interaction_concat.py 참고).
    파일 위치: src/ml/feature_engineering/split_train_val_test.py (BASE_DIR=parents[3])

산출물 (data/ml/splits/):
    A_train.parquet / A_val.parquet / A_test.parquet
    B_train.parquet / B_2024_pool.parquet / B_walkforward_folds.json
    A_returns.parquet / B_returns.parquet (반품수량, 메인 파일과 분리 저장)

핵심 설계:
    1) 지역(시도/시군구) 합산 후 센터+SKU 단위로 축소
    2) target_h{h}(원본수량) + target_h{h}_flag(분류)/target_h{h}_qty_log1p(회귀) 3종 세트 생성
    3) qty가 희소 데이터라 shift 전에 sku_id별 주간 캘린더 그리드로 reindex
       - grid_start = min(purchase 최초입고일, 첫 관측 주), data_floor로 clip, 월요일 스냅
       - stock_week은 grid_start와 동일한 값을 사용 — grid_start 자체가 이미
         "SKU가 존재한다고 확신할 수 있는 가장 이른 시점"이므로 별도 계산 시 발생하는
         구조적 결측(최초입고일이 실제 첫거래보다 늦게 기록된 SKU에서 grid_start~stock_week
         구간이 "입고 전이라 NaN"으로 잘못 처리되는 문제)이 원천적으로 없음
       - 미관측 주: 입고일(stock_week) 이후는 0(진짜 미판매), 입고일 이전은 NaN(재고 없어 수요 유무 모름)
    4) 미래(week_st+h주 > 데이터셋 마지막 관측일)는 NaN, drop하지 않고 유지 (dropna는 학습 시점에)
    5) B센터는 _with_regime 파일 사용, 레짐='post'(2023-07~)만 학습 데이터로 사용.
       레짐 필터로 가려지는 레짐 이전 판매 이력은 existed_before_regime 컬럼으로 별도 보존해
       레짐 이후 첫 판매만 보고 신상품(콜드스타트)으로 오판되지 않도록 함
    6) B walk-forward: val_weeks=1, step_weeks=1 (rolling-origin)
    7) A/B는 항상 분리 저장, 통합은 학습 시점에 코드 레벨에서 처리
    8) sku_last_active_week은 SKU 전체가 아니라 그 행 시점까지 누적(ffill)으로 계산해
       미래 시점의 판매 지속 여부가 과거 행에 섞여 들어가지 않도록 함
"""


from pathlib import Path
import json
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "final"
OUT_DIR = BASE_DIR / "data" / "ml" / "splits"

FILE_A = DATA_DIR / "master_demand_weekly_A.parquet"
FILE_B = DATA_DIR / "master_demand_weekly_B_with_regime.parquet"
FILE_PURCHASE = DATA_DIR / "cleaned_purchase_cleaned_for_pred.parquet"

# Weather Region Assignment 재구현 산출물(analysis-results/preprocessing/
# weather_region_assignment.py, dev-data-preprocessing 브랜치)을 SKU-week 단위로 병합한다.
# 기존 CALENDAR_MEAN_COLS의 평균온도/총강수량(센터-week 단일값을 모든 SKU에 broadcast,
# 총강수량 거래건수만큼 중복 합산 버그 있었음)을 완전히 대체 — 아래 CALENDAR_MEAN_COLS에서
# 두 컬럼 제거하고 이 파일에서만 공급한다.
WEATHER_SKU_WEEK_FILES = {
    "A": DATA_DIR / "weather_sku_week_A.parquet",
    "B": DATA_DIR / "weather_sku_week_B.parquet",
}
WEATHER_SKU_WEEK_RAW_SKU_COLS = ["바코드", "옵션코드", "상품클러스터"]
# region_rule/n_regions/fallback_weeks_back/is_missing/weather_day_count: 진단용 —
# 최종 테이블엔 유지하되 학습 피처에서는 feature_lag_rolling_calendar.py의
# get_excluded_cols()(WEATHER_REGION_DIAGNOSTIC_COLS)에서 명시적으로 제외한다.
WEATHER_SKU_WEEK_VALUE_COLS = ["평균온도", "총강수량"]
WEATHER_SKU_WEEK_DIAGNOSTIC_COLS = ["region_rule", "n_regions", "fallback_weeks_back", "is_missing", "weather_day_count"]

RAW_WEEK_COL = "주시작일"
RAW_SKU_PARTS = ["바코드", "옵션코드", "상품클러스터"]
RAW_CENTER_COL = "센터"
RAW_REGION_COLS = ["시도", "시군구"]
RAW_REGIME_COL = "레짐"
RAW_QTY_COL = "총판매수량"

PURCHASE_KEY_COLS = ["센터", "바코드", "옵션코드", "상품클러스터"]
PURCHASE_FIRST_STOCK_COL = "최초입고일"

WEEK_COL = "week_st"
CENTER_COL = "center_id"
SKU_COL = "sku_id"
QTY_COL = "qty"
SKU_SEP = "*"

SUM_COLS = ["총판매수량", "반품수량"]
# 평균온도/총강수량은 더 이상 여기서 공급하지 않음(위 WEATHER_SKU_WEEK_FILES 참고) —
# 센터-week 단일값을 모든 SKU에 그대로 broadcast하던 방식은 SKU가 실제로 어느 지역에서
# 팔렸는지 반영하지 못했고, 총강수량은 거래 라인 수만큼 중복 합산되는 버그가 있었음
# (사전 조사에서 최대 99배, 데이터셋 전체 평균 1.46배 과대평가 확인됨).
CALENDAR_MEAN_COLS = ["cpi", "경상지수", "불변지수"]
CALENDAR_MAX_COLS = ["공휴일"]
CALENDAR_FIRST_COLS = ["ISO_연도", "ISO_주차", "covid_영향여부"]

SKU_STATIC_COLS = ["상품명", "규격", "입수", "KAN_CODE", "KAN_대분류", "KAN_중분류", "KAN_소분류"]

A_TRAIN_START = "2021-01-01"
A_TRAIN_END = "2023-12-31"
A_VAL_START = "2024-01-01"
A_VAL_END = "2024-06-30"
A_TEST_START = "2024-07-01"
A_TEST_END = "2024-12-31"
A_HORIZONS = [1, 2, 4, 8]

B_TRAIN_START = "2023-07-01"
B_TRAIN_END = "2023-12-31"
B_WF_POOL_START = "2024-01-01"
B_WF_POOL_END = "2024-12-31"
B_HORIZONS = [1, 2, 4]

B_WF_VAL_WEEKS = 1
B_WF_STEP_WEEKS = 1


def load_first_stock_map() -> pd.DataFrame:
    if not FILE_PURCHASE.exists():
        print(f"  경고: {FILE_PURCHASE.name} 없음 - 최초입고일 매핑 생략, 첫 관측 주로 대체")
        return pd.DataFrame(columns=PURCHASE_KEY_COLS + [PURCHASE_FIRST_STOCK_COL])
    pdf = pd.read_parquet(FILE_PURCHASE)
    missing = set(PURCHASE_KEY_COLS + [PURCHASE_FIRST_STOCK_COL]) - set(pdf.columns)
    if missing:
        print(f"  경고: {FILE_PURCHASE.name}에 예상 컬럼 없음: {missing} / 실제: {list(pdf.columns)}")
        return pd.DataFrame(columns=PURCHASE_KEY_COLS + [PURCHASE_FIRST_STOCK_COL])
    pdf[PURCHASE_FIRST_STOCK_COL] = pd.to_datetime(pdf[PURCHASE_FIRST_STOCK_COL])
    keep = PURCHASE_KEY_COLS + [PURCHASE_FIRST_STOCK_COL]
    return (
        pdf[keep]
        .groupby(PURCHASE_KEY_COLS, as_index=False)[PURCHASE_FIRST_STOCK_COL]
        .min()
    )


def load_raw(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    missing = set(RAW_SKU_PARTS + [RAW_WEEK_COL, RAW_CENTER_COL, RAW_QTY_COL]) - set(df.columns)
    if missing:
        raise KeyError(f"{path.name}에 예상 컬럼 없음: {missing} / 실제: {list(df.columns)}")
    df[RAW_WEEK_COL] = pd.to_datetime(df[RAW_WEEK_COL]).astype("datetime64[ns]")
    return df


def build_calendar_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    agg_map = {c: "mean" for c in CALENDAR_MEAN_COLS if c in raw_df.columns}
    agg_map.update({c: "max" for c in CALENDAR_MAX_COLS if c in raw_df.columns})
    agg_map.update({c: "first" for c in CALENDAR_FIRST_COLS if c in raw_df.columns})
    cal = raw_df.groupby(RAW_WEEK_COL, as_index=False).agg(agg_map)
    return cal.rename(columns={RAW_WEEK_COL: WEEK_COL})


def load_sku_week_weather(center_label: str) -> pd.DataFrame:
    """Weather Region Assignment 산출물(SKU-week grain)을 로드해 이 파일의 키 형식
    (center_id/sku_id/week_st)으로 맞춘다. sku_id는 aggregate_region_to_sku()와 동일한
    바코드+SEP+옵션코드+SEP+상품클러스터 조합 방식을 그대로 재현해야 병합 키가 맞는다
    (실측 검증: A는 2,248,073건 완전 1:1 일치, B는 다운스트림 grid의 600,032건이 전부
    이 테이블에 포함됨을 확인함 — B는 레짐 이전 구간까지 포함한 더 넓은 테이블이라
    반드시 이 함수의 결과가 아니라 build_center_df()의 reindexed 쪽을 driving table로
    LEFT JOIN해야 레짐 이전 초과분이 자동으로 걸러진다).
    """
    path = WEATHER_SKU_WEEK_FILES[center_label]
    df = pd.read_parquet(path)
    df[SKU_COL] = (
        df[WEATHER_SKU_WEEK_RAW_SKU_COLS[0]].astype(str) + SKU_SEP +
        df[WEATHER_SKU_WEEK_RAW_SKU_COLS[1]].astype(str) + SKU_SEP +
        df[WEATHER_SKU_WEEK_RAW_SKU_COLS[2]].astype(str)
    )
    df = df.rename(columns={"센터": CENTER_COL, "주시작일": WEEK_COL})
    keep = [CENTER_COL, SKU_COL, WEEK_COL] + WEATHER_SKU_WEEK_VALUE_COLS + WEATHER_SKU_WEEK_DIAGNOSTIC_COLS
    return df[keep]


def aggregate_region_to_sku(raw_df: pd.DataFrame) -> pd.DataFrame:
    group_cols = [RAW_WEEK_COL, RAW_CENTER_COL] + RAW_SKU_PARTS
    agg_map = {c: "sum" for c in SUM_COLS if c in raw_df.columns}
    agg_map.update({c: "first" for c in SKU_STATIC_COLS if c in raw_df.columns})
    out = raw_df.groupby(group_cols, sort=False, as_index=False).agg(agg_map)

    out[SKU_COL] = (
        out[RAW_SKU_PARTS[0]].astype(str) + SKU_SEP +
        out[RAW_SKU_PARTS[1]].astype(str) + SKU_SEP +
        out[RAW_SKU_PARTS[2]].astype(str)
    )
    out = out.rename(columns={RAW_WEEK_COL: WEEK_COL, RAW_CENTER_COL: CENTER_COL, RAW_QTY_COL: QTY_COL})
    return out


def save_returns_table(center_label: str, agg_df: pd.DataFrame):
    cols = [CENTER_COL, SKU_COL, WEEK_COL, "반품수량"]
    if not all(c in agg_df.columns for c in cols):
        return
    out = agg_df[cols].copy()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_DIR / f"{center_label}_returns.parquet", index=False)
    print(f"  반품수량 별도 저장 -> {OUT_DIR}/{center_label}_returns.parquet ({len(out):,}행)")


def reindex_full_calendar(agg_df: pd.DataFrame, first_stock_map: pd.DataFrame,
                           global_end: pd.Timestamp) -> pd.DataFrame:
    fs = first_stock_map.copy()
    if len(fs):
        fs[SKU_COL] = (
            fs[PURCHASE_KEY_COLS[1]].astype(str) + SKU_SEP +
            fs[PURCHASE_KEY_COLS[2]].astype(str) + SKU_SEP +
            fs[PURCHASE_KEY_COLS[3]].astype(str)
        )
        fs = fs.rename(columns={PURCHASE_KEY_COLS[0]: CENTER_COL})
        fs = fs[[CENTER_COL, SKU_COL, PURCHASE_FIRST_STOCK_COL]]

    static_cols = [c for c in SKU_STATIC_COLS if c in agg_df.columns]
    sku_meta = agg_df.groupby([CENTER_COL, SKU_COL], as_index=False)[static_cols].first() if static_cols else \
        agg_df[[CENTER_COL, SKU_COL]].drop_duplicates()

    first_observed = agg_df.groupby([CENTER_COL, SKU_COL])[WEEK_COL].min().rename("first_observed_week")
    sku_ranges = first_observed.reset_index()
    if len(fs):
        sku_ranges = sku_ranges.merge(fs, on=[CENTER_COL, SKU_COL], how="left")
        sku_ranges["grid_start"] = sku_ranges[[PURCHASE_FIRST_STOCK_COL, "first_observed_week"]].min(axis=1)
        sku_ranges["grid_start"] = sku_ranges["grid_start"].fillna(sku_ranges["first_observed_week"])
    else:
        sku_ranges["grid_start"] = sku_ranges["first_observed_week"]

    data_floor = agg_df[WEEK_COL].min()
    sku_ranges["grid_start"] = sku_ranges["grid_start"].clip(lower=data_floor)
    sku_ranges["grid_start"] = sku_ranges["grid_start"] - pd.to_timedelta(
        sku_ranges["grid_start"].dt.weekday, unit="D"
    )

    sku_ranges["stock_week"] = sku_ranges["grid_start"]

    frames = []
    for _, row in sku_ranges.iterrows():
        weeks = pd.date_range(row["grid_start"], global_end, freq="7D")
        frames.append(pd.DataFrame({
            CENTER_COL: row[CENTER_COL], SKU_COL: row[SKU_COL], WEEK_COL: weeks,
        }))
    grid = pd.concat(frames, ignore_index=True)

    out = grid.merge(
        agg_df[[CENTER_COL, SKU_COL, WEEK_COL, QTY_COL]],
        on=[CENTER_COL, SKU_COL, WEEK_COL], how="left",
    )
    out = out.merge(sku_ranges[[CENTER_COL, SKU_COL, "stock_week"]], on=[CENTER_COL, SKU_COL], how="left")

    missing = out[QTY_COL].isna()
    after_stock = out[WEEK_COL] >= out["stock_week"]
    out.loc[missing & after_stock, QTY_COL] = 0

    out = out.drop(columns=["stock_week"]).merge(sku_meta, on=[CENTER_COL, SKU_COL], how="left")
    return out


def add_base_targets(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sold_flag"] = np.where(df[QTY_COL].isna(), np.nan, (df[QTY_COL] > 0).astype(float))
    df["qty_log1p"] = np.where(df["sold_flag"] == 1, np.log1p(df[QTY_COL]), np.nan)
    return df


def add_sku_last_active_week(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values([SKU_COL, WEEK_COL]).reset_index(drop=True)
    sale_week = df[WEEK_COL].where(df["sold_flag"] == 1)
    df["sku_last_active_week"] = sale_week.groupby(df[SKU_COL]).ffill()
    return df


def add_horizon_targets(df: pd.DataFrame, horizons: list[int], global_end: pd.Timestamp) -> pd.DataFrame:
    df = df.copy()
    grouped = df.groupby(SKU_COL, sort=False)
    future_cutoff = global_end
    for h in horizons:
        shifted_qty = grouped[QTY_COL].shift(-h)
        shifted_flag = grouped["sold_flag"].shift(-h)
        shifted_log1p = grouped["qty_log1p"].shift(-h)

        beyond_end = (df[WEEK_COL] + pd.Timedelta(weeks=h)) > future_cutoff
        df[f"target_h{h}"] = np.where(beyond_end, np.nan, shifted_qty)
        df[f"target_h{h}_flag"] = np.where(beyond_end, np.nan, shifted_flag)
        df[f"target_h{h}_qty_log1p"] = np.where(beyond_end, np.nan, shifted_log1p)
    return df


def report_split(name: str, df: pd.DataFrame):
    if len(df) == 0:
        print(f"  [{name}] 0행")
        return
    print(f"  [{name}] {len(df):,}행 | {df[WEEK_COL].min().date()} ~ {df[WEEK_COL].max().date()}")


def compute_pre_regime_sold_skus(raw_full: pd.DataFrame) -> set:
    if RAW_REGIME_COL not in raw_full.columns:
        return set()
    pre = raw_full[(raw_full[RAW_REGIME_COL] != "post") & (raw_full[RAW_QTY_COL] > 0)]
    if len(pre) == 0:
        return set()
    sku_keys = (
        pre[RAW_SKU_PARTS[0]].astype(str) + SKU_SEP +
        pre[RAW_SKU_PARTS[1]].astype(str) + SKU_SEP +
        pre[RAW_SKU_PARTS[2]].astype(str)
    )
    return set(sku_keys.unique())


def build_center_df(path: Path, horizons: list[int], first_stock_map: pd.DataFrame,
                     center_label: str, regime_filter: str | None = None) -> pd.DataFrame:
    raw = load_raw(path)

    pre_regime_skus = compute_pre_regime_sold_skus(raw) if regime_filter is not None else set()

    if regime_filter is not None and RAW_REGIME_COL in raw.columns:
        before = len(raw)
        raw = raw[raw[RAW_REGIME_COL] == regime_filter].copy()
        print(f"  레짐 필터({regime_filter}만): {before:,}행 -> {len(raw):,}행")

    calendar = build_calendar_features(raw)
    global_end = raw[RAW_WEEK_COL].max()

    agg = aggregate_region_to_sku(raw)
    print(f"  지역 합산: {len(raw):,}행 -> {len(agg):,}행")
    save_returns_table(center_label, agg)

    reindexed = reindex_full_calendar(agg, first_stock_map, global_end)
    print(f"  reindex(연속 주간 그리드): {len(agg):,}행(관측) -> {len(reindexed):,}행(전체 그리드)")

    reindexed = reindexed.merge(calendar, on=WEEK_COL, how="left")

    # Weather Region Assignment 병합 — reindexed(다운스트림 grid, 레짐 필터가 이미
    # 반영된 상태)를 항상 driving table로 두는 LEFT JOIN. weather_sku_week_B는 레짐
    # 이전 구간까지 포함한 더 넓은 테이블이지만, 이 방향의 LEFT JOIN이라 reindexed에
    # 없는 레짐 이전 초과행은 자동으로 버려지고 결측도 발생하지 않는다(실측 검증 완료).
    sku_week_weather = load_sku_week_weather(center_label)
    reindexed = reindexed.merge(sku_week_weather, on=[CENTER_COL, SKU_COL, WEEK_COL], how="left")
    n_missing_weather = reindexed["평균온도"].isna().sum()
    print(f"  SKU-week 날씨 병합(Weather Region Assignment): 평균온도 결측 {n_missing_weather:,}건 "
          f"({'정상' if n_missing_weather == 0 else '⚠ 확인 필요'})")

    reindexed["existed_before_regime"] = reindexed[SKU_COL].isin(pre_regime_skus)
    if pre_regime_skus:
        n_veteran = reindexed.loc[reindexed["existed_before_regime"], SKU_COL].nunique()
        print(f"  레짐 이전부터 판매 이력 있는 SKU: {n_veteran:,}개 (existed_before_regime=True)")

    df = add_base_targets(reindexed)
    df = add_sku_last_active_week(df)
    df = add_horizon_targets(df, horizons, global_end)
    df = df.sort_values([SKU_COL, WEEK_COL]).reset_index(drop=True)
    return df


def process_center_a(first_stock_map: pd.DataFrame):
    print("=== 센터 A ===")
    df = build_center_df(FILE_A, A_HORIZONS, first_stock_map, center_label="A", regime_filter=None)

    train = df[(df[WEEK_COL] >= A_TRAIN_START) & (df[WEEK_COL] <= A_TRAIN_END)]
    val = df[(df[WEEK_COL] >= A_VAL_START) & (df[WEEK_COL] <= A_VAL_END)]
    test = df[(df[WEEK_COL] >= A_TEST_START) & (df[WEEK_COL] <= A_TEST_END)]

    report_split("train", train)
    report_split("val", val)
    report_split("test", test)
    print(f"  스키마({len(df.columns)}개 컬럼): {list(df.columns)}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(OUT_DIR / "A_train.parquet", index=False)
    val.to_parquet(OUT_DIR / "A_val.parquet", index=False)
    test.to_parquet(OUT_DIR / "A_test.parquet", index=False)
    print(f"  저장 완료 -> {OUT_DIR}/A_{{train,val,test}}.parquet")


def generate_b_walkforward_folds(pool_start: str, pool_end: str, train_start: str,
                                  val_weeks: int, step_weeks: int) -> list[tuple]:
    pool_start_ts, pool_end_ts = pd.Timestamp(pool_start), pd.Timestamp(pool_end)
    folds = []
    cursor = pool_start_ts
    while cursor <= pool_end_ts:
        val_start = cursor
        val_end = min(val_start + pd.Timedelta(weeks=val_weeks) - pd.Timedelta(days=1), pool_end_ts)
        train_end = val_start - pd.Timedelta(days=1)
        folds.append((
            train_start, train_end.strftime("%Y-%m-%d"),
            val_start.strftime("%Y-%m-%d"), val_end.strftime("%Y-%m-%d"),
        ))
        cursor += pd.Timedelta(weeks=step_weeks)
    return folds


def process_center_b(first_stock_map: pd.DataFrame):
    print("=== 센터 B ===")
    df = build_center_df(FILE_B, B_HORIZONS, first_stock_map, center_label="B", regime_filter="post")

    train = df[(df[WEEK_COL] >= B_TRAIN_START) & (df[WEEK_COL] <= B_TRAIN_END)]
    pool_2024 = df[(df[WEEK_COL] >= B_WF_POOL_START) & (df[WEEK_COL] <= B_WF_POOL_END)]

    report_split("train (2023-07~12)", train)
    report_split("2024 walk-forward pool", pool_2024)
    print(f"  스키마({len(df.columns)}개 컬럼): {list(df.columns)}")

    folds = generate_b_walkforward_folds(B_WF_POOL_START, B_WF_POOL_END, B_TRAIN_START,
                                          B_WF_VAL_WEEKS, B_WF_STEP_WEEKS)
    print(f"  walk-forward fold {len(folds)}개 생성 (val_weeks={B_WF_VAL_WEEKS}, step_weeks={B_WF_STEP_WEEKS})")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    train.to_parquet(OUT_DIR / "B_train.parquet", index=False)
    pool_2024.to_parquet(OUT_DIR / "B_2024_pool.parquet", index=False)
    with open(OUT_DIR / "B_walkforward_folds.json", "w", encoding="utf-8") as f:
        json.dump(folds, f, ensure_ascii=False, indent=2)
    print(f"  저장 완료 -> {OUT_DIR}/B_train.parquet, B_2024_pool.parquet, B_walkforward_folds.json")


if __name__ == "__main__":
    fs_map = load_first_stock_map()
    process_center_a(fs_map)
    print()
    process_center_b(fs_map)