"""
weather_region_assignment.py

SKU-week 날씨 집계를 위한 Weather Region Assignment 재구현.

[배경 — 기존 파이프라인의 문제]
기존 aggregate_demand_data.py -> split_train_val_test.py 경로는 두 가지 구조적 문제가
있음이 확인됨(대화 세션 내 사전 조사):
  1) 총강수량이 거래 라인 수만큼 중복 합산됨(주간 group_cols에 거래일이 없어서
     같은 날 여러 거래 라인이 있으면 그날 강수량이 여러 번 더해짐 — 최대 99배,
     전체 평균 1.46배 과대평가 확인).
  2) split_train_val_test.py의 build_calendar_features()가 SKU/지역과 무관하게
     "센터 전체 week 단위 평균"을 모든 SKU에 그대로 broadcast — SKU가 실제로
     어느 지역에서 팔렸는지가 전혀 반영되지 않음.

[새 설계 — Weather Region Assignment]
1. SKU-week에 판매(총판매수량>0)가 있으면 그 SKU의 당주 실제 판매지역 집합 사용
2. SKU-week 판매가 0이면 해당 센터의 당주 전체 판매지역 집합 사용
3. 센터 전체도 당주 판매가 없으면, 센터의 가장 최근 과거 non-empty week
   판매지역 집합을 causal fallback으로 사용 (미래 데이터 참조 금지 — ffill만 사용)
4. 과거에도 이용 가능한 지역정보가 전혀 없는 경우에만 weather를 missing 처리
5. 지역 집합이 정해지면 거래빈도 가중 없이 지역별 날짜별 날씨를 equal-weight
   평균, 월~일 7일 기준으로 주간 집계
   - 지역별 주간값(평균온도=7일 mean, 총강수량=7일 sum)을 먼저 구하고,
     그 다음 지역 간에는 equal-weight 평균(총강수량도 지역 간엔 평균)으로 집계.

[날씨 원본]
거래 테이블과 조인되기 전, (시도,시군구,년,월,일) 기준 unique가 확인된
data/external/weather/weather_by_region_clean.csv를 사용한다(사전 조사에서 중복 0건
확인됨). 거래-조인 이후 산출물(aggregate_demand_data.py 등)은 총강수량 중복합산
버그가 있으므로 날씨 값 자체는 재사용하지 않는다.

[지역 매칭 3단계 — master_join.sql과 동일 로직 재구현]
거래 데이터의 (시도,시군구)와 날씨 원본의 지역 해상도가 다름(날씨 원본이 더 거칢).
사전 조사에서 단순 exact match만 쓰면 A 37.5%/B 48%의 지역이 누락됨을 확인했고,
master_join.sql의 3단계 매칭(①정확 ②광역시 브로드캐스트 ③시(구) 상위 접두어)을
그대로 시뮬레이션한 결과 A/B 모두 매칭 실패 0건임을 확인함 -> 동일 로직을 그대로
재구현한다.

[입고 전 처리]
NaN vs 0 판정은 기존 stock_week(=grid_start, 최초입고일 월요일 스냅) 기준을 그대로
사용(N개월 유예 등 추가 로직 없음 — 과거 결정 기록을 찾지 못해 도입하지 않기로 확정).
grid_start 이전 주는 이 스크립트의 SKU-week 그리드 자체에 행을 생성하지 않는다
(=날씨 집계 대상에서 제외). 따라서 최종 테이블에 "행이 아예 없음"=입고 전(콜드스타트
이전), "행은 있는데 is_missing=True"=규칙4(과거 지역정보 전무)로 사후 구분 가능하다.

[주 경계 커버리지]
날씨 원본은 2021-01-01~2024-12-31만 존재. 주의 7일 중 이 범위를 벗어나는 날짜는
집계에서 제외하고, 실제 사용된 날짜 수를 weather_day_count 컬럼에 남긴다
(다운스트림에서 필터링 가능하도록).
"""

from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
FINAL_DIR = DATA_DIR / "final"

WEATHER_CSV = DATA_DIR / "external" / "weather" / "weather_by_region_clean.csv"

GRAIN_FILES = {
    "A": FINAL_DIR / "master_demand_weekly_A.parquet",
    "B": FINAL_DIR / "master_demand_weekly_B_with_regime.parquet",
}
PURCHASE_FILE = FINAL_DIR / "cleaned_purchase_cleaned_for_pred.parquet"

OUT_FILES = {
    "A": FINAL_DIR / "weather_sku_week_A.parquet",
    "B": FINAL_DIR / "weather_sku_week_B.parquet",
}

SKU_KEY = ["센터", "바코드", "옵션코드", "상품클러스터"]
PURCHASE_KEY = ["센터", "바코드", "옵션코드", "상품클러스터"]
WEEK_COL = "주시작일"
REGION_COLS = ["시도", "시군구"]

METRO_SIDO = {"서울특별시", "부산광역시", "대구광역시", "울산광역시"}


# ---------------------------------------------------------------------------
# 1) 날씨 원본 로드
# ---------------------------------------------------------------------------
def load_weather_source() -> pd.DataFrame:
    df = pd.read_csv(WEATHER_CSV, encoding="cp949")
    df["date"] = pd.to_datetime(
        df["년"].astype(str) + "-" + df["월"].astype(str) + "-" + df["일"].astype(str)
    )
    n = len(df)
    n_distinct = df.drop_duplicates(subset=["시도", "시군구", "년", "월", "일"]).shape[0]
    if n != n_distinct:
        raise RuntimeError(
            f"날씨 원본이 (시도,시군구,년,월,일) 기준 unique하지 않음: "
            f"{n:,}행 vs distinct {n_distinct:,}건 — 사전 조사 전제가 깨졌으므로 중단"
        )
    print(f"[날씨 원본] {n:,}행, unique 확인됨, 기간 {df['date'].min().date()} ~ {df['date'].max().date()}")
    return df


# ---------------------------------------------------------------------------
# 2) 거래지역 -> 날씨지역 3단계 매칭 (master_join.sql 로직 재구현)
# ---------------------------------------------------------------------------
def build_region_resolution_map(weather_df: pd.DataFrame, txn_regions: pd.DataFrame) -> pd.DataFrame:
    """txn_regions: 거래 데이터에 등장하는 고유 (시도,시군구) DataFrame.
    반환: (시도,시군구,w_시도,w_시군구,match_type) — w_시군구는 광역시 브로드캐스트일 때 NaN.
    """
    w_pairs_exact = set(map(tuple, weather_df[["시도", "시군구"]].dropna().drop_duplicates().values))
    w_sigungu_by_sido = (
        weather_df.dropna(subset=["시군구"]).groupby("시도")["시군구"].apply(lambda s: sorted(set(s))).to_dict()
    )

    rows = []
    unmatched = []
    for sido, sigungu in txn_regions.itertuples(index=False):
        if (sido, sigungu) in w_pairs_exact:
            rows.append((sido, sigungu, sido, sigungu, "exact"))
            continue
        if sido in METRO_SIDO:
            rows.append((sido, sigungu, sido, np.nan, "metro_broadcast"))
            continue
        matched = False
        for w_sgg in w_sigungu_by_sido.get(sido, []):
            if isinstance(sigungu, str) and sigungu.startswith(w_sgg + " "):
                rows.append((sido, sigungu, sido, w_sgg, "city_prefix"))
                matched = True
                break
        if not matched:
            unmatched.append((sido, sigungu))
            rows.append((sido, sigungu, np.nan, np.nan, "NO_MATCH"))

    resolution = pd.DataFrame(rows, columns=["시도", "시군구", "w_시도", "w_시군구", "match_type"])
    print(f"[지역 매칭] 거래 데이터 고유 (시도,시군구) {len(txn_regions):,}건 중:")
    print(resolution["match_type"].value_counts().to_string())
    if unmatched:
        print(f"  [!] 매칭 실패 {len(unmatched)}건: {unmatched}")
    else:
        print("  매칭 실패 0건")
    return resolution


# ---------------------------------------------------------------------------
# 3) (날씨지역, 주) 단위 사전 집계 — SKU와 무관, 한 번만 계산
# ---------------------------------------------------------------------------
def build_region_week_weather(weather_df: pd.DataFrame) -> pd.DataFrame:
    df = weather_df.copy()
    df[WEEK_COL] = df["date"] - pd.to_timedelta(df["date"].dt.weekday, unit="D")
    # 시군구 NaN(광역시 대표값) 그룹도 groupby에서 유지되도록 fillna 후 groupby, 이후 복원
    df["_sgg_key"] = df["시군구"].fillna("__NULL__")
    agg = df.groupby(["시도", "_sgg_key", WEEK_COL], as_index=False).agg(
        평균온도=("평균온도", "mean"),
        총강수량=("총강수량", "sum"),
        weather_day_count=("date", "count"),
    )
    agg["시군구"] = agg["_sgg_key"].replace("__NULL__", np.nan)
    agg = agg.drop(columns="_sgg_key")
    print(f"[지역-주 날씨 사전집계] {len(agg):,}행 (distinct 날씨지역 {agg[['시도','시군구']].drop_duplicates().shape[0]}개 x 주)")
    max_count = agg["weather_day_count"].max()
    partial = (agg["weather_day_count"] < 7).sum()
    print(f"  7일 미만 커버리지(주 경계 등) 행 수: {partial:,} / {len(agg):,} (최대 count={max_count})")
    return agg


# ---------------------------------------------------------------------------
# 4) 판매지역 결정용 grain 로드
# ---------------------------------------------------------------------------
def load_region_sales_grain(center: str) -> pd.DataFrame:
    df = pd.read_parquet(
        GRAIN_FILES[center],
        columns=[WEEK_COL, "센터", "시도", "시군구", "바코드", "옵션코드", "상품클러스터", "총판매수량"],
    )
    return df


def build_sku_actual_regions(grain: pd.DataFrame) -> pd.DataFrame:
    """규칙1: SKU-week 판매(총판매수량>0)가 있으면 그 SKU의 당주 실제 판매지역."""
    sold = grain[grain["총판매수량"] > 0]
    grp = sold.groupby(SKU_KEY + [WEEK_COL])
    out = grp.apply(lambda g: frozenset(zip(g["시도"], g["시군구"])), include_groups=False)
    out = out.rename("actual_regions").reset_index()
    return out


def build_center_week_regions(grain: pd.DataFrame) -> pd.DataFrame:
    """센터-week 전체 판매지역(총판매수량>0 기준) — 규칙2/3/4의 기반."""
    sold = grain[grain["총판매수량"] > 0]
    grp = sold.groupby(["센터", WEEK_COL])
    out = grp.apply(lambda g: frozenset(zip(g["시도"], g["시군구"])), include_groups=False)
    out = out.rename("region_set").reset_index()
    return out


def build_causal_fallback_index(center_week_regions: pd.DataFrame, center: str,
                                 data_floor: pd.Timestamp, global_end: pd.Timestamp) -> pd.DataFrame:
    """센터별 주 단위 dense 인덱스를 만들고, region_set을 causal(과거만) ffill.

    [의도적 결정 — 레짐 경계 제한을 두지 않음]
    B센터는 '레짐'(반품률 성격 변화 + SKU 구성 turnover + 판매지역 커버리지 자체도
    부분적으로 변경 — 실측: 레짐전만의 지역 6개, 레짐후만의 지역 7개 확인됨)이 있어,
    이 함수가 만드는 인덱스는 data_floor(=B 전체 이력의 시작, 레짐 이전 포함)부터
    시작하므로 causal fallback이 레짐 경계를 넘어 과거(레짐 이전) 판매지역을 끌어올
    수 있는 구조다. 개념적으로는 레짐별로 fallback 탐색 범위를 분리하는 것이 더
    안전하지만, 아래 이유로 이번엔 적용하지 않음:
      (a) 규칙3(causal fallback) 자체가 A/B 전체 이력 기준으로 0건 발동 확인됨
          (센터-week 전체 무판매 gap이 관측 기간 내 한 번도 없었음)
      (b) 이 프로젝트는 현재 보유한 과거 데이터로만 진행하고 향후 신규 데이터 유입이
          없어, "향후 발동 가능성" 리스크가 실질적으로 없음
      (c) 레짐 경계 자체가 별도 changepoint 분석으로 확정된 값이라 이 함수 안에서
          다시 재해석할 필요가 없음
    향후 이 파이프라인을 새 데이터로 재실행하게 되면 (a)(b)의 전제가 깨지므로, 그때는
    레짐 경계 제한 적용 여부를 다시 검토할 것.
    """
    weeks = pd.date_range(data_floor, global_end, freq="7D")
    idx = pd.DataFrame({WEEK_COL: weeks})
    cw = center_week_regions[center_week_regions["센터"] == center][[WEEK_COL, "region_set"]]
    idx = idx.merge(cw, on=WEEK_COL, how="left")
    idx = idx.sort_values(WEEK_COL).reset_index(drop=True)

    idx["source_week"] = idx[WEEK_COL].where(idx["region_set"].notna())
    idx["region_set"] = idx["region_set"].ffill()
    idx["source_week"] = idx["source_week"].ffill()

    idx["weeks_back"] = ((idx[WEEK_COL] - idx["source_week"]).dt.days // 7)
    idx["rule"] = np.where(idx["region_set"].isna(), 4, np.where(idx["weeks_back"] == 0, 2, 3))
    return idx[[WEEK_COL, "region_set", "weeks_back", "rule"]]


# ---------------------------------------------------------------------------
# 5) 입고일 기준 grid_start/stock_week (split_train_val_test.py 로직 재구현)
# ---------------------------------------------------------------------------
def load_first_stock_map() -> pd.DataFrame:
    pdf = pd.read_parquet(PURCHASE_FILE, columns=PURCHASE_KEY + ["최초입고일"])
    pdf["최초입고일"] = pd.to_datetime(pdf["최초입고일"])
    return pdf.groupby(PURCHASE_KEY, as_index=False)["최초입고일"].min()


def build_sku_grid_ranges(grain: pd.DataFrame, first_stock_map: pd.DataFrame, center: str) -> pd.DataFrame:
    sku_meta = grain[SKU_KEY].drop_duplicates()
    first_observed = grain.groupby(SKU_KEY, as_index=False)[WEEK_COL].min().rename(
        columns={WEEK_COL: "first_observed_week"}
    )
    ranges = sku_meta.merge(first_observed, on=SKU_KEY, how="left")
    ranges = ranges.merge(first_stock_map, on=PURCHASE_KEY, how="left")

    ranges["grid_start"] = ranges[["최초입고일", "first_observed_week"]].min(axis=1)
    ranges["grid_start"] = ranges["grid_start"].fillna(ranges["first_observed_week"])

    data_floor = grain[WEEK_COL].min()
    global_end = grain[WEEK_COL].max()
    ranges["grid_start"] = ranges["grid_start"].clip(lower=data_floor)
    ranges["grid_start"] = ranges["grid_start"] - pd.to_timedelta(ranges["grid_start"].dt.weekday, unit="D")
    ranges["stock_week"] = ranges["grid_start"]

    n_no_purchase = ranges["최초입고일"].isna().sum()
    print(f"[{center}] SKU {len(ranges):,}개, 최초입고일 매핑 실패(첫관측주로 대체) {n_no_purchase:,}개")
    print(f"[{center}] data_floor={data_floor.date()}, global_end={global_end.date()}")
    return ranges, data_floor, global_end


def expand_grid(ranges: pd.DataFrame, global_end: pd.Timestamp) -> pd.DataFrame:
    """SKU별 stock_week ~ global_end 를 7일 간격으로 벡터화 전개."""
    n_weeks = ((global_end - ranges["stock_week"]).dt.days // 7 + 1).clip(lower=0).astype(int)
    total = int(n_weeks.sum())
    print(f"  그리드 총 행수: {total:,}")

    rep_idx = np.repeat(np.arange(len(ranges)), n_weeks.values)
    offsets = np.concatenate([np.arange(n) for n in n_weeks.values]) if total > 0 else np.array([], dtype=int)

    grid = ranges.iloc[rep_idx][SKU_KEY + ["stock_week"]].reset_index(drop=True)
    grid[WEEK_COL] = grid["stock_week"] + pd.to_timedelta(offsets * 7, unit="D")
    grid = grid.drop(columns="stock_week")
    return grid


# ---------------------------------------------------------------------------
# 6) 지역집합 -> 날씨값 (equal-weight, 벡터화 — merge/groupby 기반)
#
# 성능 참고: 행 단위 grid.apply(python dict lookup)는 센터당 220만행 규모에서
# 처리 시간이 너무 길어 explode+merge+groupby 기반 벡터 연산으로 구현함.
#
# 가중 방식: "지역별 equal-weight 평균"의 "지역"은 날씨 관측소(=매칭된 날씨지역
# w_시도/w_시군구) 기준 distinct 값으로 해석한다 — 예: 포항시북구/포항시남구처럼
# 같은 날씨지역으로 resolve되는 여러 판매지역이 있어도 그 날씨지역은 1표만
# 갖는다. 그렇지 않으면(원본 판매지역 그대로 평균) 행정구역이 잘게 나뉜 지역이
# 과대 반영되는, 기존 파이프라인에서 지적된 것과 같은 종류의 왜곡(문제4:
# "SKU 다양성이 큰 지역 쏠림")이 날씨지역 단위로 재발할 수 있어 이를 피하기
# 위한 선택.
# ---------------------------------------------------------------------------
def resolve_regions_to_weather_vectorized(grid: pd.DataFrame, resolution_map: pd.DataFrame,
                                           region_week_weather: pd.DataFrame) -> pd.DataFrame:
    work = grid[["region_set_final", WEEK_COL]].copy()
    work["row_id"] = np.arange(len(work))

    exploded = work.explode("region_set_final")
    valid = exploded["region_set_final"].notna()
    exploded_valid = exploded[valid].copy()

    tuples = exploded_valid["region_set_final"].tolist()
    exploded_valid["시도"] = [t[0] for t in tuples]
    exploded_valid["시군구"] = [t[1] for t in tuples]
    exploded_valid = exploded_valid.drop(columns="region_set_final")

    merged = exploded_valid.merge(
        resolution_map[["시도", "시군구", "w_시도", "w_시군구"]], on=["시도", "시군구"], how="left"
    )
    # 날씨지역(w_시도,w_시군구) 기준 distinct — 같은 관측소로 매칭되는 여러 판매지역 중복 제거
    merged["w_시군구_key"] = merged["w_시군구"].fillna("__NULL__")
    merged = merged.drop_duplicates(subset=["row_id", "w_시도", "w_시군구_key"])

    weather_lookup = region_week_weather.copy()
    weather_lookup["시군구_key"] = weather_lookup["시군구"].fillna("__NULL__")
    weather_lookup = weather_lookup.rename(columns={"시도": "w_시도", "시군구_key": "w_시군구_key"})

    merged = merged.merge(
        weather_lookup[["w_시도", "w_시군구_key", WEEK_COL, "평균온도", "총강수량", "weather_day_count"]],
        on=["w_시도", "w_시군구_key", WEEK_COL],
        how="left",
    )

    agg = merged.groupby("row_id", as_index=False).agg(
        평균온도=("평균온도", "mean"),
        총강수량=("총강수량", "mean"),
        weather_day_count=("weather_day_count", "min"),
        n_regions=("w_시도", "count"),
    )

    row_index = pd.DataFrame({"row_id": np.arange(len(grid))})
    result = row_index.merge(agg, on="row_id", how="left")
    result["n_regions"] = result["n_regions"].fillna(0).astype(int)
    return result[["평균온도", "총강수량", "weather_day_count", "n_regions"]]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def process_center(center: str, weather_df: pd.DataFrame, region_week_weather: pd.DataFrame,
                    resolution_map: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{'='*70}\n[센터 {center}] 처리 시작\n{'='*70}")
    grain = load_region_sales_grain(center)
    grain["센터"] = center

    first_stock_map = load_first_stock_map()
    ranges, data_floor, global_end = build_sku_grid_ranges(grain, first_stock_map, center)

    print(f"[{center}] 그리드 전개 중...")
    grid = expand_grid(ranges, global_end)

    sku_actual = build_sku_actual_regions(grain)
    center_week_regions = build_center_week_regions(grain)
    fallback_idx = build_causal_fallback_index(center_week_regions, center, data_floor, global_end)

    print(f"[{center}] 규칙1(실제 판매지역) 병합...")
    grid = grid.merge(sku_actual, on=SKU_KEY + [WEEK_COL], how="left")

    print(f"[{center}] 규칙2/3/4(fallback) 병합...")
    grid = grid.merge(fallback_idx, on=WEEK_COL, how="left")

    has_actual = grid["actual_regions"].notna()
    grid["region_rule"] = np.where(has_actual, 1, grid["rule"])
    grid["region_set_final"] = np.where(has_actual, grid["actual_regions"], grid["region_set"])
    grid["fallback_weeks_back"] = np.where(has_actual, 0, grid["weeks_back"].fillna(0)).astype(int)
    grid["is_missing"] = grid["region_rule"] == 4

    print(f"[{center}] region_rule 분포:")
    print(grid["region_rule"].value_counts().sort_index().to_string())
    print(f"[{center}] causal fallback(규칙3) 발동: {(grid['region_rule']==3).sum():,}건")
    print(f"[{center}] missing(규칙4) 발동: {(grid['region_rule']==4).sum():,}건")

    print(f"[{center}] 날씨 값 계산 중 (행 {len(grid):,}개, 벡터화)...")
    results = resolve_regions_to_weather_vectorized(grid, resolution_map, region_week_weather)
    grid = pd.concat([grid.reset_index(drop=True), results.reset_index(drop=True)], axis=1)

    grid.loc[grid["is_missing"], ["평균온도", "총강수량", "weather_day_count"]] = np.nan
    grid.loc[grid["is_missing"], "n_regions"] = 0

    out_cols = SKU_KEY[1:] + ["센터", WEEK_COL, "평균온도", "총강수량", "weather_day_count",
                              "region_rule", "n_regions", "fallback_weeks_back", "is_missing"]
    final = grid[["센터"] + [c for c in out_cols if c != "센터"]]
    return final


def main():
    weather_df = load_weather_source()
    region_week_weather = build_region_week_weather(weather_df)

    all_txn_regions = []
    for center in ["A", "B"]:
        g = load_region_sales_grain(center)
        all_txn_regions.append(g[["시도", "시군구"]].drop_duplicates())
    txn_regions = pd.concat(all_txn_regions, ignore_index=True).drop_duplicates()
    resolution_map = build_region_resolution_map(weather_df, txn_regions)

    results = {}
    for center in ["A", "B"]:
        final = process_center(center, weather_df, region_week_weather, resolution_map)
        out_path = OUT_FILES[center]
        final.to_parquet(out_path, index=False)
        print(f"[{center}] 저장 완료 -> {out_path} ({len(final):,}행)")
        results[center] = final

    return results, region_week_weather, resolution_map


if __name__ == "__main__":
    main()
