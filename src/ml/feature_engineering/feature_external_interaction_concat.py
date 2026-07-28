"""
feature_external_interaction_concat.py
Day3 - 외부변수 Feature + 상호작용(Interaction) Feature 생성 + A/B 통합 Final Feature Table

Day2 산출물(data/ml/splits/*_feat.parquet, 5개: A_train/A_val/A_test/B_train/B_pool)을
읽어 하나로 합친 뒤, 아래를 추가하고 단일 parquet(feature_table_final.parquet)으로 저장한다.

핵심 설계:
    1) 공휴일_W0/W-1/W+1: 예전엔 (center_id, week_st) 단위로 거래 데이터에 조인된 원본
       `공휴일`(설날/추석 여부, 해당 날짜에 실제 매출/출고 거래가 있어야만 값이 매겨짐)을
       그대로 shift해서 D0/D-1/D+1을 만들었는데, **B센터는 명절 당일 전후로 출고 거래
       자체를 거의 기록하지 않아 B 전체(train+pool 60만 행)에서 이 컬럼이 단 한 번도
       1인 적이 없는 구조적 결함이 있었음**(실측 확인 — sql/join/master_join.sql의
       공휴일 조인은 (년,월,일)만으로 매칭해 center 의존성이 아예 없고 A/B 동일 로직인데도,
       B는 매칭할 거래 행 자체가 없어서 발생한 문제. `B센터_명절휴무`라는 B 전용 하드코딩
       패치가 있었던 이유이기도 함). 그래서 **거래 유무와 완전히 무관한 순수 캘린더 기준**
       (KOREAN_LUNAR_HOLIDAYS, 공식 설날/추석 연휴+대체공휴일/임시공휴일 날짜를 A(2021~)/
       B(2023~) 관측 범위 전체에 대해 하드코딩)으로 재설계해 A/B 공통 단일 변수 3개로
       통합했다. W0=이번 주가 명절 포함 주, W-1=다음 주가 명절 포함 주(LEAD),
       W+1=지난 주가 명절 포함 주(LAG) — 팀 확정 정의(숫자 부호와 시제가 반대인 비통상적
       관례이니 주의). 거래-조인 기반 원본 `공휴일`/`공휴일_D0/D-1/D+1`/`B센터_명절휴무`는
       전부 제거하고 이 3개로 완전히 대체함.
    2) 강수량_호우_flag/count: 주 단위로 미리 집계된 총강수량에서 분위수를 구하면 주간
       평탄화로 특정 하루의 극단 이벤트가 상쇄되어 묻히므로, 반드시 일별(raw) 원본 기후
       데이터에서 먼저 하루 단위로 극단 여부를 판정한 뒤 주 단위로 집계(count/flag)한다.
       일별 원본은 별도 외부 기후 CSV를 다시 지역 매칭하지 않고, data/final/
       aggregated_daily_demand.parquet(거래일 단위 raw, 센터/시도/시군구별 평균온도·총강수량이
       이미 정확히 join되어 있는 소스)을 그대로 사용한다.
       (총강수량이 누적값이 아니라 그날 하루치인지 확인함 — 같은 지역 시계열에서 값이
       0으로 자주 리셋되고 전날보다 줄어드는 경우도 다수 확인돼 일일 합계로 판단)
       - (센터, 거래일, 시도, 시군구) 기준으로 drop_duplicates 후 센터·날짜별로 지역 평균을 내어
         센터 단위 일별 시계열 하나를 만든다.
       - 임계값은 분위수(percentile) 대신 절대 기준(기상청 특보 기준과 유사한 고정값) 80mm
         이상을 호우로 판정(상수 HEAVY_RAIN_THRESHOLD). A/B 공통 고정값.
       - 해당 주(week_st, 월~일)에 며칠 발생했는지 count하고 count>0이면 flag=1.
       - 원본 거래일 커버리지가 2024-12-31까지라 마지막 주(week_st=2024-12-30)는 2일치
         (12/30, 12/31)만 반영됨 — 알려진 데이터 한계, 별도 처리 없이 있는 날짜만으로 집계.
       - aggregated_daily_demand.parquet은 거래(판매) 발생 일자만 기록되어 있어, 그 센터
         관할 전 지역에서 하루 종일 판매가 0건이면 그 날짜 자체가 통째로 빠짐(진짜 캘린더
         365일 전체가 아님) — 극단 기후가 있었어도 그날 아무 지역도 판매가 없었다면
         count에서 누락될 수 있는 알려진 한계.
       - 기온_극단_flag/count(폭염 33도/한파 -12도)는 시도했으나 폐기함: 원본에 일 최고/최저
         기온이 없고 일 평균기온만 존재해, 특보 기준 절대값을 평균기온에 그대로 적용하면
         전체 데이터에서 양성 발생이 0~2건뿐인 영분산(zero-variance) dead feature가 됨을
         확인. 대신 평균온도/총강수량을 그대로 연속형 feature로 남겨(트리 모델이 스스로
         비선형 임계점을 학습하도록) temp_x_precip(=평균온도*총강수량) 상호작용만 추가.
    3) center_is_B / temp_x_precip: 통합 모델 1개가 센터별 차이를 반영하도록 하는 상호작용
       feature. `qty_lag1 * center_is_B` 형태의 명시적 곱셈 상호작용은 의도적으로 생성하지
       않음 — qty_lag1은 Base Model 핵심 공통 feature라 트리 모델(LightGBM)이 center 계열
       feature와 조합해 스스로 상호작용을 학습할 수 있고, 인위적 곱 feature가 B센터(26주
       학습 데이터로 상대적으로 적음)의 과적합 위험을 키울 수 있다고 판단했기 때문(Day4 ML
       성능 검증에서 B 오차가 크게 나오면 그때 추가 실험 과제로 재검토). 기온_극단_flag가
       폐기되면서 여기 의존하던 inter_center_temp_extreme도 함께 제거함 — 대신 평균온도*
       총강수량의 연속형 상호작용(temp_x_precip)을 추가.
    4) weeks_since_last_active_filled: Day2에서 만든 원본(datetime 파생, 미판매 행은 NaN)을
       SVM/신경망 등 NaN 미지원 모델용으로, 임의의 큰 값(999 등) 대신 관측 가능한
       최대 주수(156, 상수 MAX_WEEKS_SINCE_ACTIVE)로 capping하여 채움.
    5) get_excluded_cols()는 Day2(feature_lag_rolling_calendar.py)의 정의를 그대로 재사용.
       target_*, ID 컬럼, 진단용 _fill_source, sku_last_active_week(datetime 원본)을 제외.
       center_id는 ID_COLS에 포함되어 있으므로 원본 문자열은 여전히 제외되고,
       대신 숫자형 파생인 center_is_B가 feature로 흘러들어가는 구조를 그대로 유지한다.
"""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from feature_lag_rolling_calendar import get_excluded_cols

BASE_DIR = Path(__file__).resolve().parents[3]
SPLIT_DIR = BASE_DIR / "data" / "ml" / "splits"
OUT_PATH = SPLIT_DIR / "feature_table_final.parquet"

FINAL_DIR = BASE_DIR / "data" / "final"
DAILY_DEMAND_PATH = FINAL_DIR / "aggregated_daily_demand.parquet"
DAILY_DATE_COL = "거래일"

WEEK_COL = "week_st"
CENTER_COL = "center_id"

FEAT_FILES = {
    ("A", "train"): SPLIT_DIR / "A_train_feat.parquet",
    ("A", "val"): SPLIT_DIR / "A_val_feat.parquet",
    ("A", "test"): SPLIT_DIR / "A_test_feat.parquet",
    ("B", "train"): SPLIT_DIR / "B_train_feat.parquet",
    ("B", "pool"): SPLIT_DIR / "B_pool_feat.parquet",
}

MAX_WEEKS_SINCE_ACTIVE = 156

# 기상청 특보 기준과 유사한 절대 임계값 (percentile 아님, A/B 공통 고정값)
# 기온(폭염 33/한파 -12)은 일 평균기온 기준으로는 영분산 dead feature가 되어 폐기 — 아래 참고
HEAVY_RAIN_THRESHOLD = 80.0      # 이상이면 호우

# 공식 대한민국 설날/추석 연휴 + 대체공휴일/임시공휴일 날짜 전체 목록(A 2021~/B 2023~ 관측
# 범위를 모두 커버). 거래 유무와 무관한 순수 캘린더 기준이라 A/B 공통으로 그대로 쓴다.
KOREAN_LUNAR_HOLIDAYS = pd.to_datetime([
    "2021-02-11", "2021-02-12", "2021-02-13",  # 2021 설날
    "2021-09-20", "2021-09-21", "2021-09-22",  # 2021 추석
    "2022-01-31", "2022-02-01", "2022-02-02",  # 2022 설날
    "2022-09-09", "2022-09-10", "2022-09-11", "2022-09-12",  # 2022 추석(+대체공휴일 9/12)
    "2023-01-21", "2023-01-22", "2023-01-23", "2023-01-24",  # 2023 설날(+대체공휴일 1/24)
    "2023-09-28", "2023-09-29", "2023-09-30", "2023-10-02",  # 2023 추석(+임시공휴일 10/2)
    "2024-02-09", "2024-02-10", "2024-02-11", "2024-02-12",  # 2024 설날(+대체공휴일 2/12)
    "2024-09-16", "2024-09-17", "2024-09-18",  # 2024 추석
])


def load_all_feat() -> pd.DataFrame:
    frames = []
    for (center, split), path in FEAT_FILES.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{path} 없음 — Day2(feature_lag_rolling_calendar.py)를 먼저 실행해야 함"
            )
        df = pd.read_parquet(path)
        df["split"] = split
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def add_holiday_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """공휴일_W0/W-1/W+1: 거래 유무와 무관한 순수 캘린더(설날/추석) 기준 A/B 공통 단일 변수.
    옛 거래-조인 방식(공휴일_D0)은 B가 명절 당일 출고를 거의 기록하지 않아 B 전체가 항상
    0이 되는 구조적 결함이 있었음(모듈 docstring 1번 참고). KOREAN_LUNAR_HOLIDAYS의 각 날짜를
    그 날짜가 속한 주(week_st, 월요일)로 변환해 "명절이 포함된 주" 집합을 만들고, 이 집합에
    대한 소속 여부만으로 판정하므로 어떤 거래 기록도 필요 없다.
    W0=이번 주, W-1=다음 주(LEAD), W+1=지난 주(LAG) — 팀 확정 정의."""
    holiday_dates = pd.Series(KOREAN_LUNAR_HOLIDAYS)
    holiday_week_starts = holiday_dates - pd.to_timedelta(holiday_dates.dt.weekday, unit="D")
    holiday_weeks = set(holiday_week_starts)

    df["공휴일_W0"] = df[WEEK_COL].isin(holiday_weeks).astype(int)
    df["공휴일_W-1"] = (df[WEEK_COL] + pd.Timedelta(weeks=1)).isin(holiday_weeks).astype(int)
    df["공휴일_W+1"] = (df[WEEK_COL] - pd.Timedelta(weeks=1)).isin(holiday_weeks).astype(int)
    return df


DAILY_CENTER_COL = "센터"  # aggregated_daily_demand.parquet 원본 컬럼명 (Day1 이후로는 center_id로 통일됨)


def _load_daily_center_climate() -> pd.DataFrame:
    """aggregated_daily_demand.parquet(거래일 grain, 센터/시도/시군구별 평균온도·총강수량이
    이미 정확히 join되어 있는 raw 소스)에서 (센터, 거래일, 시도, 시군구) distinct 조합만 뽑고,
    센터·날짜별로 지역 평균을 내 센터 단위 일별 기후 시계열을 만든다."""
    raw = pd.read_parquet(
        DAILY_DEMAND_PATH, columns=[DAILY_DATE_COL, DAILY_CENTER_COL, "시도", "시군구", "평균온도", "총강수량"]
    )
    region_daily = raw.drop_duplicates(subset=[DAILY_CENTER_COL, DAILY_DATE_COL, "시도", "시군구"])
    daily = (
        region_daily.groupby([DAILY_CENTER_COL, DAILY_DATE_COL], as_index=False)[["평균온도", "총강수량"]]
        .mean()
        .rename(columns={DAILY_DATE_COL: "date", DAILY_CENTER_COL: CENTER_COL})
    )
    return daily


def add_climate_extreme_flags(df: pd.DataFrame) -> pd.DataFrame:
    """일별 원본 기후로 하루 단위 호우 여부를 먼저 판정한 뒤 주(week_st) 단위로
    발생 일수(count)/발생 여부(flag)를 집계해 붙인다. 임계값이 A/B 공통 고정값이라
    (percentile이 아니므로) 판정 자체는 센터 구분 없이 한 번에 처리하고,
    주 단위 집계만 센터별로 나눈다.
    기온 극단(폭염/한파) flag/count는 폐기함 — 일 평균기온 기준 절대 임계값(33/-12)을
    적용하면 전체 데이터에서 양성 발생이 0~2건뿐인 영분산 feature가 되기 때문
    (모듈 docstring 2번 항목 참고)."""
    daily = _load_daily_center_climate()

    daily["_rain_extreme_day"] = daily["총강수량"] >= HEAVY_RAIN_THRESHOLD

    daily[WEEK_COL] = daily["date"] - pd.to_timedelta(daily["date"].dt.weekday, unit="D")
    weekly = daily.groupby([CENTER_COL, WEEK_COL], as_index=False).agg(
        **{"강수량_호우_count": ("_rain_extreme_day", "sum")},
    )
    weekly["강수량_호우_flag"] = (weekly["강수량_호우_count"] > 0).astype(int)

    df = df.merge(weekly, on=[CENTER_COL, WEEK_COL], how="left")
    return df


def add_covid_flag(df: pd.DataFrame) -> pd.DataFrame:
    df["covid_flag"] = df["covid_영향여부"].astype(int)
    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """qty_lag1 * center_is_B 형태의 명시적 곱셈 상호작용은 의도적으로 생성하지 않음
    (모듈 docstring 4번 항목 참고 — 트리 모델 자체 학습에 맡기고, B 과적합 위험 회피).
    temp_x_precip: 평균온도*총강수량 연속형 상호작용 — 기온_극단_flag(영분산이라 폐기)를
    대체해 트리 모델이 스스로 비선형 임계점을 학습할 수 있도록 연속형 그대로 곱해서 제공."""
    df["center_is_B"] = (df[CENTER_COL] == "B").astype(int)
    df["temp_x_precip"] = df["평균온도"] * df["총강수량"]
    return df


def add_weeks_since_active_filled(df: pd.DataFrame) -> pd.DataFrame:
    df["weeks_since_last_active_filled"] = (
        df["weeks_since_last_active"].fillna(MAX_WEEKS_SINCE_ACTIVE).clip(upper=MAX_WEEKS_SINCE_ACTIVE)
    )
    return df


# 2단계 모델링(1단계 A+B 통합 Base Model + 2단계 A센터 잔차 보정) 확정에 따라,
# B_LAG_WEEKS=[1] 설계상 B센터에는 애초에 존재하지 않는(concat 시 B 전체가 NaN인)
# qty_lag2/4 계열은 Base Model 학습 feature에서 제외한다. _fill_source는 get_excluded_cols()가
# 이미 접미사로 걸러내므로 여기서는 원본/log1p/filled 3개씩만 추가하면 됨.
BASE_MODEL_EXCLUDED_EXTRA = [
    "qty_lag2", "qty_lag2_log1p", "qty_lag2_filled",
    "qty_lag4", "qty_lag4_log1p", "qty_lag4_filled",
]


def get_base_model_excluded_cols(df: pd.DataFrame) -> list[str]:
    """1단계 Base Model(A+B 통합 단일 모델) 학습용 제외 컬럼 = get_excluded_cols() + qty_lag2/4 계열.
    A센터 전용 2단계 잔차 보정 모델은 A에 qty_lag2/4가 정상 존재하므로 get_excluded_cols()를
    그대로 쓰고 이 함수는 쓰지 않는다."""
    return get_excluded_cols(df) + [c for c in BASE_MODEL_EXCLUDED_EXTRA if c in df.columns]


def run_sanity_checks(df: pd.DataFrame) -> None:
    print("=" * 80)
    print("[Sanity Check 1] 통합 Shape / 센터별 행 수")
    print(f"  전체 shape: {df.shape}")
    print(df[CENTER_COL].value_counts().to_string())
    print(df.groupby([CENTER_COL, "split"]).size().to_string())

    print()
    print("=" * 80)
    print("[Sanity Check 2] 상호작용 / 외부변수 Feature 샘플 5행")
    sample_cols = [
        CENTER_COL, WEEK_COL, "center_is_B", "평균온도", "총강수량", "temp_x_precip",
        "강수량_호우_flag", "강수량_호우_count",
        "공휴일_W-1", "공휴일_W0", "공휴일_W+1",
    ]
    print(df[sample_cols].sample(5, random_state=42).to_string(index=False))
    print("  (참고: qty_lag1 * center_is_B 상호작용 및 기온_극단_flag/count(영분산으로 폐기)는 미생성)")

    print()
    print("=" * 80)
    print("[Sanity Check 3] weeks_since_last_active / _filled 통계량")
    print(df[["weeks_since_last_active", "weeks_since_last_active_filled"]].describe().to_string())
    n_nan_raw = df["weeks_since_last_active"].isna().sum()
    n_nan_filled = df["weeks_since_last_active_filled"].isna().sum()
    print(f"  원본 NaN(한번도 안팔림): {n_nan_raw:,}건 / filled 잔여 NaN: {n_nan_filled:,}건 "
          f"({'정상' if n_nan_filled == 0 else '⚠ 확인 필요'})")

    print()
    print("=" * 80)
    print("[Sanity Check 4] _filled 컬럼 잔여 결측 (Day2에서 넘어온 4개 + 신규 1개, 센터별 분해)")
    filled_cols = [c for c in df.columns if c.endswith("_filled")]
    for c in filled_cols:
        n_na = df[c].isna().sum()
        by_center = df.loc[df[c].isna(), CENTER_COL].value_counts().to_dict()
        b_total = len(df[df[CENTER_COL] == "B"])
        if by_center.get("B", 0) == b_total and b_total > 0 and by_center.get("A", 0) < b_total:
            note = f"B 전체({b_total:,}행)가 원천적으로 이 컬럼을 생성하지 않음(B_LAG_WEEKS 설계상 없음) — 정상. A쪽 잔여 {by_center.get('A', 0):,}건은 과거 참고 데이터 없는 시작 시점"
        elif n_na > 0:
            note = "정상(과거 참고 데이터 없는 시작 시점)"
        else:
            note = "결측 없음"
        print(f"  {c:35s} 잔여 결측 {n_na:,}건 (센터별: {by_center}) — {note}")

    print()
    print("=" * 80)
    print("[Sanity Check 5] target_* / get_excluded_cols() 정합성")
    excluded = get_excluded_cols(df)
    target_cols = [c for c in df.columns if c.startswith("target_")]
    missing_targets = [c for c in target_cols if c not in excluded]
    feature_cols = [c for c in df.columns if c not in excluded and c != "split"]
    print(f"  target_* 컬럼({len(target_cols)}개): {target_cols}")
    print(f"  get_excluded_cols() 제외 목록({len(excluded)}개): {excluded}")
    print(f"  제외 목록에서 빠진 target 컬럼: {missing_targets if missing_targets else '없음 (전부 포함됨)'}")
    print(f"  최종 feature_cols 후보 개수(참고용, split 제외): {len(feature_cols)}개")
    leaked_in_features = [c for c in feature_cols if c.startswith("target_")]
    print(f"  feature_cols에 target_ 섞여 들어간 것: {leaked_in_features if leaked_in_features else '없음'}")

    print()
    print("=" * 80)
    print("[Sanity Check 6] 1단계 Base Model(A+B 통합) vs 2단계 A잔차보정 모델 feature set 분리")
    base_excluded = get_base_model_excluded_cols(df)
    base_feature_cols = [c for c in df.columns if c not in base_excluded and c != "split"]
    dropped_for_base = [c for c in feature_cols if c not in base_feature_cols]
    print(f"  Base Model feature_cols 개수: {len(base_feature_cols)}개 "
          f"(잔차보정 모델 대비 -{len(dropped_for_base)}개)")
    print(f"  Base Model에서만 추가로 제외된 컬럼: {dropped_for_base}")
    still_in_residual = [c for c in dropped_for_base if c in feature_cols]
    print(f"  A잔차보정 모델(get_excluded_cols 그대로 사용)에는 여전히 존재: {still_in_residual}")
    leaked_lag24_in_base = [c for c in base_feature_cols if c.startswith(("qty_lag2", "qty_lag4"))]
    print(f"  Base Model feature_cols에 qty_lag2/4 계열 잔존 여부: "
          f"{leaked_lag24_in_base if leaked_lag24_in_base else '없음 (정상 제외됨)'}")


def main():
    df = load_all_feat()
    print(f"[Day3] 5개 Day2 산출물 통합: {len(df):,}행 (A+B, train/val/test/pool)")

    df = add_holiday_calendar_features(df)
    df = add_climate_extreme_flags(df)
    df = add_covid_flag(df)
    df = add_interaction_features(df)
    df = add_weeks_since_active_filled(df)

    # 거래-조인 기반 원본 공휴일 컬럼은 B에서 구조적으로 항상 0이라(모듈 docstring 1번 참고)
    # 공휴일_W0/W-1/W+1로 완전히 대체됨 — 혼동 방지를 위해 최종 산출물에서 제거
    df = df.drop(columns=["공휴일"])

    df[CENTER_COL] = df[CENTER_COL].astype("category")

    run_sanity_checks(df)

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print()
    print("=" * 80)
    print(f"[Day3] 최종 저장 완료 -> {OUT_PATH} ({len(df):,}행, {len(df.columns)}개 컬럼)")


if __name__ == "__main__":
    main()
