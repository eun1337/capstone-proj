"""
holiday_calendar.py
거래 유무와 무관한 순수 캘린더(설날/추석) 기준 공휴일 feature 생성 helper.
archive/feature_engineering/feature_external_interaction_concat.py에서 statistical
트랙(03_prepare_arimax_exog.py)이 필요로 하는 부분만 active source로 분리했다
(나머지 feature_table_final.parquet 생성 로직은 legacy이므로 archive에 그대로 둔다).
"""

import pandas as pd

WEEK_COL = "week_st"

HOLIDAY_TARGET_HORIZONS = [1, 2, 4]

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


def add_holiday_calendar_features(
    df: pd.DataFrame, horizons: list[int] = HOLIDAY_TARGET_HORIZONS
) -> pd.DataFrame:
    """target_h{h}_공휴일_W0/W-1/W+1: 거래 유무와 무관한 순수 캘린더(설날/추석) 기준 A/B 공통
    단일 변수. 옛 거래-조인 방식(공휴일_D0)은 B가 명절 당일 출고를 거의 기록하지 않아 B 전체가
    항상 0이 되는 구조적 결함이 있었음. KOREAN_LUNAR_HOLIDAYS의 각
    날짜를 그 날짜가 속한 주(월요일)로 변환해 "명절이 포함된 주" 집합을 만들고, 이 집합에
    대한 소속 여부만으로 판정하므로 어떤 거래 기록도 필요 없다.
    각 horizon h에 대해 target_date = week_st + h주를 기준점으로 삼는다(옛 origin 주 기준
    버전은 h1 target 주만 우연히 커버했고 h2/h4는 전혀 못 봤음).
    W0=target 주 자체, W-1=target 주 바로 이전 1주, W+1=target 주 바로 다음 1주 — 표준 부호
    (target_date 기준 -1주/+1주 그대로, LEAD/LAG 아님)."""
    holiday_dates = pd.Series(KOREAN_LUNAR_HOLIDAYS)
    holiday_week_starts = holiday_dates - pd.to_timedelta(holiday_dates.dt.weekday, unit="D")
    holiday_weeks = set(holiday_week_starts)

    for h in horizons:
        target_date = df[WEEK_COL] + pd.Timedelta(weeks=h)
        df[f"target_h{h}_공휴일_W0"] = target_date.isin(holiday_weeks).astype(int)
        df[f"target_h{h}_공휴일_W-1"] = (target_date - pd.Timedelta(weeks=1)).isin(holiday_weeks).astype(int)
        df[f"target_h{h}_공휴일_W+1"] = (target_date + pd.Timedelta(weeks=1)).isin(holiday_weeks).astype(int)
    return df
