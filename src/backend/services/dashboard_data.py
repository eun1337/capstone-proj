"""data/dashboard/*.parquet 공용 로더 — 프로세스당 1회만 읽고 캐시한다."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

DASHBOARD_DIR = Path(__file__).resolve().parents[3] / "data" / "dashboard"
DEFAULT_BASIS_WEEK = pd.Timestamp("2024-09-30")
# 일간 운영 데이터(daily_demand/daily_transactions/inventory_daily)의 실제 관측 마지막 일자.
# "실시간"이 아니라 보유 데이터 범위 내 값이며, 이 날짜를 넘는 조회일은 허용하지 않는다.
DEFAULT_DAILY_DATE = pd.Timestamp("2024-12-31")


@lru_cache(maxsize=None)
def load_product_master() -> pd.DataFrame:
    return pd.read_parquet(DASHBOARD_DIR / "product_master.parquet")


@lru_cache(maxsize=None)
def load_forecast_sku_keys() -> frozenset:
    fc = pd.read_parquet(DASHBOARD_DIR / "forecast_2024.parquet", columns=["center_id", "sku_id"])
    return frozenset(zip(fc["center_id"], fc["sku_id"]))


@lru_cache(maxsize=None)
def load_weekly_demand() -> pd.DataFrame:
    return pd.read_parquet(DASHBOARD_DIR / "weekly_demand.parquet")


@lru_cache(maxsize=None)
def load_forecast_2024() -> pd.DataFrame:
    return pd.read_parquet(DASHBOARD_DIR / "forecast_2024.parquet")


def load_default_basis_week() -> pd.Timestamp:
    """대시보드 첫 진입 시 보여주는 기본 시연 기준주(DEFAULT_BASIS_WEEK)를 반환한다.

    최신 주라는 의미가 아니라, h1/h2/h4(1/2/4주 후) 예측을 한 화면에서
    모두 보여줄 수 있도록 고정한 UI 초기 표시용 기준주다."""
    return DEFAULT_BASIS_WEEK


@lru_cache(maxsize=None)
def load_weekly_transactions() -> pd.DataFrame:
    return pd.read_parquet(DASHBOARD_DIR / "weekly_transactions.parquet")


@lru_cache(maxsize=None)
def load_inventory_weekly() -> pd.DataFrame:
    return pd.read_parquet(DASHBOARD_DIR / "inventory_weekly.parquet")


@lru_cache(maxsize=None)
def load_sales_transactions_raw() -> pd.DataFrame:
    """cleaned_main_joined_cleaned_for_pred.parquet(거래 단위 매출, 지역 포함) 원본을 캐시한다.

    5개 dashboard parquet과는 별개 소스다 — 지역별 매출처럼 dashboard 산출물에
    없는 지역 정보가 필요한 인사이트 endpoint 전용이며, sku_id/week_st 구성과
    금액 정의(수량>0 행의 금액)는 scripts/dashboard/build_dashboard_dataset.py의
    load_sales_amount_weekly()와 동일하게 맞춰 기존 sales_amount 총액과 어긋나지 않게 한다."""
    df = pd.read_parquet(
        DASHBOARD_DIR / "cleaned_main_joined_cleaned_for_pred.parquet",
        columns=["센터", "바코드", "옵션코드", "상품클러스터", "상품명", "거래일", "수량", "금액", "시도", "시군구"],
    )
    df["수량"] = pd.to_numeric(df["수량"], errors="coerce")
    df["금액"] = pd.to_numeric(df["금액"], errors="coerce")
    df["거래일"] = pd.to_datetime(df["거래일"])
    df["week_st"] = df["거래일"] - pd.to_timedelta(df["거래일"].dt.dayofweek, unit="D")
    df["center_id"] = df["센터"].astype(str)
    df["sku_id"] = (
        df["바코드"].astype(str) + "*" + df["옵션코드"].astype(str) + "*" + df["상품클러스터"].astype(str)
    )
    return df


@lru_cache(maxsize=None)
def load_daily_demand() -> pd.DataFrame:
    return pd.read_parquet(DASHBOARD_DIR / "daily_demand.parquet")


@lru_cache(maxsize=None)
def load_daily_transactions() -> pd.DataFrame:
    return pd.read_parquet(DASHBOARD_DIR / "daily_transactions.parquet")


@lru_cache(maxsize=None)
def load_inventory_daily() -> pd.DataFrame:
    return pd.read_parquet(DASHBOARD_DIR / "inventory_daily.parquet")


def load_default_daily_date() -> pd.Timestamp:
    """일간 운영 데이터 기본 조회일. 실제 보유 데이터의 마지막 관측일이며 '오늘'/'실시간'이 아니다."""
    return DEFAULT_DAILY_DATE


@lru_cache(maxsize=None)
def load_daily_summary_agg() -> pd.DataFrame:
    """/daily/summary, /daily/category-sales 전용 serving derivative.
    canonical daily_demand(31,127,078행)를 (center, date, KAN_대/중/소분류, option_code) 단위로
    미리 합산해 둔 것이며, 활동이 전혀 없는(모든 지표 0) 조합은 제외한 sparse 집계다."""
    return pd.read_parquet(DASHBOARD_DIR / "daily_summary_agg.parquet")


@lru_cache(maxsize=None)
def load_daily_region_agg() -> pd.DataFrame:
    """/daily/region-sales 전용 serving derivative. raw 거래 파일(수량>0)을
    (center, date, 시도, 시군구, sku_id, 상품명, 옵션코드) 단위로 미리 합산해 둔 것이다."""
    return pd.read_parquet(DASHBOARD_DIR / "daily_region_agg.parquet")


@lru_cache(maxsize=None)
def load_daily_transactions_sparse() -> pd.DataFrame:
    """/daily/returns, /daily/sales-surge, /daily/transactions 전용 serving derivative.
    canonical daily_transactions과 스키마가 동일한 SKU-day sparse 버전이며,
    입고/반출/판매/반품이 전부 0인 행은 제외돼 있다(없는 날 = 0으로 취급)."""
    return pd.read_parquet(DASHBOARD_DIR / "daily_transactions_sparse.parquet")


@lru_cache(maxsize=None)
def load_daily_sku_grid() -> pd.DataFrame:
    """(center_id, sku_id)당 1행, grid_start(해당 SKU 일간 grid 시작일). dense canonical과
    달리 sparse 소스만으로는 "이 날짜가 애초에 grid 범위 안인가"를 판정할 수 없어
    (전부 0인 날은 sparse에서 사라지므로) 이 작은 보조 테이블로 grid 존재 여부를 판정한다."""
    return pd.read_parquet(DASHBOARD_DIR / "daily_sku_grid.parquet")


def resolve_forecast_basis_week_for_date(operational_date: pd.Timestamp):
    """주어진 조회일(operational_date) 시점에 이미 존재했을 가장 최근 forecast_2024 원본
    주(week_st <= operational_date)를 반환한다. 미래 forecast origin은 절대 끌어오지 않는다.
    해당 조회일보다 이전에 origin이 하나도 없으면 None을 반환한다."""
    fc = load_forecast_2024()
    candidates = fc.loc[fc["week_st"] <= operational_date, "week_st"]
    if candidates.empty:
        return None
    return candidates.max()
