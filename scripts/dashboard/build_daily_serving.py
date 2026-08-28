"""
build_daily_serving.py

daily_demand/daily_transactions/inventory_daily(각 31,127,078행, canonical) 위에 API serving
전용 소형 derivative 4종을 생성한다. canonical 3개 parquet과 weekly parquet/API/model/forecast는
이 스크립트에서 전혀 읽기만 하고 절대 재작성하지 않는다 — canonical은 검증용 단일 진실 소스로
그대로 둔다.

서로 다른 grain을 하나의 parquet에 억지로 합치지 않는다:
  1. daily_summary_agg.parquet   — (center, date, KAN_대/중/소분류, option_code) 집계
     → GET /daily/summary, /daily/category-sales 전용
  2. daily_region_agg.parquet    — (center, date, 시도, 시군구, sku_id, 상품명, 옵션코드) 집계
     → GET /daily/region-sales 전용 (원본은 raw 거래 파일, weekly와는 별개 산출물)
  3. daily_transactions_sparse.parquet — daily_transactions와 동일 스키마의 SKU-day sparse 버전
     (입고/반출/판매/반품이 전부 0인 행 제외) → /daily/returns, /daily/sales-surge, /daily/transactions
  4. daily_sku_grid.parquet      — (center, sku) 당 1행, grid_start(해당 SKU 일간 grid 시작일).
     dense canonical과 달리 sparse 소스만으로는 "이 SKU/날짜가 애초에 grid 범위 안인가"를
     판정할 수 없으므로(전부 0인 날은 sparse에서 사라지므로) 이 작은 보조 테이블로 대체한다.

금액/부호/유효 SKU 규칙은 canonical 산출물을 그대로 합산하는 것뿐이므로 재계산하지 않는다
(sum의 결합법칙 — 미리 부분합을 내도 최종 합계는 원본과 동일).
"""

from pathlib import Path

import pandas as pd

OUT_DIR = Path("data/dashboard")
FINAL_DIR = Path("data/final")


def build_summary_agg(daily_demand: pd.DataFrame, pm: pd.DataFrame) -> pd.DataFrame:
    merged = daily_demand.merge(
        pm[["center_id", "sku_id", "KAN_대분류", "KAN_중분류", "KAN_소분류", "option_code"]],
        on=["center_id", "sku_id"], how="inner",
    )
    active = (
        (merged["sales_qty"] != 0) | (merged["return_qty"] != 0)
        | (merged["sales_amount"] != 0) | (merged["return_amount"] != 0)
    )
    merged = merged[active]

    agg = merged.groupby(
        ["center_id", "date", "KAN_대분류", "KAN_중분류", "KAN_소분류", "option_code"], as_index=False
    ).agg(
        sales_qty=("sales_qty", "sum"),
        return_qty=("return_qty", "sum"),
        sales_amount=("sales_amount", "sum"),
        return_amount=("return_amount", "sum"),
        net_sales_amount=("net_sales_amount", "sum"),
        active_sku_count=("sales_qty", lambda s: int((s > 0).sum())),
    )
    return agg


def build_region_agg(pm: pd.DataFrame) -> pd.DataFrame:
    raw = pd.read_parquet(
        FINAL_DIR / "cleaned_main_joined_cleaned_for_pred.parquet",
        columns=["센터", "바코드", "옵션코드", "상품클러스터", "상품명", "거래일", "수량", "금액", "시도", "시군구"],
    )
    raw["수량"] = pd.to_numeric(raw["수량"], errors="coerce")
    raw["금액"] = pd.to_numeric(raw["금액"], errors="coerce")
    raw["거래일"] = pd.to_datetime(raw["거래일"])
    raw["center_id"] = raw["센터"].astype(str)
    raw["sku_id"] = (
        raw["바코드"].astype(str) + "*" + raw["옵션코드"].astype(str) + "*" + raw["상품클러스터"].astype(str)
    )

    raw = raw[raw["수량"] > 0]
    valid_sku_keys = set(zip(pm["center_id"], pm["sku_id"]))
    matched = [k in valid_sku_keys for k in zip(raw["center_id"], raw["sku_id"])]
    raw = raw[matched]

    agg = raw.groupby(
        ["center_id", "거래일", "시도", "시군구", "sku_id", "상품명", "옵션코드"], as_index=False
    )["금액"].sum()
    agg = agg.rename(columns={
        "거래일": "date", "시도": "sido", "시군구": "sigungu",
        "상품명": "product_name", "옵션코드": "option_code", "금액": "sales_amount",
    })
    return agg


def build_transactions_sparse(daily_transactions: pd.DataFrame) -> pd.DataFrame:
    active = (
        (daily_transactions["inbound_qty"] != 0) | (daily_transactions["outbound_qty"] != 0)
        | (daily_transactions["sales_qty"] != 0) | (daily_transactions["return_qty"] != 0)
    )
    return daily_transactions[active].reset_index(drop=True)


def build_sku_grid(daily_demand: pd.DataFrame, pm: pd.DataFrame) -> pd.DataFrame:
    """(center_id, sku_id)당 1행. grid_start(그 SKU 일간 grid 시작일) +
    option_code/KAN_대·중·소분류(product_master의 정적 속성, 그대로 복사)를 함께 들고 있어
    sparse serving만으로도 "이 조합/날짜가 애초에 grid 범위 안인가"를 판정할 수 있게 한다
    (전부 0인 날은 sparse 파일에서 사라지므로 sparse만으로는 판정 불가능)."""
    grid = daily_demand.groupby(["center_id", "sku_id"], as_index=False)["date"].min()
    grid = grid.rename(columns={"date": "grid_start"})
    grid = grid.merge(
        pm[["center_id", "sku_id", "option_code", "KAN_대분류", "KAN_중분류", "KAN_소분류"]],
        on=["center_id", "sku_id"], how="left",
    )
    return grid


def main() -> None:
    pm = pd.read_parquet(OUT_DIR / "product_master.parquet")
    daily_demand = pd.read_parquet(OUT_DIR / "daily_demand.parquet")
    daily_transactions = pd.read_parquet(OUT_DIR / "daily_transactions.parquet")

    summary_agg = build_summary_agg(daily_demand, pm)
    summary_agg.to_parquet(OUT_DIR / "daily_summary_agg.parquet", index=False)
    print(f"daily_summary_agg.parquet: {len(summary_agg):,} rows")

    region_agg = build_region_agg(pm)
    region_agg.to_parquet(OUT_DIR / "daily_region_agg.parquet", index=False)
    print(f"daily_region_agg.parquet: {len(region_agg):,} rows")

    tx_sparse = build_transactions_sparse(daily_transactions)
    tx_sparse.to_parquet(OUT_DIR / "daily_transactions_sparse.parquet", index=False)
    print(f"daily_transactions_sparse.parquet: {len(tx_sparse):,} rows")

    sku_grid = build_sku_grid(daily_demand, pm)
    sku_grid.to_parquet(OUT_DIR / "daily_sku_grid.parquet", index=False)
    print(f"daily_sku_grid.parquet: {len(sku_grid):,} rows (product_master: {len(pm):,})")


if __name__ == "__main__":
    main()
