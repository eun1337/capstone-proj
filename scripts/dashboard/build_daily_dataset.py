"""
build_daily_dataset.py

일간 운영 데이터(daily_demand/daily_transactions/inventory_daily) 파생 데이터셋을
data/dashboard/ 아래에 생성한다. 기존 weekly parquet/API/model/forecast는 전혀 건드리지
않는다(별도 산출물). SKU 매핑(barcode*option_code*product_cluster), 금액/수량 부호 규칙,
multi-cluster 재배정 로직은 scripts/dashboard/build_dashboard_dataset.py에서 그대로
재사용한다(임의 계산기준 없음) — 최종 집계 단위만 week_st 대신 date(일)로 바꾼다.

AI 예측(h1/h2/h4, sale_probability, forecast_2024)은 이 스크립트가 다루지 않는다.

산출:
  daily_demand.parquet, daily_transactions.parquet, inventory_daily.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

from scripts.dashboard.build_dashboard_dataset import (
    _sku_id,
    build_cluster_info,
    find_ambiguous_cluster_keys,
    load_sales_raw,
    resolve_ambiguous_cluster,
)

OUT_DIR = Path("data/dashboard")
DAILY_GRID_END = pd.Timestamp("2024-12-31")  # 실제 관측된 마지막 일자(원본 데이터 최대 거래일/일자)


def load_sales_daily(valid_sku_keys: set) -> pd.DataFrame:
    """cleaned_main_joined_cleaned_for_pred.parquet을 거래일 단위로 집계한다.
    부호 규칙은 weekly의 load_sales_amount_weekly와 동일(수량>0 -> 판매, 수량<0 -> 반품)하며
    금액뿐 아니라 수량도 같은 규칙으로 함께 집계한다."""
    tx = pd.read_parquet(
        "data/final/cleaned_main_joined_cleaned_for_pred.parquet",
        columns=["센터", "바코드", "옵션코드", "상품클러스터", "거래일", "수량", "금액"],
    )
    tx["수량"] = pd.to_numeric(tx["수량"], errors="coerce")
    tx["금액"] = pd.to_numeric(tx["금액"], errors="coerce")
    tx["거래일"] = pd.to_datetime(tx["거래일"])
    tx["center_id"] = tx["센터"].astype(str)
    tx["sku_id"] = _sku_id(tx["바코드"], tx["옵션코드"], tx["상품클러스터"])

    matched_mask = [k in valid_sku_keys for k in zip(tx["center_id"], tx["sku_id"])]
    tx = tx[matched_mask].copy()

    tx["_sales_qty"] = tx["수량"].where(tx["수량"] > 0, 0.0)
    tx["_return_qty"] = (-tx["수량"]).where(tx["수량"] < 0, 0.0)
    tx["_sales_amount"] = tx["금액"].where(tx["수량"] > 0, 0.0)
    tx["_return_amount"] = (-tx["금액"]).where(tx["수량"] < 0, 0.0)

    agg = tx.groupby(["center_id", "sku_id", "거래일"], as_index=False).agg(
        sales_qty=("_sales_qty", "sum"), return_qty=("_return_qty", "sum"),
        sales_amount=("_sales_amount", "sum"), return_amount=("_return_amount", "sum"),
    )
    return agg.rename(columns={"거래일": "date"})


def load_purchase_daily(valid_sku_keys: set, ambiguous_keys: set, cluster_info: dict) -> tuple:
    """load_purchase_weekly와 완전히 동일한 multi-cluster 재배정 로직(행 단위)을 재사용하고,
    최종 집계 단위만 week_st 대신 date(일)로 바꾼다."""
    pur = pd.read_parquet(
        "data/final/cleaned_purchase_cleaned_for_pred.parquet",
        columns=["센터", "바코드", "옵션코드", "상품클러스터", "작업유형", "일자", "수량", "규격", "상품명", "금액", "부가세"],
    )
    pur["수량"] = pd.to_numeric(pur["수량"], errors="coerce")
    pur["금액"] = pd.to_numeric(pur["금액"], errors="coerce")
    pur["부가세"] = pd.to_numeric(pur["부가세"], errors="coerce")
    pur["일자"] = pd.to_datetime(pur["일자"])
    # resolve_ambiguous_cluster의 2순위(활동기간) 판정은 week_st 버킷을 기준으로 하므로
    # 원본과 동일하게 week_st도 계산해 그대로 넘긴다(실제로는 1순위 상품명/규격에서 전량 해결됨).
    pur["week_st"] = pur["일자"] - pd.to_timedelta(pur["일자"].dt.dayofweek, unit="D")
    pur["center_id"] = pur["센터"].astype(str)

    n_total = len(pur)
    bco_key = list(zip(pur["center_id"], pur["바코드"], pur["옵션코드"]))
    is_ambiguous = pd.Series([k in ambiguous_keys for k in bco_key], index=pur.index)

    resolved_cluster = pur["상품클러스터"].copy()
    n_spec, n_date = 0, 0
    for i in pur.index[is_ambiguous]:
        row = pur.loc[i]
        key = (row["center_id"], row["바코드"], row["옵션코드"])
        cluster, how = resolve_ambiguous_cluster(row["규격"], row["상품명"], row["week_st"], cluster_info[key])
        if cluster is None:
            resolved_cluster.loc[i] = None
        else:
            resolved_cluster.loc[i] = cluster
            n_spec += how == "spec"
            n_date += how == "date"
    n_still_ambiguous = int(is_ambiguous.sum() - n_spec - n_date)

    pur["resolved_cluster"] = resolved_cluster
    pur = pur[pur["resolved_cluster"].notna()].copy()
    pur["resolved_cluster"] = pur["resolved_cluster"].astype(int)
    pur["sku_id"] = _sku_id(pur["바코드"], pur["옵션코드"], pur["resolved_cluster"])

    sku_key = list(zip(pur["center_id"], pur["sku_id"]))
    matched_mask = [k in valid_sku_keys for k in sku_key]
    pur = pur[matched_mask].copy()
    n_matched = len(pur)

    inbound = pur[pur["작업유형"] == "입고"].groupby(["center_id", "sku_id", "일자"], as_index=False).agg(
        inbound_qty=("수량", "sum"), inbound_amount=("금액", "sum"), inbound_vat=("부가세", "sum"),
    )
    outbound = pur[pur["작업유형"] == "반출"].groupby(["center_id", "sku_id", "일자"], as_index=False).agg(
        outbound_qty=("수량", "sum"), outbound_amount=("금액", "sum"), outbound_vat=("부가세", "sum"),
    )
    inbound = inbound.rename(columns={"일자": "date"})
    outbound = outbound.rename(columns={"일자": "date"})

    daily = inbound.merge(outbound, on=["center_id", "sku_id", "date"], how="outer")
    amount_cols = ["inbound_qty", "outbound_qty", "inbound_amount", "outbound_amount", "inbound_vat", "outbound_vat"]
    daily[amount_cols] = daily[amount_cols].fillna(0.0)
    daily["inbound_total_amount"] = daily["inbound_amount"] + daily["inbound_vat"]
    daily["outbound_total_amount"] = daily["outbound_amount"] + daily["outbound_vat"]

    inbound_skus = set(zip(pur.loc[pur["작업유형"] == "입고", "center_id"], pur.loc[pur["작업유형"] == "입고", "sku_id"]))
    return daily, n_total, n_spec, n_date, n_still_ambiguous, n_matched, inbound_skus


def build_daily_grid(sales_daily: pd.DataFrame, purchase_daily: pd.DataFrame) -> pd.DataFrame:
    sales_first = sales_daily.groupby(["center_id", "sku_id"])["date"].min()
    pur_first = purchase_daily.groupby(["center_id", "sku_id"])["date"].min()
    first = pd.concat([sales_first, pur_first], axis=1)
    first.columns = ["sales_first", "pur_first"]
    grid_start = first.min(axis=1).rename("grid_start").reset_index()

    frames = []
    for row in grid_start.itertuples(index=False):
        days = pd.date_range(row.grid_start, DAILY_GRID_END, freq="D")
        frames.append(pd.DataFrame({"center_id": row.center_id, "sku_id": row.sku_id, "date": days}))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # product_master는 이미 확정된 산출물을 그대로 읽는다(재계산하지 않음 — 단일 진실 소스 유지).
    product_master = pd.read_parquet(OUT_DIR / "product_master.parquet")
    valid_sku_keys = set(zip(product_master["center_id"], product_master["sku_id"]))

    # ambiguous 판정/cluster_info는 주간 소스(sales_raw)로 충분하다 — 판정 자체가 판매 활동
    # 이력의 상품명/규격/기간에 기반하므로 일/주 집계 단위와 무관하다.
    sales_raw = load_sales_raw()
    ambiguous_keys = find_ambiguous_cluster_keys(sales_raw)
    cluster_info = build_cluster_info(sales_raw, ambiguous_keys)

    sales_daily = load_sales_daily(valid_sku_keys)
    purchase_daily, n_pur_total, n_spec, n_date, n_still_ambiguous, n_pur_matched, inbound_skus = (
        load_purchase_daily(valid_sku_keys, ambiguous_keys, cluster_info)
    )

    grid = build_daily_grid(sales_daily, purchase_daily)

    # ---- daily_demand.parquet ----
    dd = grid.merge(sales_daily, on=["center_id", "sku_id", "date"], how="left")
    fill_cols = ["sales_qty", "return_qty", "sales_amount", "return_amount"]
    dd[fill_cols] = dd[fill_cols].fillna(0.0)
    dd["net_sales_amount"] = dd["sales_amount"] - dd["return_amount"]
    daily_demand = dd[[
        "center_id", "sku_id", "date", "sales_qty", "return_qty",
        "sales_amount", "return_amount", "net_sales_amount",
    ]]
    daily_demand.to_parquet(OUT_DIR / "daily_demand.parquet", index=False)

    # ---- daily_transactions.parquet ----
    option_map = product_master.set_index(["center_id", "sku_id"])["option_code"]
    dt = grid.merge(purchase_daily, on=["center_id", "sku_id", "date"], how="left")
    dt = dt.merge(sales_daily, on=["center_id", "sku_id", "date"], how="left")
    qty_amount_cols = [
        "inbound_qty", "outbound_qty", "sales_qty", "return_qty",
        "inbound_amount", "outbound_amount", "inbound_vat", "outbound_vat",
        "inbound_total_amount", "outbound_total_amount", "sales_amount", "return_amount",
    ]
    dt[qty_amount_cols] = dt[qty_amount_cols].fillna(0.0)
    dt["option_code"] = dt.set_index(["center_id", "sku_id"]).index.map(option_map).to_numpy()
    daily_transactions = dt[
        ["center_id", "sku_id", "date", "option_code", "inbound_qty", "outbound_qty", "sales_qty", "return_qty"]
        + [
            "inbound_amount", "outbound_amount",
            "inbound_total_amount", "outbound_total_amount", "sales_amount", "return_amount",
        ]
    ]
    daily_transactions.to_parquet(OUT_DIR / "daily_transactions.parquet", index=False)

    # ---- inventory_daily.parquet ----
    # weekly inventory_weekly와 완전히 동일한 원칙(최소 기초재고, signed inbound/outbound,
    # gross sales 차감, return excluded/included)을 일 단위로 그대로 적용한다.
    inv = dt[
        ["center_id", "sku_id", "date", "option_code", "inbound_qty", "outbound_qty", "sales_qty", "return_qty"]
    ].copy()
    inv["inventory_available"] = inv.set_index(["center_id", "sku_id"]).index.isin(inbound_skus)
    inv = inv.sort_values(["center_id", "sku_id", "date"])

    inv["delta_return_excluded"] = inv["inbound_qty"] + inv["outbound_qty"] - inv["sales_qty"]
    inv["cum_return_excluded"] = inv.groupby(["center_id", "sku_id"])["delta_return_excluded"].cumsum()
    min_cum = inv.groupby(["center_id", "sku_id"])["cum_return_excluded"].transform("min")
    opening_inventory = (-min_cum).clip(lower=0)
    inv["estimated_opening_inventory"] = np.where(inv["inventory_available"], opening_inventory, np.nan)
    inv["estimated_inventory_return_excluded"] = np.where(
        inv["inventory_available"], inv["estimated_opening_inventory"] + inv["cum_return_excluded"], np.nan,
    )
    inv["cum_return"] = inv.groupby(["center_id", "sku_id"])["return_qty"].cumsum()
    inv["estimated_inventory_return_included"] = np.where(
        inv["inventory_available"],
        inv["estimated_inventory_return_excluded"] + inv["cum_return"],
        np.nan,
    )

    inventory_daily = inv[[
        "center_id", "sku_id", "date", "option_code",
        "estimated_opening_inventory", "estimated_inventory_return_excluded",
        "estimated_inventory_return_included", "inventory_available",
    ]]
    inventory_daily.to_parquet(OUT_DIR / "inventory_daily.parquet", index=False)

    print(f"purchase 매핑: {n_pur_matched:,}/{n_pur_total:,} ({n_pur_matched/n_pur_total:.4%}) matched")
    print(f"  multi-cluster 조합 중 상품정보(규격/상품명) 기준 재배정: {n_spec:,}건")
    print(f"  multi-cluster 조합 중 시점(활동기간) 기준 재배정: {n_date:,}건")
    print(f"  multi-cluster 조합 중 끝까지 ambiguous(제외): {n_still_ambiguous:,}건")
    print(f"  나머지 unmatched(제외): {n_pur_total - n_still_ambiguous - n_pur_matched:,}건")
    print(f"daily_demand rows: {len(daily_demand):,}")
    print(f"daily_transactions rows: {len(daily_transactions):,}")
    print(f"inventory_daily rows: {len(inventory_daily):,}")
    print("저장 완료:", ["daily_demand.parquet", "daily_transactions.parquet", "inventory_daily.parquet"])


if __name__ == "__main__":
    main()
