"""
build_dashboard_dataset.py

대시보드용 read-only 파생 데이터셋을 data/dashboard/ 아래에 생성한다. 원본/모델 산출물은
전혀 수정하지 않는다(읽기만 함). SKU=center+barcode+option_code+product_cluster, EA/BX/CS는
환산하지 않고 각 SKU 원 단위를 그대로 유지한다.

산출:
  product_master.parquet, weekly_demand.parquet, weekly_transactions.parquet,
  inventory_weekly.parquet, forecast_2024.parquet
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path("data/dashboard")
GRID_END = pd.Timestamp("2024-12-30")  # 2024-12-31을 포함하는 월요일 시작 주
REGIME_SHIFT_DATE = pd.Timestamp("2023-07-01")


def _sku_id(barcode: pd.Series, option: pd.Series, cluster: pd.Series) -> pd.Series:
    return barcode.astype(str) + "*" + option.astype(str) + "*" + cluster.astype(str)


def load_sales_raw() -> pd.DataFrame:
    cols = [
        "주시작일", "센터", "시도", "시군구", "바코드", "옵션코드", "상품클러스터",
        "총판매수량", "반품수량", "상품명", "규격", "입수",
        "KAN_대분류", "KAN_중분류", "KAN_소분류",
    ]
    a = pd.read_parquet("data/final/master_demand_weekly_A.parquet", columns=cols)
    b = pd.read_parquet("data/final/master_demand_weekly_B_with_regime.parquet", columns=cols + ["레짐"])
    a["레짐"] = None
    df = pd.concat([a, b], ignore_index=True)
    df = df.rename(columns={"주시작일": "week_st", "센터": "center_id"})
    df["sku_id"] = _sku_id(df["바코드"], df["옵션코드"], df["상품클러스터"])
    return df


# 공식 KAN 분류표(data/master/[대한상공회의소]KAN상품분류코드.xlsx) 기준 metadata correction.
# master_demand_weekly_A/B 원본의 KAN 분류 실패로 "ERROR"가 남아 있던 건 중, 공식표 대조로
# 유일하게 확정 가능했던 건만 적용한다. ERROR가 아닌 정상 KAN 값은 건드리지 않는다.
KAN_ERROR_CORRECTIONS = {
    ("A", "18801074254059*BX*1"): {  # 한성배즙숙성후랑크70g — KAN_CODE 1030101/1030102 대조
        "KAN_대분류": "가공식품",
        "KAN_중분류": "축산가공식품",
        "KAN_소분류": "햄/소시지(통조림/병제외)",
    },
}


def apply_kan_error_corrections(product_master: pd.DataFrame) -> pd.DataFrame:
    df = product_master.copy()
    for (center_id, sku_id), correction in KAN_ERROR_CORRECTIONS.items():
        mask = (df["center_id"] == center_id) & (df["sku_id"] == sku_id)
        for col, val in correction.items():
            df.loc[mask, col] = val
    return df


def build_product_master(sales_raw: pd.DataFrame) -> pd.DataFrame:
    desc_cols = ["상품명", "규격", "입수", "KAN_대분류", "KAN_중분류", "KAN_소분류"]
    agg = sales_raw.sort_values("week_st").groupby(["center_id", "sku_id"], as_index=False).agg(
        barcode=("바코드", "first"), option_code=("옵션코드", "first"), product_cluster=("상품클러스터", "first"),
        **{c: (c, "first") for c in desc_cols},
    )
    result = agg[["center_id", "sku_id", "barcode", "option_code", "product_cluster"] + desc_cols]
    return apply_kan_error_corrections(result)


def build_sales_weekly(sales_raw: pd.DataFrame) -> pd.DataFrame:
    agg = sales_raw.groupby(["center_id", "sku_id", "week_st"], as_index=False).agg(
        sales_qty=("총판매수량", "sum"), return_qty=("반품수량", "sum"),
    )
    return agg


def load_sales_amount_weekly(valid_sku_keys: set) -> pd.DataFrame:
    """cleaned_main_joined_cleaned_for_pred.parquet(거래 단위 매출)을 기존 SKU 매핑
    기준(barcode+option_code+상품클러스터, product_master에 존재하는 sku만)으로 연결해
    주별 sales_amount/return_amount를 집계한다. 재고 계산에는 사용하지 않는다."""
    tx = pd.read_parquet(
        "data/final/cleaned_main_joined_cleaned_for_pred.parquet",
        columns=["센터", "바코드", "옵션코드", "상품클러스터", "거래일", "수량", "금액"],
    )
    tx["수량"] = pd.to_numeric(tx["수량"], errors="coerce")
    tx["금액"] = pd.to_numeric(tx["금액"], errors="coerce")
    tx["거래일"] = pd.to_datetime(tx["거래일"])
    tx["week_st"] = tx["거래일"] - pd.to_timedelta(tx["거래일"].dt.dayofweek, unit="D")
    tx["center_id"] = tx["센터"].astype(str)
    tx["sku_id"] = _sku_id(tx["바코드"], tx["옵션코드"], tx["상품클러스터"])

    matched_mask = [k in valid_sku_keys for k in zip(tx["center_id"], tx["sku_id"])]
    tx = tx[matched_mask].copy()

    tx["_sales_amount"] = tx["금액"].where(tx["수량"] > 0, 0.0)
    tx["_return_amount"] = (-tx["금액"]).where(tx["수량"] < 0, 0.0)
    agg = tx.groupby(["center_id", "sku_id", "week_st"], as_index=False).agg(
        sales_amount=("_sales_amount", "sum"), return_amount=("_return_amount", "sum"),
    )
    return agg


def find_ambiguous_cluster_keys(sales_raw: pd.DataFrame) -> set:
    """동일 (center,바코드,옵션코드)에 product_cluster가 2개 이상 존재하는 조합.
    매입의 상품클러스터 원값은 이 조합에서는 신뢰하지 않고 resolve_ambiguous_cluster로
    다시 판정한다."""
    nunique = sales_raw.groupby(["center_id", "바코드", "옵션코드"])["상품클러스터"].nunique()
    return set(nunique[nunique > 1].index)


def build_cluster_info(sales_raw: pd.DataFrame, ambiguous_keys: set) -> dict:
    """ambiguous 조합별 cluster의 규격/상품명 집합과 판매 활동기간(min~max week_st)."""
    idx = sales_raw.set_index(["center_id", "바코드", "옵션코드"]).index
    sub = sales_raw[idx.isin(ambiguous_keys)]
    info = {}
    for (center, bc, opt), g in sub.groupby(["center_id", "바코드", "옵션코드"]):
        clusters = {}
        for cluster, cg in g.groupby("상품클러스터"):
            clusters[cluster] = {
                "규격": set(cg["규격"].dropna().astype(str).str.strip().unique()),
                "상품명": set(cg["상품명"].dropna().astype(str).str.strip().unique()),
                "min_week": cg["week_st"].min(),
                "max_week": cg["week_st"].max(),
            }
        info[(center, bc, opt)] = clusters
    return info


def resolve_ambiguous_cluster(spec, name, week_st, clusters: dict):
    """1순위: 규격/상품명이 유일한 cluster와 일치. 2순위: cluster 활동기간이 서로 겹치지
    않고 매입 week_st가 그중 한 cluster 기간에만 속함. 둘 다 안 되면 (None, None)."""
    spec = str(spec).strip() if pd.notna(spec) else None
    name = str(name).strip() if pd.notna(name) else None
    # 상품명/규격을 각각 독립적으로 확인한다(둘을 합쳐서 union으로 보면, 두 cluster가
    # 우연히 규격은 같고 상품명만 다른 경우 규격 쪽에서 양쪽 다 걸려 오히려 모호해진다).
    name_matches = {c for c, info in clusters.items() if name is not None and name in info["상품명"]}
    if len(name_matches) == 1:
        return name_matches.pop(), "spec"
    spec_matches = {c for c, info in clusters.items() if spec is not None and spec in info["규격"]}
    if len(spec_matches) == 1:
        return spec_matches.pop(), "spec"

    ranges = sorted(clusters.items(), key=lambda kv: kv[1]["min_week"])
    overlap = any(ranges[i][1]["max_week"] >= ranges[i + 1][1]["min_week"] for i in range(len(ranges) - 1))
    if not overlap:
        containing = [c for c, info in clusters.items() if info["min_week"] <= week_st <= info["max_week"]]
        if len(containing) == 1:
            return containing[0], "date"

    return None, None


def load_purchase_weekly(valid_sku_keys: set, ambiguous_keys: set, cluster_info: dict) -> tuple:
    pur = pd.read_parquet(
        "data/final/cleaned_purchase_cleaned_for_pred.parquet",
        columns=["센터", "바코드", "옵션코드", "상품클러스터", "작업유형", "일자", "수량", "규격", "상품명", "금액", "부가세"],
    )
    pur["수량"] = pd.to_numeric(pur["수량"], errors="coerce")
    pur["금액"] = pd.to_numeric(pur["금액"], errors="coerce")
    pur["부가세"] = pd.to_numeric(pur["부가세"], errors="coerce")
    pur["일자"] = pd.to_datetime(pur["일자"])
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
    # .loc 부분 할당 과정에서 int64 컬럼 전체가 float64로 승격되어(1 -> 1.0) sku_id
    # 문자열이 달라지는 것을 막기 위해 필터링 후 다시 정수로 되돌린다.
    pur["resolved_cluster"] = pur["resolved_cluster"].astype(int)
    pur["sku_id"] = _sku_id(pur["바코드"], pur["옵션코드"], pur["resolved_cluster"])

    sku_key = list(zip(pur["center_id"], pur["sku_id"]))
    matched_mask = [k in valid_sku_keys for k in sku_key]
    pur = pur[matched_mask].copy()
    n_matched = len(pur)

    # 작업유형/수량 부호를 그대로 보존해서 합산한다(abs() 일괄 적용 금지) - 입고는
    # 원본 부호 그대로(드문 음수는 취소/정정성 거래로 확인됨), 반출도 원본 부호(거의
    # 항상 음수) 그대로 두고 stock delta 계산 시 더하기만 하면 자연히 재고가 줄어든다.
    inbound = pur[pur["작업유형"] == "입고"].groupby(["center_id", "sku_id", "week_st"], as_index=False).agg(
        inbound_qty=("수량", "sum"), inbound_amount=("금액", "sum"), inbound_vat=("부가세", "sum"),
    )
    outbound = pur[pur["작업유형"] == "반출"].groupby(["center_id", "sku_id", "week_st"], as_index=False).agg(
        outbound_qty=("수량", "sum"), outbound_amount=("금액", "sum"), outbound_vat=("부가세", "sum"),
    )

    weekly = inbound.merge(outbound, on=["center_id", "sku_id", "week_st"], how="outer")
    amount_cols = ["inbound_qty", "outbound_qty", "inbound_amount", "outbound_amount", "inbound_vat", "outbound_vat"]
    weekly[amount_cols] = weekly[amount_cols].fillna(0.0)
    weekly["inbound_total_amount"] = weekly["inbound_amount"] + weekly["inbound_vat"]
    weekly["outbound_total_amount"] = weekly["outbound_amount"] + weekly["outbound_vat"]

    inbound_skus = set(zip(pur.loc[pur["작업유형"] == "입고", "center_id"], pur.loc[pur["작업유형"] == "입고", "sku_id"]))
    return weekly, n_total, n_spec, n_date, n_still_ambiguous, n_matched, inbound_skus


def build_grid(sales_weekly: pd.DataFrame, purchase_weekly: pd.DataFrame) -> pd.DataFrame:
    sales_first = sales_weekly.groupby(["center_id", "sku_id"])["week_st"].min()
    pur_first = purchase_weekly.groupby(["center_id", "sku_id"])["week_st"].min()
    first = pd.concat([sales_first, pur_first], axis=1)
    first.columns = ["sales_first", "pur_first"]
    grid_start = first.min(axis=1).rename("grid_start").reset_index()

    frames = []
    for row in grid_start.itertuples(index=False):
        weeks = pd.date_range(row.grid_start, GRID_END, freq="7D")
        frames.append(pd.DataFrame({"center_id": row.center_id, "sku_id": row.sku_id, "week_st": weeks}))
    return pd.concat(frames, ignore_index=True)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    sales_raw = load_sales_raw()
    product_master = build_product_master(sales_raw)
    sales_weekly = build_sales_weekly(sales_raw)

    valid_sku_keys = set(zip(product_master["center_id"], product_master["sku_id"]))
    ambiguous_keys = find_ambiguous_cluster_keys(sales_raw)
    cluster_info = build_cluster_info(sales_raw, ambiguous_keys)
    purchase_weekly, n_pur_total, n_spec, n_date, n_still_ambiguous, n_pur_matched, inbound_skus = (
        load_purchase_weekly(valid_sku_keys, ambiguous_keys, cluster_info)
    )

    sales_amount_weekly = load_sales_amount_weekly(valid_sku_keys)

    grid = build_grid(sales_weekly, purchase_weekly)

    # ---- weekly_demand.parquet ----
    wd = grid.merge(sales_weekly, on=["center_id", "sku_id", "week_st"], how="left")
    wd = wd.merge(sales_amount_weekly, on=["center_id", "sku_id", "week_st"], how="left")
    amount_fill_cols = ["sales_qty", "return_qty", "sales_amount", "return_amount"]
    wd[amount_fill_cols] = wd[amount_fill_cols].fillna(0.0)
    wd["net_qty"] = wd["sales_qty"] - wd["return_qty"]
    wd["net_sales_amount"] = wd["sales_amount"] - wd["return_amount"]
    is_b = wd["center_id"] == "B"
    wd["regime"] = np.where(is_b, np.where(wd["week_st"] >= REGIME_SHIFT_DATE, "post", "pre"), None)
    weekly_demand = wd[[
        "center_id", "sku_id", "week_st", "sales_qty", "return_qty", "net_qty",
        "sales_amount", "return_amount", "net_sales_amount", "regime",
    ]]
    weekly_demand.to_parquet(OUT_DIR / "weekly_demand.parquet", index=False)

    # ---- weekly_transactions.parquet ----
    option_map = product_master.set_index(["center_id", "sku_id"])["option_code"]
    wt = grid.merge(purchase_weekly, on=["center_id", "sku_id", "week_st"], how="left")
    wt = wt.merge(sales_weekly, on=["center_id", "sku_id", "week_st"], how="left")
    wt = wt.merge(sales_amount_weekly, on=["center_id", "sku_id", "week_st"], how="left")
    qty_amount_cols = [
        "inbound_qty", "outbound_qty", "sales_qty", "return_qty",
        "inbound_amount", "outbound_amount", "inbound_vat", "outbound_vat",
        "inbound_total_amount", "outbound_total_amount", "sales_amount", "return_amount",
    ]
    wt[qty_amount_cols] = wt[qty_amount_cols].fillna(0.0)
    wt["option_code"] = wt.set_index(["center_id", "sku_id"]).index.map(option_map).to_numpy()
    weekly_transactions = wt[
        ["center_id", "sku_id", "week_st", "option_code", "inbound_qty", "outbound_qty", "sales_qty", "return_qty"]
        + [
            "inbound_amount", "outbound_amount", "inbound_vat", "outbound_vat",
            "inbound_total_amount", "outbound_total_amount", "sales_amount", "return_amount",
        ]
    ]
    weekly_transactions.to_parquet(OUT_DIR / "weekly_transactions.parquet", index=False)

    # ---- inventory_weekly.parquet ----
    # inventory_available은 "유효한 입고 거래가 최소 1건" 있는 SKU만 true(반출만 있는 SKU는 false).
    inv = wt[
        ["center_id", "sku_id", "week_st", "option_code", "inbound_qty", "outbound_qty", "sales_qty", "return_qty"]
    ].copy()
    inv["inventory_available"] = inv.set_index(["center_id", "sku_id"]).index.isin(inbound_skus)

    # 반출 수량은 원본이 이미 음수라 더하기만 하면 재고가 줄어든다(빼면 이중 음수가 됨).
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

    inventory_weekly = inv[[
        "center_id", "sku_id", "week_st", "option_code",
        "estimated_opening_inventory", "estimated_inventory_return_excluded",
        "estimated_inventory_return_included", "inventory_available",
    ]]
    inventory_weekly.to_parquet(OUT_DIR / "inventory_weekly.parquet", index=False)

    product_master.to_parquet(OUT_DIR / "product_master.parquet", index=False)

    # ---- forecast_2024.parquet ----
    horizon_frames = []
    for h in (1, 2, 4):
        pred = pd.read_parquet(
            f"outputs/final_holdout/lightgbm_hurdle/prediction_lightgbm_hurdle_h{h}.parquet",
            columns=["center_id", "sku_id", "week_st", "target_date", "y_pred", "sale_probability"],
        )
        pred = pred.rename(columns={
            "target_date": f"h{h}_target_date", "y_pred": f"h{h}_pred", "sale_probability": f"h{h}_sale_probability",
        })
        horizon_frames.append(pred)

    forecast_2024 = horizon_frames[0]
    for frame in horizon_frames[1:]:
        forecast_2024 = forecast_2024.merge(frame, on=["center_id", "sku_id", "week_st"], how="outer")
    forecast_2024.to_parquet(OUT_DIR / "forecast_2024.parquet", index=False)

    print(f"purchase 매핑: {n_pur_matched:,}/{n_pur_total:,} ({n_pur_matched/n_pur_total:.4%}) matched")
    print(f"  multi-cluster 조합 중 상품정보(규격/상품명) 기준 재배정: {n_spec:,}건")
    print(f"  multi-cluster 조합 중 시점(활동기간) 기준 재배정: {n_date:,}건")
    print(f"  multi-cluster 조합 중 끝까지 ambiguous(제외): {n_still_ambiguous:,}건")
    print(f"  나머지 unmatched(제외): {n_pur_total - n_still_ambiguous - n_pur_matched:,}건")
    print("저장 완료:", sorted(p.name for p in OUT_DIR.glob("*.parquet")))


if __name__ == "__main__":
    main()
