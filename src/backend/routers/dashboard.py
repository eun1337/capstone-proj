from datetime import date
from typing import Dict, List, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from services.dashboard_data import (
    load_daily_demand,
    load_daily_region_agg,
    load_daily_sku_grid,
    load_daily_summary_agg,
    load_daily_transactions,
    load_daily_transactions_sparse,
    load_default_basis_week,
    load_default_daily_date,
    load_forecast_2024,
    load_forecast_sku_keys,
    load_inventory_daily,
    load_inventory_weekly,
    load_product_master,
    load_sales_transactions_raw,
    load_weekly_demand,
    load_weekly_transactions,
    resolve_forecast_basis_week_for_date,
)

router = APIRouter()


def _filter_by_center(df: pd.DataFrame, center: Optional[str]) -> pd.DataFrame:
    if center and center != "ALL":
        df = df[df["center_id"] == center]
    return df


class CategoryNode(BaseModel):
    label:      str
    sku_count:  int
    children:   Optional[List["CategoryNode"]] = None


@router.get("/categories", response_model=List[CategoryNode])
def get_categories(center: Optional[str] = Query(None, description="A / B / ALL")):
    df = _filter_by_center(load_product_master(), center)

    tree: List[CategoryNode] = []
    for large, g_large in df.groupby("KAN_대분류", sort=True):
        middles: List[CategoryNode] = []
        for middle, g_middle in g_large.groupby("KAN_중분류", sort=True):
            smalls = [
                CategoryNode(label=small, sku_count=len(g_small))
                for small, g_small in g_middle.groupby("KAN_소분류", sort=True)
            ]
            middles.append(CategoryNode(label=middle, sku_count=len(g_middle), children=smalls))
        tree.append(CategoryNode(label=large, sku_count=len(g_large), children=middles))
    return tree


class ProductItem(BaseModel):
    center_id:           str
    sku_id:               str
    barcode:              str
    product_name:         str
    option_code:          str
    product_cluster:      int
    spec:                 Optional[str]
    pack_size:            Optional[float]
    KAN_대분류:            str
    KAN_중분류:            str
    KAN_소분류:            str
    forecast_available:   bool


@router.get("/products", response_model=List[ProductItem])
def get_products(
    center:          Optional[str] = Query(None, description="A / B / ALL"),
    category_large:  Optional[str] = Query(None),
    category_middle: Optional[str] = Query(None),
    category_small:  Optional[str] = Query(None),
    search:          Optional[str] = Query(None, description="상품명/바코드 부분일치"),
):
    df = _filter_by_center(load_product_master(), center)
    if category_large:
        df = df[df["KAN_대분류"] == category_large]
    if category_middle:
        df = df[df["KAN_중분류"] == category_middle]
    if category_small:
        df = df[df["KAN_소분류"] == category_small]
    if search:
        mask = (
            df["상품명"].str.contains(search, case=False, na=False)
            | df["barcode"].str.contains(search, case=False, na=False)
        )
        df = df[mask]

    forecast_keys = load_forecast_sku_keys()
    return [
        ProductItem(
            center_id=row.center_id,
            sku_id=row.sku_id,
            barcode=row.barcode,
            product_name=row.상품명,
            option_code=row.option_code,
            product_cluster=int(row.product_cluster),
            spec=None if pd.isna(row.규격) else row.규격,
            pack_size=None if pd.isna(row.입수) else float(row.입수),
            KAN_대분류=row.KAN_대분류,
            KAN_중분류=row.KAN_중분류,
            KAN_소분류=row.KAN_소분류,
            forecast_available=(row.center_id, row.sku_id) in forecast_keys,
        )
        for row in df.itertuples(index=False)
    ]


class ProductOptionItem(BaseModel):
    center_id:      str
    sku_id:          str
    option_code:     str
    product_name:    str


@router.get("/product-options", response_model=List[ProductOptionItem])
def get_product_options(
    center: str = Query(..., description="A / B"),
    sku_id: str = Query(...),
):
    """주어진 SKU와 '같은 상품의 다른 판매단위(EA/BX/CS)'로 안전하게 판단되는 형제 SKU 목록을
    반환한다. barcode/product_cluster/sku_id는 단위(option_code)마다 서로 다르므로(EA/BX/CS는
    실제로 다른 바코드를 가진 별개 SKU다), 상품명만으로는 절대 같은 상품으로 묶지 않는다 —
    상품명+규격+KAN_소분류가 모두 일치하는 SKU만 같은 상품의 다른 단위로 취급한다."""
    pm = load_product_master()
    match = pm[(pm["center_id"] == center) & (pm["sku_id"] == sku_id)]
    if match.empty:
        raise HTTPException(status_code=404, detail="해당 center_id+sku_id 상품을 찾을 수 없습니다.")
    row = match.iloc[0]

    same_spec = pm["규격"].fillna("\0__NULL__") == (row["규격"] if pd.notna(row["규격"]) else "\0__NULL__")
    siblings = pm[
        (pm["center_id"] == center)
        & (pm["상품명"] == row["상품명"])
        & (pm["KAN_소분류"] == row["KAN_소분류"])
        & same_spec
    ].sort_values("option_code")

    return [
        ProductOptionItem(
            center_id=r.center_id, sku_id=r.sku_id, option_code=r.option_code, product_name=r.상품명,
        )
        for r in siblings.itertuples(index=False)
    ]


class HistoryPoint(BaseModel):
    week_st:      date
    sales_qty:    float
    return_qty:   float
    sales_amount: float


class HorizonForecast(BaseModel):
    target_date:       date
    predicted_qty:     float
    sale_probability:  float


class ForecastResponse(BaseModel):
    center_id:            str
    sku_id:                str
    product_name:          str
    option_code:           str
    basis_week:            date
    forecast_available:    bool
    history:                List[HistoryPoint]
    h1:                    Optional[HorizonForecast]
    h2:                    Optional[HorizonForecast]
    h4:                    Optional[HorizonForecast]


def _horizon(fc_row: Optional[pd.Series], prefix: str) -> Optional[HorizonForecast]:
    if fc_row is None:
        return None
    pred = fc_row[f"{prefix}_pred"]
    if pd.isna(pred):
        return None
    return HorizonForecast(
        target_date=fc_row[f"{prefix}_target_date"].date(),
        predicted_qty=float(pred),
        sale_probability=float(fc_row[f"{prefix}_sale_probability"]),
    )


@router.get("/forecast", response_model=ForecastResponse)
def get_forecast(
    center:         str            = Query(..., description="A / B"),
    sku_id:         str            = Query(...),
    week_st:        Optional[str]  = Query(None, description="YYYY-MM-DD, 2024 기준주"),
    history_weeks:  int            = Query(12, ge=1),
):
    pm = load_product_master()
    pm_match = pm[(pm["center_id"] == center) & (pm["sku_id"] == sku_id)]
    if pm_match.empty:
        raise HTTPException(status_code=404, detail="해당 center_id+sku_id 상품을 찾을 수 없습니다.")
    pm_row = pm_match.iloc[0]

    forecast_available = (center, sku_id) in load_forecast_sku_keys()

    fc_sku = load_forecast_2024()
    fc_sku = fc_sku[(fc_sku["center_id"] == center) & (fc_sku["sku_id"] == sku_id)]

    wd_sku = load_weekly_demand()
    wd_sku = wd_sku[(wd_sku["center_id"] == center) & (wd_sku["sku_id"] == sku_id)]

    if week_st:
        try:
            basis_week = pd.Timestamp(week_st)
        except ValueError:
            raise HTTPException(status_code=400, detail="week_st 형식이 올바르지 않습니다 (YYYY-MM-DD).")
        if basis_week.weekday() != 0:
            raise HTTPException(status_code=400, detail="week_st는 월요일 기준주만 허용됩니다.")
        if not (wd_sku["week_st"] == basis_week).any():
            raise HTTPException(status_code=400, detail="해당 center_id+sku_id에 존재하지 않는 week_st입니다.")
    else:
        basis_week = load_default_basis_week()
        if not (wd_sku["week_st"] == basis_week).any():
            raise HTTPException(
                status_code=400,
                detail=f"기본 기준주({basis_week.date()})에 해당 SKU 데이터가 없습니다. week_st를 명시해 조회해주세요.",
            )

    history_df = (
        wd_sku[wd_sku["week_st"] <= basis_week]
        .sort_values("week_st", ascending=False)
        .head(history_weeks)
        .sort_values("week_st")
    )
    history = [
        HistoryPoint(
            week_st=row.week_st.date(),
            sales_qty=row.sales_qty,
            return_qty=row.return_qty,
            sales_amount=row.sales_amount,
        )
        for row in history_df.itertuples(index=False)
    ]

    fc_at_basis = fc_sku[fc_sku["week_st"] == basis_week]
    fc_row = fc_at_basis.iloc[0] if not fc_at_basis.empty else None

    return ForecastResponse(
        center_id=center,
        sku_id=sku_id,
        product_name=pm_row.상품명,
        option_code=pm_row.option_code,
        basis_week=basis_week.date(),
        forecast_available=forecast_available,
        history=history,
        h1=_horizon(fc_row, "h1"),
        h2=_horizon(fc_row, "h2"),
        h4=_horizon(fc_row, "h4"),
    )


class TrendHistoryPoint(BaseModel):
    week_st:    date
    sales_qty:  float


class TrendHorizon(BaseModel):
    target_date:    date
    predicted_qty:  float


class DemandTrendResponse(BaseModel):
    center_id:    str
    option_code:  str
    basis_week:   date
    history:      List[TrendHistoryPoint]
    h1:           Optional[TrendHorizon]
    h2:           Optional[TrendHorizon]
    h4:           Optional[TrendHorizon]


@router.get("/demand-trend", response_model=DemandTrendResponse)
def get_demand_trend(
    center:          str            = Query(..., description="A / B"),
    option_code:     str            = Query(..., description="EA / BX / CS — SKU 미선택 시 메인 차트 집계 단위"),
    category_large:  Optional[str]  = Query(None),
    category_middle: Optional[str]  = Query(None),
    category_small:  Optional[str]  = Query(None),
    week_st:         Optional[str]  = Query(None, description="YYYY-MM-DD, 월요일 기준주"),
    history_weeks:   int            = Query(12, ge=1),
):
    """메인 '수요예측 추이' 차트의 집계(카테고리/센터 범위, SKU 미선택) 모드 전용.
    weekly_demand(실제 판매수량)와 forecast_2024(h1/h2/h4)를 이미 존재하는 값 그대로
    같은 옵션 단위 SKU들에 대해 합산만 한다 — 새로운 예측을 만들지 않으며, EA/BX/CS는
    option_code로 미리 좁혀 서로 합산되지 않게 한다."""
    if week_st:
        try:
            basis_week = pd.Timestamp(week_st)
        except ValueError:
            raise HTTPException(status_code=400, detail="week_st 형식이 올바르지 않습니다 (YYYY-MM-DD).")
        if basis_week.weekday() != 0:
            raise HTTPException(status_code=400, detail="week_st는 월요일 기준주만 허용됩니다.")
    else:
        basis_week = load_default_basis_week()

    pm = load_product_master()
    pm = pm[(pm["center_id"] == center) & (pm["option_code"] == option_code)]
    if category_large:
        pm = pm[pm["KAN_대분류"] == category_large]
    if category_middle:
        pm = pm[pm["KAN_중분류"] == category_middle]
    if category_small:
        pm = pm[pm["KAN_소분류"] == category_small]
    sku_ids = set(pm["sku_id"])

    wd = load_weekly_demand()
    wd = wd[(wd["center_id"] == center) & wd["sku_id"].isin(sku_ids)]
    hist_start = basis_week - pd.Timedelta(weeks=history_weeks - 1)
    wd = wd[(wd["week_st"] >= hist_start) & (wd["week_st"] <= basis_week)]
    hist_sums = wd.groupby("week_st")["sales_qty"].sum().sort_index()
    history = [
        TrendHistoryPoint(week_st=week.date(), sales_qty=float(qty))
        for week, qty in hist_sums.items()
    ]

    fc = load_forecast_2024()
    fc = fc[(fc["center_id"] == center) & fc["sku_id"].isin(sku_ids) & (fc["week_st"] == basis_week)]

    def _agg_horizon(prefix: str) -> Optional[TrendHorizon]:
        sub = fc[fc[f"{prefix}_pred"].notna()]
        if sub.empty:
            return None
        return TrendHorizon(
            target_date=sub[f"{prefix}_target_date"].iloc[0].date(),
            predicted_qty=float(sub[f"{prefix}_pred"].sum()),
        )

    return DemandTrendResponse(
        center_id=center,
        option_code=option_code,
        basis_week=basis_week.date(),
        history=history,
        h1=_agg_horizon("h1"),
        h2=_agg_horizon("h2"),
        h4=_agg_horizon("h4"),
    )


class SparklinePoint(BaseModel):
    week_st:            date
    sales_amount:        float
    return_amount:        float
    net_sales_amount:    float


class SummaryResponse(BaseModel):
    basis_week:                    date
    total_sales_amount:            float
    return_amount:                  float
    net_sales_amount:              float
    active_sku_count:              int
    return_rate_amount:            Optional[float]
    wow_sales_amount_change_pct:   Optional[float]
    sales_qty_by_unit:              Dict[str, float]
    return_qty_by_unit:            Dict[str, float]
    sparkline:                      List[SparklinePoint]


@router.get("/summary", response_model=SummaryResponse)
def get_summary(
    center:          str            = Query("ALL", description="A / B / ALL"),
    week_st:         Optional[str]  = Query(None, description="YYYY-MM-DD, 월요일 기준주"),
    category_large:  Optional[str]  = Query(None),
    category_middle: Optional[str]  = Query(None),
    category_small:  Optional[str]  = Query(None),
    sku_id:          Optional[str]  = Query(None),
):
    pm = _filter_by_center(load_product_master(), center)
    if category_large:
        pm = pm[pm["KAN_대분류"] == category_large]
    if category_middle:
        pm = pm[pm["KAN_중분류"] == category_middle]
    if category_small:
        pm = pm[pm["KAN_소분류"] == category_small]
    if sku_id:
        pm = pm[pm["sku_id"] == sku_id]
    if pm.empty:
        raise HTTPException(status_code=404, detail="필터 조건에 해당하는 상품이 없습니다.")

    wd = load_weekly_demand().merge(
        pm[["center_id", "sku_id", "option_code"]], on=["center_id", "sku_id"], how="inner"
    )
    if wd.empty:
        raise HTTPException(status_code=404, detail="필터 조건에 해당하는 실적 데이터가 없습니다.")

    if week_st:
        try:
            basis_week = pd.Timestamp(week_st)
        except ValueError:
            raise HTTPException(status_code=400, detail="week_st 형식이 올바르지 않습니다 (YYYY-MM-DD).")
        if basis_week.weekday() != 0:
            raise HTTPException(status_code=400, detail="week_st는 월요일 기준주만 허용됩니다.")
        if not (wd["week_st"] == basis_week).any():
            raise HTTPException(status_code=400, detail="필터 조건에 해당 week_st 데이터가 없습니다.")
    else:
        basis_week = load_default_basis_week()
        if not (wd["week_st"] == basis_week).any():
            raise HTTPException(
                status_code=400,
                detail=f"기본 기준주({basis_week.date()})에 해당 필터 조건의 데이터가 없습니다. week_st를 명시해 조회해주세요.",
            )

    wd_at_basis = wd[wd["week_st"] == basis_week]
    prev_week = basis_week - pd.Timedelta(days=7)
    prev_sales_amount = wd.loc[wd["week_st"] == prev_week, "sales_amount"].sum()

    total_sales_amount = float(wd_at_basis["sales_amount"].sum())
    return_amount = float(wd_at_basis["return_amount"].sum())
    net_sales_amount = float(wd_at_basis["net_sales_amount"].sum())
    active_sku_count = wd_at_basis.loc[wd_at_basis["sales_qty"] > 0, ["center_id", "sku_id"]].drop_duplicates().shape[0]

    return_rate_amount = (return_amount / total_sales_amount) if total_sales_amount > 0 else None
    wow_sales_amount_change_pct = (
        (total_sales_amount - prev_sales_amount) / prev_sales_amount * 100
        if prev_sales_amount > 0
        else None
    )

    sales_qty_by_unit = wd_at_basis.groupby("option_code")["sales_qty"].sum().to_dict()
    return_qty_by_unit = wd_at_basis.groupby("option_code")["return_qty"].sum().to_dict()

    spark_df = (
        wd[wd["week_st"] <= basis_week]
        .groupby("week_st")[["sales_amount", "return_amount", "net_sales_amount"]]
        .sum()
        .sort_index()
        .tail(12)
        .reset_index()
    )
    sparkline = [
        SparklinePoint(
            week_st=row.week_st.date(),
            sales_amount=row.sales_amount,
            return_amount=row.return_amount,
            net_sales_amount=row.net_sales_amount,
        )
        for row in spark_df.itertuples(index=False)
    ]

    return SummaryResponse(
        basis_week=basis_week.date(),
        total_sales_amount=total_sales_amount,
        return_amount=return_amount,
        net_sales_amount=net_sales_amount,
        active_sku_count=active_sku_count,
        return_rate_amount=return_rate_amount,
        wow_sales_amount_change_pct=wow_sales_amount_change_pct,
        sales_qty_by_unit={k: float(v) for k, v in sales_qty_by_unit.items()},
        return_qty_by_unit={k: float(v) for k, v in return_qty_by_unit.items()},
        sparkline=sparkline,
    )


def _resolve_basis_week_for_sku(grid_weeks: pd.Series, week_st: Optional[str]) -> pd.Timestamp:
    """/inventory, /transactions 공용 기준주 결정 로직 (forecast/summary 코드는 건드리지 않는다)."""
    if week_st:
        try:
            basis_week = pd.Timestamp(week_st)
        except ValueError:
            raise HTTPException(status_code=400, detail="week_st 형식이 올바르지 않습니다 (YYYY-MM-DD).")
        if basis_week.weekday() != 0:
            raise HTTPException(status_code=400, detail="week_st는 월요일 기준주만 허용됩니다.")
        if not (grid_weeks == basis_week).any():
            raise HTTPException(status_code=400, detail="해당 center_id+sku_id에 존재하지 않는 week_st입니다.")
        return basis_week

    default_basis = load_default_basis_week()
    if (grid_weeks == default_basis).any():
        return default_basis
    raise HTTPException(
        status_code=400,
        detail=f"기본 기준주({default_basis.date()})에 해당 SKU 데이터가 없습니다. week_st를 명시해 조회해주세요.",
    )


class CurrentWeekTransaction(BaseModel):
    inbound_qty:      float
    outbound_qty:     float
    sales_qty:        float
    return_qty:       float
    inbound_amount:   float
    outbound_amount:  float
    sales_amount:     float
    return_amount:    float


class InventoryHistoryPoint(BaseModel):
    week_st:              date
    inbound_qty:           float
    outbound_qty:          float
    sales_qty:             float
    return_qty:            float
    estimated_inventory:   Optional[float]


class InventoryResponse(BaseModel):
    center_id:                     str
    sku_id:                         str
    product_name:                   str
    option_code:                     str
    basis_week:                      date
    return_policy:                   str
    inventory_available:             bool
    estimated_inventory:             Optional[float]
    estimated_opening_inventory:     Optional[float]
    current_week:                     CurrentWeekTransaction
    history:                          List[InventoryHistoryPoint]


@router.get("/inventory", response_model=InventoryResponse)
def get_inventory(
    center:         str            = Query(..., description="A / B"),
    sku_id:         str            = Query(...),
    week_st:        Optional[str]  = Query(None, description="YYYY-MM-DD, 월요일 기준주"),
    return_policy:  str            = Query("excluded", description="excluded / included"),
    history_weeks:  int            = Query(12, ge=1),
):
    if return_policy not in ("excluded", "included"):
        raise HTTPException(status_code=400, detail="return_policy는 excluded 또는 included여야 합니다.")

    pm = load_product_master()
    pm_match = pm[(pm["center_id"] == center) & (pm["sku_id"] == sku_id)]
    if pm_match.empty:
        raise HTTPException(status_code=404, detail="해당 center_id+sku_id 상품을 찾을 수 없습니다.")
    pm_row = pm_match.iloc[0]

    tx_sku = load_weekly_transactions()
    tx_sku = tx_sku[(tx_sku["center_id"] == center) & (tx_sku["sku_id"] == sku_id)]
    inv_sku = load_inventory_weekly()
    inv_sku = inv_sku[(inv_sku["center_id"] == center) & (inv_sku["sku_id"] == sku_id)]
    if tx_sku.empty or inv_sku.empty:
        raise HTTPException(status_code=404, detail="해당 SKU의 실적/재고 데이터가 없습니다.")

    basis_week = _resolve_basis_week_for_sku(tx_sku["week_st"], week_st)

    tx_row = tx_sku[tx_sku["week_st"] == basis_week].iloc[0]
    inv_row = inv_sku[inv_sku["week_st"] == basis_week].iloc[0]

    inventory_available = bool(inv_row["inventory_available"])
    inv_col = (
        "estimated_inventory_return_excluded"
        if return_policy == "excluded"
        else "estimated_inventory_return_included"
    )
    estimated_inventory = None if not inventory_available else float(inv_row[inv_col])
    estimated_opening_inventory = (
        None if not inventory_available else float(inv_row["estimated_opening_inventory"])
    )

    current_week = CurrentWeekTransaction(
        inbound_qty=float(tx_row["inbound_qty"]),
        outbound_qty=float(tx_row["outbound_qty"]),
        sales_qty=float(tx_row["sales_qty"]),
        return_qty=float(tx_row["return_qty"]),
        inbound_amount=float(tx_row["inbound_amount"]),
        outbound_amount=float(tx_row["outbound_amount"]),
        sales_amount=float(tx_row["sales_amount"]),
        return_amount=float(tx_row["return_amount"]),
    )

    hist_tx = (
        tx_sku[tx_sku["week_st"] <= basis_week]
        .sort_values("week_st", ascending=False)
        .head(history_weeks)
        .sort_values("week_st")
    )
    hist_inv = inv_sku[
        ["week_st", "estimated_inventory_return_excluded", "estimated_inventory_return_included", "inventory_available"]
    ]
    hist = hist_tx.merge(hist_inv, on="week_st", how="left")

    history = [
        InventoryHistoryPoint(
            week_st=row.week_st.date(),
            inbound_qty=row.inbound_qty,
            outbound_qty=row.outbound_qty,
            sales_qty=row.sales_qty,
            return_qty=row.return_qty,
            estimated_inventory=(
                None if not bool(row.inventory_available) else float(getattr(row, inv_col))
            ),
        )
        for row in hist.itertuples(index=False)
    ]

    return InventoryResponse(
        center_id=center,
        sku_id=sku_id,
        product_name=pm_row.상품명,
        option_code=pm_row.option_code,
        basis_week=basis_week.date(),
        return_policy=return_policy,
        inventory_available=inventory_available,
        estimated_inventory=estimated_inventory,
        estimated_opening_inventory=estimated_opening_inventory,
        current_week=current_week,
        history=history,
    )


class TransactionHistoryPoint(BaseModel):
    week_st:                date
    inbound_qty:             float
    outbound_qty:            float
    sales_qty:               float
    return_qty:              float
    inbound_amount:          float
    outbound_amount:         float
    inbound_total_amount:    float
    outbound_total_amount:   float
    sales_amount:            float
    return_amount:           float


class TransactionsResponse(BaseModel):
    center_id:     str
    sku_id:         str
    product_name:   str
    option_code:    str
    basis_week:     date
    history:        List[TransactionHistoryPoint]


@router.get("/transactions", response_model=TransactionsResponse)
def get_transactions(
    center:         str            = Query(..., description="A / B"),
    sku_id:         str            = Query(...),
    week_st:        Optional[str]  = Query(None, description="YYYY-MM-DD, 월요일 기준주"),
    history_weeks:  int            = Query(12, ge=1),
):
    pm = load_product_master()
    pm_match = pm[(pm["center_id"] == center) & (pm["sku_id"] == sku_id)]
    if pm_match.empty:
        raise HTTPException(status_code=404, detail="해당 center_id+sku_id 상품을 찾을 수 없습니다.")
    pm_row = pm_match.iloc[0]

    tx_sku = load_weekly_transactions()
    tx_sku = tx_sku[(tx_sku["center_id"] == center) & (tx_sku["sku_id"] == sku_id)]
    if tx_sku.empty:
        raise HTTPException(status_code=404, detail="해당 SKU의 실적 데이터가 없습니다.")

    basis_week = _resolve_basis_week_for_sku(tx_sku["week_st"], week_st)

    hist = (
        tx_sku[tx_sku["week_st"] <= basis_week]
        .sort_values("week_st", ascending=False)
        .head(history_weeks)
        .sort_values("week_st")
    )
    history = [
        TransactionHistoryPoint(
            week_st=row.week_st.date(),
            inbound_qty=row.inbound_qty,
            outbound_qty=row.outbound_qty,
            sales_qty=row.sales_qty,
            return_qty=row.return_qty,
            inbound_amount=row.inbound_amount,
            outbound_amount=row.outbound_amount,
            inbound_total_amount=row.inbound_total_amount,
            outbound_total_amount=row.outbound_total_amount,
            sales_amount=row.sales_amount,
            return_amount=row.return_amount,
        )
        for row in hist.itertuples(index=False)
    ]

    return TransactionsResponse(
        center_id=center,
        sku_id=sku_id,
        product_name=pm_row.상품명,
        option_code=pm_row.option_code,
        basis_week=basis_week.date(),
        history=history,
    )


class ForecastProductItem(BaseModel):
    center_id:          str
    sku_id:               str
    product_name:         str
    barcode:              str
    option_code:          str
    recent_sales_qty:     Optional[float]
    h1_pred:               Optional[float]
    h2_pred:               Optional[float]
    h4_pred:               Optional[float]


@router.get("/forecast-products", response_model=List[ForecastProductItem])
def get_forecast_products(
    center:          str            = Query(..., description="A / B"),
    week_st:         Optional[str]  = Query(None, description="YYYY-MM-DD, 월요일 기준주"),
    category_large:  Optional[str]  = Query(None),
    category_middle: Optional[str]  = Query(None),
    category_small:  Optional[str]  = Query(None),
    option_code:     Optional[str]  = Query(None, description="EA/BX/CS 등 — 지정 시 해당 단위 SKU만 반환(수량 ranking 단위 혼합 방지). 생략 시 기존과 동일."),
    top:             Optional[int]  = Query(None, ge=1),
):
    """해당 center+category 범위 안에서, 지정한 basis_week 기준 h1/h2/h4 중
    최소 하나라도 predicted_qty가 존재하는 SKU만 h1_pred 내림차순으로 반환한다.
    SKU별 개별 /forecast 호출 없이 한 번의 join으로 TOP N과 '현재 주 예측 가능 여부'를
    동시에 제공하기 위한 endpoint다."""
    if week_st:
        try:
            basis_week = pd.Timestamp(week_st)
        except ValueError:
            raise HTTPException(status_code=400, detail="week_st 형식이 올바르지 않습니다 (YYYY-MM-DD).")
        if basis_week.weekday() != 0:
            raise HTTPException(status_code=400, detail="week_st는 월요일 기준주만 허용됩니다.")
    else:
        basis_week = load_default_basis_week()

    pm = load_product_master()
    pm = pm[pm["center_id"] == center]
    if category_large:
        pm = pm[pm["KAN_대분류"] == category_large]
    if category_middle:
        pm = pm[pm["KAN_중분류"] == category_middle]
    if category_small:
        pm = pm[pm["KAN_소분류"] == category_small]
    if option_code:
        pm = pm[pm["option_code"] == option_code]

    fc = load_forecast_2024()
    fc = fc[(fc["center_id"] == center) & (fc["week_st"] == basis_week)]
    fc = fc[fc["h1_pred"].notna() | fc["h2_pred"].notna() | fc["h4_pred"].notna()]

    merged = fc.merge(
        pm[["center_id", "sku_id", "상품명", "barcode", "option_code"]], on=["center_id", "sku_id"], how="inner"
    )

    wd_at_basis = load_weekly_demand()
    wd_at_basis = wd_at_basis[
        (wd_at_basis["center_id"] == center) & (wd_at_basis["week_st"] == basis_week)
    ][["center_id", "sku_id", "sales_qty"]]
    merged = merged.merge(wd_at_basis, on=["center_id", "sku_id"], how="left")

    merged = merged.sort_values("h1_pred", ascending=False, na_position="last")
    if top:
        merged = merged.head(top)

    return [
        ForecastProductItem(
            center_id=row.center_id,
            sku_id=row.sku_id,
            product_name=row.상품명,
            barcode=row.barcode,
            option_code=row.option_code,
            recent_sales_qty=None if pd.isna(row.sales_qty) else float(row.sales_qty),
            h1_pred=None if pd.isna(row.h1_pred) else float(row.h1_pred),
            h2_pred=None if pd.isna(row.h2_pred) else float(row.h2_pred),
            h4_pred=None if pd.isna(row.h4_pred) else float(row.h4_pred),
        )
        for row in merged.itertuples(index=False)
    ]


def _resolve_insight_basis_week(week_st: Optional[str]) -> pd.Timestamp:
    """/insights/* 공용 기준주 결정. 카드/모달은 특정 SKU 단위가 아니라 집계이므로
    grid 존재 여부를 강제하지 않는다 — 데이터가 없는 주는 빈 ranking으로 자연스럽게 나타난다."""
    if not week_st:
        return load_default_basis_week()
    try:
        basis_week = pd.Timestamp(week_st)
    except ValueError:
        raise HTTPException(status_code=400, detail="week_st 형식이 올바르지 않습니다 (YYYY-MM-DD).")
    if basis_week.weekday() != 0:
        raise HTTPException(status_code=400, detail="week_st는 월요일 기준주만 허용됩니다.")
    return basis_week


# ── 1) 카테고리별 매출 TOP5 ──────────────────────────────────────
class CategorySalesItem(BaseModel):
    category:      str
    sales_amount:  float
    share:         float


class CategorySalesResponse(BaseModel):
    center_id:            str
    basis_week:            date
    total_sales_amount:    float
    top5:                   List[CategorySalesItem]
    ranking:                List[CategorySalesItem]


@router.get("/insights/category-sales", response_model=CategorySalesResponse)
def get_insights_category_sales(
    center:          str            = Query(..., description="A / B"),
    week_st:         Optional[str]  = Query(None, description="YYYY-MM-DD, 월요일 기준주"),
    category_large:  Optional[str]  = Query(None),
    category_middle: Optional[str]  = Query(None),
    category_small:  Optional[str]  = Query(None),
):
    basis_week = _resolve_insight_basis_week(week_st)

    pm = load_product_master()
    pm = pm[pm["center_id"] == center]
    if category_large:
        pm = pm[pm["KAN_대분류"] == category_large]
    if category_middle:
        pm = pm[pm["KAN_중분류"] == category_middle]
    if category_small:
        pm = pm[pm["KAN_소분류"] == category_small]

    # 선택된 카테고리 한 단계 아래 레벨로 자동 드릴다운(예: 대분류 선택 시 중분류별로 breakdown).
    # 소분류까지 선택된 경우 더 내려갈 레벨이 없어 소분류 기준(사실상 1행)을 유지한다.
    group_col = "KAN_대분류"
    if category_middle:
        group_col = "KAN_소분류"
    elif category_large:
        group_col = "KAN_중분류"

    wd = load_weekly_demand()
    wd = wd[(wd["center_id"] == center) & (wd["week_st"] == basis_week)]

    merged = wd.merge(pm[["center_id", "sku_id", group_col]], on=["center_id", "sku_id"], how="inner")
    agg = merged.groupby(group_col)["sales_amount"].sum().sort_values(ascending=False)
    total = float(agg.sum())

    ranking = [
        CategorySalesItem(
            category=cat,
            sales_amount=float(v),
            share=(float(v) / total) if total > 0 else 0.0,
        )
        for cat, v in agg.items()
    ]

    return CategorySalesResponse(
        center_id=center,
        basis_week=basis_week.date(),
        total_sales_amount=total,
        top5=ranking[:5],
        ranking=ranking,
    )


# ── 2) 판매지역별 매출 TOP5 ──────────────────────────────────────
class RegionSalesItem(BaseModel):
    sido:           str
    sigungu:        str
    sales_amount:   float
    share:          float


class RegionProductItem(BaseModel):
    sku_id:         str
    product_name:   str
    option_code:    str
    sales_amount:   float


class RegionSalesResponse(BaseModel):
    center_id:             str
    basis_week:              date
    total_sales_amount:      float
    top5:                     List[RegionSalesItem]
    ranking:                  List[RegionSalesItem]
    top_products:             Optional[List[RegionProductItem]] = None


@router.get("/insights/region-sales", response_model=RegionSalesResponse)
def get_insights_region_sales(
    center:          str            = Query(..., description="A / B"),
    week_st:         Optional[str]  = Query(None, description="YYYY-MM-DD, 월요일 기준주"),
    sido:            Optional[str]  = Query(None, description="지정 시 해당 시도+시군구의 상위 상품도 반환"),
    sigungu:         Optional[str]  = Query(None, description="sido와 함께 지정해야 top_products가 채워진다"),
    category_large:  Optional[str]  = Query(None),
    category_middle: Optional[str]  = Query(None),
    category_small:  Optional[str]  = Query(None),
):
    basis_week = _resolve_insight_basis_week(week_st)

    pm = load_product_master()
    pm_center = pm[pm["center_id"] == center]
    if category_large:
        pm_center = pm_center[pm_center["KAN_대분류"] == category_large]
    if category_middle:
        pm_center = pm_center[pm_center["KAN_중분류"] == category_middle]
    if category_small:
        pm_center = pm_center[pm_center["KAN_소분류"] == category_small]
    valid_sku_keys = set(zip(pm_center["center_id"], pm_center["sku_id"]))

    tx = load_sales_transactions_raw()
    tx = tx[(tx["center_id"] == center) & (tx["week_st"] == basis_week) & (tx["수량"] > 0)]
    # tx가 이미 0행이면 파이썬 list(빈 리스트)로 인덱싱할 때 pandas가 이를 "컬럼 목록"으로
    # 오인해 컬럼까지 통째로 사라진다 — 인덱스가 명시된 Series로 만들어 행 boolean mask임을
    # 명확히 한다(빈 주에도 KeyError 없이 정상적으로 빈 결과가 나오게 하기 위한 안정성 수정).
    matched = pd.Series(
        [k in valid_sku_keys for k in zip(tx["center_id"], tx["sku_id"])], index=tx.index, dtype=bool,
    )
    tx = tx[matched]

    total = float(tx["금액"].sum())

    region_agg = (
        tx.groupby(["시도", "시군구"], as_index=False)["금액"]
        .sum()
        .rename(columns={"금액": "sales_amount"})
        .sort_values("sales_amount", ascending=False)
    )

    ranking = [
        RegionSalesItem(
            sido=row.시도,
            sigungu=row.시군구,
            sales_amount=float(row.sales_amount),
            share=(float(row.sales_amount) / total) if total > 0 else 0.0,
        )
        for row in region_agg.itertuples(index=False)
    ]

    top_products = None
    if sido and sigungu:
        region_tx = tx[(tx["시도"] == sido) & (tx["시군구"] == sigungu)]
        prod_agg = (
            region_tx.groupby(["sku_id", "상품명", "옵션코드"], as_index=False)["금액"]
            .sum()
            .rename(columns={"금액": "sales_amount"})
            .sort_values("sales_amount", ascending=False)
            .head(10)
        )
        top_products = [
            RegionProductItem(
                sku_id=row.sku_id,
                product_name=row.상품명,
                option_code=row.옵션코드,
                sales_amount=float(row.sales_amount),
            )
            for row in prod_agg.itertuples(index=False)
        ]

    return RegionSalesResponse(
        center_id=center,
        basis_week=basis_week.date(),
        total_sales_amount=total,
        top5=ranking[:5],
        ranking=ranking,
        top_products=top_products,
    )


# ── 3) 1주 예상수요 대비 재고 부족 ────────────────────────────────
class ShortageItem(BaseModel):
    center_id:              str
    sku_id:                  str
    product_name:            str
    barcode:                 str
    option_code:             str
    estimated_inventory:     float
    h1_pred:                  float
    shortage_qty:             float
    fulfillment_rate:         float


class InventoryShortageResponse(BaseModel):
    center_id:              str
    basis_week:               date
    description:              str
    eligible_sku_count:       int
    shortage_sku_count:       int
    shortage_rate:            float
    top5:                      List[ShortageItem]
    ranking:                   List[ShortageItem]


@router.get("/insights/inventory-shortage", response_model=InventoryShortageResponse)
def get_insights_inventory_shortage(
    center:          str            = Query(..., description="A / B"),
    week_st:         Optional[str]  = Query(None, description="YYYY-MM-DD, 월요일 기준주"),
    category_large:  Optional[str]  = Query(None),
    category_middle: Optional[str]  = Query(None),
    category_small:  Optional[str]  = Query(None),
    option_code:     Optional[str]  = Query(None, description="EA/BX/CS 등 — 지정 시 해당 단위 SKU만 반환(수량 ranking 단위 혼합 방지). 생략 시 기존과 동일."),
):
    basis_week = _resolve_insight_basis_week(week_st)

    pm = load_product_master()
    pm_scope = pm[pm["center_id"] == center]
    if category_large:
        pm_scope = pm_scope[pm_scope["KAN_대분류"] == category_large]
    if category_middle:
        pm_scope = pm_scope[pm_scope["KAN_중분류"] == category_middle]
    if category_small:
        pm_scope = pm_scope[pm_scope["KAN_소분류"] == category_small]
    if option_code:
        pm_scope = pm_scope[pm_scope["option_code"] == option_code]

    inv = load_inventory_weekly()
    inv = inv[(inv["center_id"] == center) & (inv["week_st"] == basis_week) & (inv["inventory_available"])]
    inv = inv.merge(pm_scope[["center_id", "sku_id"]], on=["center_id", "sku_id"], how="inner")

    fc = load_forecast_2024()
    fc = fc[(fc["center_id"] == center) & (fc["week_st"] == basis_week) & (fc["h1_pred"].notna())]

    merged = inv.merge(fc[["center_id", "sku_id", "h1_pred"]], on=["center_id", "sku_id"], how="inner")
    merged = merged.merge(pm_scope[["center_id", "sku_id", "상품명", "barcode"]], on=["center_id", "sku_id"], how="left")

    eligible_count = len(merged)

    merged["shortage_qty"] = (merged["h1_pred"] - merged["estimated_inventory_return_excluded"]).clip(lower=0)
    shortage = merged[merged["estimated_inventory_return_excluded"] < merged["h1_pred"]].copy()
    shortage["fulfillment_rate"] = shortage["estimated_inventory_return_excluded"] / shortage["h1_pred"]
    shortage = shortage.sort_values("shortage_qty", ascending=False)

    shortage_count = len(shortage)

    ranking = [
        ShortageItem(
            center_id=row.center_id,
            sku_id=row.sku_id,
            product_name=row.상품명,
            barcode=row.barcode,
            option_code=row.option_code,
            estimated_inventory=float(row.estimated_inventory_return_excluded),
            h1_pred=float(row.h1_pred),
            shortage_qty=float(row.shortage_qty),
            fulfillment_rate=float(row.fulfillment_rate),
        )
        for row in shortage.itertuples(index=False)
    ]

    return InventoryShortageResponse(
        center_id=center,
        basis_week=basis_week.date(),
        description=(
            "추정재고(estimated_inventory_return_excluded, 입출고 이력 기반 추정치) 대비 "
            "1주 후(h1) 예상수요 부족분입니다. 실사재고가 아닙니다."
        ),
        eligible_sku_count=eligible_count,
        shortage_sku_count=shortage_count,
        shortage_rate=(shortage_count / eligible_count) if eligible_count > 0 else 0.0,
        top5=ranking[:5],
        ranking=ranking,
    )


# ── 4) 판매 증가 TOP5 ────────────────────────────────────────────
class SalesIncreaseItem(BaseModel):
    center_id:           str
    sku_id:                str
    product_name:          str
    barcode:               str
    option_code:           str
    prev_sales_qty:        float
    current_sales_qty:     float
    increase_qty:          float
    increase_pct:          Optional[float]


class SalesIncreaseResponse(BaseModel):
    center_id:              str
    basis_week:               date
    prev_week:                date
    increased_sku_count:      int
    top5:                      List[SalesIncreaseItem]
    ranking:                   List[SalesIncreaseItem]


@router.get("/insights/sales-surge", response_model=SalesIncreaseResponse)
def get_insights_sales_surge(
    center:          str            = Query(..., description="A / B"),
    week_st:         Optional[str]  = Query(None, description="YYYY-MM-DD, 월요일 기준주"),
    category_large:  Optional[str]  = Query(None),
    category_middle: Optional[str]  = Query(None),
    category_small:  Optional[str]  = Query(None),
):
    basis_week = _resolve_insight_basis_week(week_st)
    prev_week = basis_week - pd.Timedelta(days=7)

    wd = load_weekly_demand()
    wd_center = wd[wd["center_id"] == center]

    cur = wd_center[wd_center["week_st"] == basis_week][["sku_id", "sales_qty"]].rename(
        columns={"sales_qty": "current_sales_qty"}
    )
    prev = wd_center[wd_center["week_st"] == prev_week][["sku_id", "sales_qty"]].rename(
        columns={"sales_qty": "prev_sales_qty"}
    )
    merged = cur.merge(prev, on="sku_id", how="left")
    merged["prev_sales_qty"] = merged["prev_sales_qty"].fillna(0.0)
    merged["increase_qty"] = merged["current_sales_qty"] - merged["prev_sales_qty"]
    merged = merged[merged["increase_qty"] > 0].copy()
    merged["increase_pct"] = merged.apply(
        lambda r: (r["increase_qty"] / r["prev_sales_qty"] * 100) if r["prev_sales_qty"] > 0 else None,
        axis=1,
    )
    merged = merged.sort_values("increase_qty", ascending=False)

    pm = load_product_master()
    pm_center = pm[pm["center_id"] == center]
    if category_large:
        pm_center = pm_center[pm_center["KAN_대분류"] == category_large]
    if category_middle:
        pm_center = pm_center[pm_center["KAN_중분류"] == category_middle]
    if category_small:
        pm_center = pm_center[pm_center["KAN_소분류"] == category_small]
    merged = merged.merge(pm_center[["sku_id", "상품명", "barcode", "option_code"]], on="sku_id", how="inner")
    merged = merged.sort_values("increase_qty", ascending=False)

    ranking = [
        SalesIncreaseItem(
            center_id=center,
            sku_id=row.sku_id,
            product_name=row.상품명,
            barcode=row.barcode,
            option_code=row.option_code,
            prev_sales_qty=float(row.prev_sales_qty),
            current_sales_qty=float(row.current_sales_qty),
            increase_qty=float(row.increase_qty),
            increase_pct=None if pd.isna(row.increase_pct) else float(row.increase_pct),
        )
        for row in merged.itertuples(index=False)
    ]

    return SalesIncreaseResponse(
        center_id=center,
        basis_week=basis_week.date(),
        prev_week=prev_week.date(),
        increased_sku_count=len(ranking),
        top5=ranking[:5],
        ranking=ranking,
    )


# ── 5) 반품 TOP5(금액 기준) ──────────────────────────────────────
class ReturnItem(BaseModel):
    center_id:      str
    sku_id:           str
    product_name:     str
    barcode:          str
    option_code:      str
    return_qty:       float
    return_amount:    float


class ReturnsResponse(BaseModel):
    center_id:      str
    basis_week:       date
    top5:              List[ReturnItem]
    ranking:           List[ReturnItem]


@router.get("/insights/returns", response_model=ReturnsResponse)
def get_insights_returns(
    center:          str            = Query(..., description="A / B"),
    week_st:         Optional[str]  = Query(None, description="YYYY-MM-DD, 월요일 기준주"),
    category_large:  Optional[str]  = Query(None),
    category_middle: Optional[str]  = Query(None),
    category_small:  Optional[str]  = Query(None),
):
    basis_week = _resolve_insight_basis_week(week_st)

    wt = load_weekly_transactions()
    wt = wt[(wt["center_id"] == center) & (wt["week_st"] == basis_week) & (wt["return_amount"] > 0)]

    pm = load_product_master()
    pm_center = pm[pm["center_id"] == center]
    if category_large:
        pm_center = pm_center[pm_center["KAN_대분류"] == category_large]
    if category_middle:
        pm_center = pm_center[pm_center["KAN_중분류"] == category_middle]
    if category_small:
        pm_center = pm_center[pm_center["KAN_소분류"] == category_small]
    wt = wt.merge(pm_center[["sku_id", "상품명", "barcode"]], on="sku_id", how="inner")
    wt = wt.sort_values("return_amount", ascending=False)

    ranking = [
        ReturnItem(
            center_id=center,
            sku_id=row.sku_id,
            product_name=row.상품명,
            barcode=row.barcode,
            option_code=row.option_code,
            return_qty=float(row.return_qty),
            return_amount=float(row.return_amount),
        )
        for row in wt.itertuples(index=False)
    ]

    return ReturnsResponse(
        center_id=center,
        basis_week=basis_week.date(),
        top5=ranking[:5],
        ranking=ranking,
    )


# ══════════════════════════════════════════════════════════════════
# 일간 운영 데이터(daily_demand/daily_transactions/inventory_daily) API
# 운영실적=일간, AI예측(h1/h2/h4, forecast_2024, sale_probability)은 여기서 다루지 않으며
# 위 weekly 엔드포인트들을 그대로 유지한다(주간).
# ══════════════════════════════════════════════════════════════════


def _resolve_daily_basis_date(date_str: Optional[str]) -> pd.Timestamp:
    """/daily/category-sales, /daily/region-sales, /daily/returns, /daily/sales-surge 공용 조회일 결정.
    insights 엔드포인트와 동일하게 SKU/grid 존재 여부는 강제하지 않는다 — 데이터가 없는 날짜는
    빈 ranking으로 자연스럽게 나타난다. 다만 보유 데이터 범위를 넘는 미래 조회일은 거부한다
    ("실시간"이 아니라 실제 보유 데이터 범위까지만 조회 가능하다)."""
    default_date = load_default_daily_date()
    if not date_str:
        return default_date
    try:
        basis_date = pd.Timestamp(date_str)
    except ValueError:
        raise HTTPException(status_code=400, detail="date 형식이 올바르지 않습니다 (YYYY-MM-DD).")
    if basis_date > default_date:
        raise HTTPException(
            status_code=400,
            detail=f"조회 가능한 최신 일자는 {default_date.date()}입니다 (보유 데이터 범위를 초과했습니다).",
        )
    return basis_date


def _resolve_daily_date_for_sku(grid_dates: pd.Series, date_str: Optional[str]) -> pd.Timestamp:
    """/daily/transactions, /daily/inventory 공용 조회일 결정(_resolve_basis_week_for_sku의 일간 버전).
    해당 SKU의 grid에 실제로 존재하는 date만 허용하고, 보유 데이터 범위를 넘는 미래 조회일은 거부한다."""
    default_date = load_default_daily_date()
    if date_str:
        try:
            basis_date = pd.Timestamp(date_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="date 형식이 올바르지 않습니다 (YYYY-MM-DD).")
        if basis_date > default_date:
            raise HTTPException(
                status_code=400,
                detail=f"조회 가능한 최신 일자는 {default_date.date()}입니다 (보유 데이터 범위를 초과했습니다).",
            )
        if not (grid_dates == basis_date).any():
            raise HTTPException(status_code=400, detail="해당 center_id+sku_id에 존재하지 않는 date입니다.")
        return basis_date

    if (grid_dates == default_date).any():
        return default_date
    raise HTTPException(
        status_code=400,
        detail=f"기본 조회일({default_date.date()})에 해당 SKU 데이터가 없습니다. date를 명시해 조회해주세요.",
    )


def _resolve_daily_date_for_sku_grid_start(grid_start: pd.Timestamp, date_str: Optional[str]) -> pd.Timestamp:
    """/daily/transactions 전용 — daily_transactions_sparse는 활동 없는 날이 빠져 있어
    _resolve_daily_date_for_sku처럼 per-row 존재 여부로 grid를 판정할 수 없다. 대신
    grid_start<=date<=default_date(=DAILY_GRID_END) 구간 판정으로 동일한 결과를 낸다
    (canonical dense grid는 이 구간에 대해 항상 dense하므로 동치)."""
    default_date = load_default_daily_date()
    if date_str:
        try:
            basis_date = pd.Timestamp(date_str)
        except ValueError:
            raise HTTPException(status_code=400, detail="date 형식이 올바르지 않습니다 (YYYY-MM-DD).")
        if basis_date > default_date:
            raise HTTPException(
                status_code=400,
                detail=f"조회 가능한 최신 일자는 {default_date.date()}입니다 (보유 데이터 범위를 초과했습니다).",
            )
        if basis_date < grid_start:
            raise HTTPException(status_code=400, detail="해당 center_id+sku_id에 존재하지 않는 date입니다.")
        return basis_date

    if grid_start <= default_date:
        return default_date
    raise HTTPException(
        status_code=400,
        detail=f"기본 조회일({default_date.date()})에 해당 SKU 데이터가 없습니다. date를 명시해 조회해주세요.",
    )


# ── 일간 요약 ─────────────────────────────────────────────────────
class DailySparklinePoint(BaseModel):
    date:               date
    sales_amount:        float
    return_amount:        float
    net_sales_amount:    float


class DailySummaryResponse(BaseModel):
    date:                           date
    forecast_basis_week:            Optional[date]
    total_sales_amount:              float
    return_amount:                    float
    net_sales_amount:                float
    active_sku_count:                int
    sales_record_count:              int
    return_rate_amount:              Optional[float]
    sales_qty_by_unit:                Dict[str, float]
    sparkline:                          List[DailySparklinePoint]


@router.get("/daily/summary", response_model=DailySummaryResponse)
def get_daily_summary(
    center:          str            = Query("ALL", description="A / B / ALL"),
    date_:           Optional[str]  = Query(None, alias="date", description="YYYY-MM-DD, 조회일"),
    category_large:  Optional[str]  = Query(None),
    category_middle: Optional[str]  = Query(None),
    category_small:  Optional[str]  = Query(None),
    sku_id:          Optional[str]  = Query(None),
):
    pm = _filter_by_center(load_product_master(), center)
    if category_large:
        pm = pm[pm["KAN_대분류"] == category_large]
    if category_middle:
        pm = pm[pm["KAN_중분류"] == category_middle]
    if category_small:
        pm = pm[pm["KAN_소분류"] == category_small]
    if sku_id:
        pm = pm[pm["sku_id"] == sku_id]
    if pm.empty:
        raise HTTPException(status_code=404, detail="필터 조건에 해당하는 상품이 없습니다.")

    default_date = load_default_daily_date()
    if date_:
        try:
            basis_date = pd.Timestamp(date_)
        except ValueError:
            raise HTTPException(status_code=400, detail="date 형식이 올바르지 않습니다 (YYYY-MM-DD).")
        if basis_date > default_date:
            raise HTTPException(
                status_code=400,
                detail=f"조회 가능한 최신 일자는 {default_date.date()}입니다 (보유 데이터 범위를 초과했습니다).",
            )
    else:
        basis_date = default_date

    spark_start = basis_date - pd.Timedelta(days=29)

    if sku_id:
        # sku_id 필터는 frontend가 쓰지 않는 드문 경로라 canonical 31M을 그대로 쓴다
        # (단일 SKU만 남기 때문에 실제 스캔 비용은 미미하다) — API contract는 완전히 동일하다.
        dd_raw = _filter_by_center(load_daily_demand(), center)
        dd = dd_raw[(dd_raw["date"] >= spark_start) & (dd_raw["date"] <= basis_date)].merge(
            pm[["center_id", "sku_id", "option_code"]], on=["center_id", "sku_id"], how="inner"
        )
        if dd.empty:
            raise HTTPException(status_code=404, detail="필터 조건에 해당하는 실적 데이터가 없습니다.")
        if not (dd["date"] == basis_date).any():
            if date_:
                raise HTTPException(status_code=400, detail="필터 조건에 해당 date 데이터가 없습니다.")
            raise HTTPException(
                status_code=400,
                detail=f"기본 조회일({basis_date.date()})에 해당 필터 조건의 데이터가 없습니다. date를 명시해 조회해주세요.",
            )

        dd_at_date = dd[dd["date"] == basis_date]
        active_sku_count = dd_at_date.loc[dd_at_date["sales_qty"] > 0, ["center_id", "sku_id"]].drop_duplicates().shape[0]
        sales_qty_by_unit = dd_at_date.groupby("option_code")["sales_qty"].sum().to_dict()
        spark_df = (
            dd.groupby("date")[["sales_amount", "return_amount", "net_sales_amount"]]
            .sum().sort_index().reset_index()
        )
    else:
        # serving derivative(daily_summary_agg, canonical 31M을 미리 (center,date,KAN 3단,
        # option_code) 단위로 합산해 활동 없는 조합을 제거한 sparse 집계) 경로 — 실제 frontend가
        # 쓰는 경로다. sum은 결합법칙이 성립하므로 미리 합산해도 최종 합계는 canonical과 동일하다.
        grid = load_daily_sku_grid()
        grid_scope = _filter_by_center(grid, center)
        if category_large:
            grid_scope = grid_scope[grid_scope["KAN_대분류"] == category_large]
        if category_middle:
            grid_scope = grid_scope[grid_scope["KAN_중분류"] == category_middle]
        if category_small:
            grid_scope = grid_scope[grid_scope["KAN_소분류"] == category_small]

        if not (grid_scope["grid_start"] <= basis_date).any():
            if date_:
                raise HTTPException(status_code=400, detail="필터 조건에 해당 date 데이터가 없습니다.")
            raise HTTPException(
                status_code=400,
                detail=f"기본 조회일({basis_date.date()})에 해당 필터 조건의 데이터가 없습니다. date를 명시해 조회해주세요.",
            )

        agg = _filter_by_center(load_daily_summary_agg(), center)
        if category_large:
            agg = agg[agg["KAN_대분류"] == category_large]
        if category_middle:
            agg = agg[agg["KAN_중분류"] == category_middle]
        if category_small:
            agg = agg[agg["KAN_소분류"] == category_small]

        dd = agg[(agg["date"] >= spark_start) & (agg["date"] <= basis_date)]
        dd_at_date = dd[dd["date"] == basis_date]
        # active_sku_count/총액류는 순수 합산이라 활동 없는(0) 조합을 미리 제거해도 값이 그대로다.
        active_sku_count = int(dd_at_date["active_sku_count"].sum())

        # sales_qty_by_unit는 "그 날 존재했지만 값이 0"인 unit도 canonical(dense grid)에서는
        # 명시적으로 나타난다 — sparse 집계에는 그런 0행이 아예 없으므로, grid_scope에서
        # basis_date 시점에 이미 존재했던 unit 목록을 구해 0으로 채워 넣어 동일하게 재현한다.
        existing_units = grid_scope.loc[grid_scope["grid_start"] <= basis_date, "option_code"].unique()
        unit_sums = dd_at_date.groupby("option_code")["sales_qty"].sum()
        sales_qty_by_unit = {u: float(unit_sums.get(u, 0.0)) for u in existing_units}

        # sparkline도 마찬가지 이유로, scope 전체에서 가장 이른 grid_start ~ basis_date 구간을
        # 0으로 채워 canonical dense 결과와 동일한 날짜 개수/순서를 재현한다(더 짧은 grid라면
        # canonical도 원래 그만큼만 나왔으므로 그 경계도 그대로 따른다).
        range_start = max(spark_start, grid_scope["grid_start"].min())
        full_range = pd.date_range(range_start, basis_date, freq="D")
        spark_df = (
            dd.groupby("date")[["sales_amount", "return_amount", "net_sales_amount"]]
            .sum()
            .reindex(full_range, fill_value=0.0)
            .rename_axis("date")
            .reset_index()
        )

    total_sales_amount = float(dd_at_date["sales_amount"].sum())
    return_amount = float(dd_at_date["return_amount"].sum())
    net_sales_amount = float(dd_at_date["net_sales_amount"].sum())
    return_rate_amount = (return_amount / total_sales_amount) if total_sales_amount > 0 else None

    sparkline = [
        DailySparklinePoint(
            date=row.date.date(),
            sales_amount=row.sales_amount,
            return_amount=row.return_amount,
            net_sales_amount=row.net_sales_amount,
        )
        for row in spark_df.itertuples(index=False)
    ]

    forecast_basis_week = resolve_forecast_basis_week_for_date(basis_date)

    # 판매건수 — 원본 매출 기록(raw transaction) 행 수 기준. daily_summary_agg/daily_demand는
    # 이미 SKU-day 단위로 합쳐져 있어 "몇 건의 판매 기록이 있었는지"를 알 수 없으므로,
    # region-sales와 동일하게 raw 파일에서 직접 센다(수량>0 행만 — 새 계산이 아니라 단순 count,
    # 주문번호가 없는 데이터라 주문건수가 아닌 판매기록 행수로 정의한다).
    valid_sku_keys = set(zip(pm["center_id"], pm["sku_id"]))
    raw = _filter_by_center(load_sales_transactions_raw(), center)
    raw = raw[(raw["거래일"] == basis_date) & (raw["수량"] > 0)]
    matched = pd.Series(
        [k in valid_sku_keys for k in zip(raw["center_id"], raw["sku_id"])], index=raw.index, dtype=bool,
    )
    sales_record_count = int(matched.sum())

    return DailySummaryResponse(
        date=basis_date.date(),
        forecast_basis_week=None if forecast_basis_week is None else forecast_basis_week.date(),
        total_sales_amount=total_sales_amount,
        return_amount=return_amount,
        net_sales_amount=net_sales_amount,
        active_sku_count=active_sku_count,
        sales_record_count=sales_record_count,
        return_rate_amount=return_rate_amount,
        sales_qty_by_unit={k: float(v) for k, v in sales_qty_by_unit.items()},
        sparkline=sparkline,
    )


# ── 일간 카테고리별 매출 ───────────────────────────────────────────
class CategoryProductSalesItem(BaseModel):
    sku_id:         str
    product_name:   str
    barcode:        str
    option_code:    str
    sales_amount:   float
    share:          float


class DailyCategorySalesResponse(BaseModel):
    center_id:            str
    date:                  date
    total_sales_amount:    float
    top5:                   List[CategorySalesItem]
    ranking:                List[CategorySalesItem]
    # 소분류까지 선택된 경우에만 채워진다(그 외에는 None, 기존 호출/응답과 완전히 동일).
    # 소분류 1개만 남으면 top5/ranking이 100% 1행이라 의미가 없어, 그 범위 안의
    # 상품별 매출 TOP5/전체 랭킹을 별도 필드로 추가 제공한다.
    product_top5:          Optional[List[CategoryProductSalesItem]] = None
    product_ranking:       Optional[List[CategoryProductSalesItem]] = None


@router.get("/daily/category-sales", response_model=DailyCategorySalesResponse)
def get_daily_category_sales(
    center:          str            = Query(..., description="A / B"),
    date_:           Optional[str]  = Query(None, alias="date", description="YYYY-MM-DD, 조회일"),
    category_large:  Optional[str]  = Query(None),
    category_middle: Optional[str]  = Query(None),
    category_small:  Optional[str]  = Query(None),
):
    basis_date = _resolve_daily_basis_date(date_)

    group_col = "KAN_대분류"
    if category_middle:
        group_col = "KAN_소분류"
    elif category_large:
        group_col = "KAN_중분류"

    # daily_summary_agg는 이미 (center,date,KAN 3단,option_code) 단위 합산본이라 merge 없이
    # 자체 KAN 컬럼으로 바로 필터링한다 — 결과는 canonical을 merge하던 것과 동일한 합계다.
    agg = load_daily_summary_agg()
    agg = agg[(agg["center_id"] == center) & (agg["date"] == basis_date)]
    if category_large:
        agg = agg[agg["KAN_대분류"] == category_large]
    if category_middle:
        agg = agg[agg["KAN_중분류"] == category_middle]
    if category_small:
        agg = agg[agg["KAN_소분류"] == category_small]

    # canonical(dense grid)에서는 그 날 매출이 0인 카테고리도 명시적으로 0원 행으로 나타난다
    # (그 카테고리 SKU가 이미 grid_start를 지나 존재했다면). sparse 집계에는 그런 0행이
    # 아예 없으므로, grid에서 basis_date 시점 이미 존재했던 카테고리 목록을 구해 0으로 채운다.
    grid_scope = load_daily_sku_grid()
    grid_scope = grid_scope[grid_scope["center_id"] == center]
    if category_large:
        grid_scope = grid_scope[grid_scope["KAN_대분류"] == category_large]
    if category_middle:
        grid_scope = grid_scope[grid_scope["KAN_중분류"] == category_middle]
    if category_small:
        grid_scope = grid_scope[grid_scope["KAN_소분류"] == category_small]
    # groupby(...).sum()은 기본적으로 그룹 키를 오름차순 정렬한 상태로 반환하므로, canonical과
    # 동일한 sort_values(0값 tie-break 순서 포함) 결과를 재현하려면 reindex 대상도 같은
    # 오름차순으로 정렬해 넣어야 한다.
    existing_categories = sorted(grid_scope.loc[grid_scope["grid_start"] <= basis_date, group_col].unique())

    ranking_sums = (
        agg.groupby(group_col)["sales_amount"].sum()
        .reindex(existing_categories, fill_value=0.0)
        .sort_values(ascending=False)
    )
    total = float(ranking_sums.sum())

    ranking = [
        CategorySalesItem(
            category=cat,
            sales_amount=float(v),
            share=(float(v) / total) if total > 0 else 0.0,
        )
        for cat, v in ranking_sums.items()
    ]

    product_ranking = None
    product_top5 = None
    if category_large or category_middle or category_small:
        # 카테고리가 어느 레벨이든 선택되면(대/중/소 무관) 그 범위의 SKU를 daily_transactions_sparse
        # (활동 있는 SKU-day만 남은 sparse 버전)에서 골라 매출(sales_amount) 기준으로 랭킹한다.
        pm_scope = load_product_master()
        pm_scope = pm_scope[pm_scope["center_id"] == center]
        if category_large:
            pm_scope = pm_scope[pm_scope["KAN_대분류"] == category_large]
        if category_middle:
            pm_scope = pm_scope[pm_scope["KAN_중분류"] == category_middle]
        if category_small:
            pm_scope = pm_scope[pm_scope["KAN_소분류"] == category_small]
        scope_keys = set(zip(pm_scope["center_id"], pm_scope["sku_id"]))

        dt = load_daily_transactions_sparse()
        dt = dt[(dt["center_id"] == center) & (dt["date"] == basis_date) & (dt["sales_amount"] > 0)]
        matched = pd.Series(
            [k in scope_keys for k in zip(dt["center_id"], dt["sku_id"])], index=dt.index, dtype=bool,
        )
        dt = dt[matched]
        dt = dt.merge(pm_scope[["sku_id", "상품명", "barcode"]], on="sku_id", how="inner")
        dt = dt.sort_values("sales_amount", ascending=False)

        product_ranking = [
            CategoryProductSalesItem(
                sku_id=row.sku_id,
                product_name=row.상품명,
                barcode=row.barcode,
                option_code=row.option_code,
                sales_amount=float(row.sales_amount),
                share=(float(row.sales_amount) / total) if total > 0 else 0.0,
            )
            for row in dt.itertuples(index=False)
        ]
        product_top5 = product_ranking[:5]

    return DailyCategorySalesResponse(
        center_id=center,
        date=basis_date.date(),
        total_sales_amount=total,
        top5=ranking[:5],
        ranking=ranking,
        product_top5=product_top5,
        product_ranking=product_ranking,
    )


# ── 일간 판매지역별 매출 ───────────────────────────────────────────
class DailyRegionSalesResponse(BaseModel):
    center_id:             str
    date:                    date
    total_sales_amount:      float
    top5:                     List[RegionSalesItem]
    ranking:                  List[RegionSalesItem]
    top_products:             Optional[List[RegionProductItem]] = None


@router.get("/daily/region-sales", response_model=DailyRegionSalesResponse)
def get_daily_region_sales(
    center:          str            = Query(..., description="A / B"),
    date_:           Optional[str]  = Query(None, alias="date", description="YYYY-MM-DD, 조회일"),
    sido:            Optional[str]  = Query(None, description="지정 시 해당 시도+시군구의 상위 상품도 반환"),
    sigungu:         Optional[str]  = Query(None, description="sido와 함께 지정해야 top_products가 채워진다"),
    category_large:  Optional[str]  = Query(None),
    category_middle: Optional[str]  = Query(None),
    category_small:  Optional[str]  = Query(None),
):
    basis_date = _resolve_daily_basis_date(date_)

    pm = load_product_master()
    pm_center = pm[pm["center_id"] == center]
    if category_large:
        pm_center = pm_center[pm_center["KAN_대분류"] == category_large]
    if category_middle:
        pm_center = pm_center[pm_center["KAN_중분류"] == category_middle]
    if category_small:
        pm_center = pm_center[pm_center["KAN_소분류"] == category_small]
    valid_sku_keys = set(zip(pm_center["center_id"], pm_center["sku_id"]))

    # daily_region_agg는 raw 거래 파일을 (center,date,시도,시군구,sku_id,상품명,옵션코드)
    # 단위로 미리 합산해 둔 것이다(수량>0만, product_master 전체 유효 SKU로 이미 필터됨) —
    # 여기서는 카테고리 범위만 추가로 좁힌다.
    tx = load_daily_region_agg()
    tx = tx[(tx["center_id"] == center) & (tx["date"] == basis_date)]
    # tx가 이미 0행(무거래일)이면 파이썬 list(빈 리스트)로 인덱싱할 때 pandas가 이를
    # "컬럼 목록"으로 오인해 컬럼까지 통째로 사라져 이후 tx["sales_amount"]가 KeyError로
    # 500을 낸다 — 인덱스가 명시된 Series로 만들어 행 boolean mask임을 명확히 한다.
    matched = pd.Series(
        [k in valid_sku_keys for k in zip(tx["center_id"], tx["sku_id"])], index=tx.index, dtype=bool,
    )
    tx = tx[matched]

    total = float(tx["sales_amount"].sum())

    region_agg = (
        tx.groupby(["sido", "sigungu"], as_index=False)["sales_amount"]
        .sum()
        .sort_values("sales_amount", ascending=False)
    )

    ranking = [
        RegionSalesItem(
            sido=row.sido,
            sigungu=row.sigungu,
            sales_amount=float(row.sales_amount),
            share=(float(row.sales_amount) / total) if total > 0 else 0.0,
        )
        for row in region_agg.itertuples(index=False)
    ]

    top_products = None
    if sido and sigungu:
        region_tx = tx[(tx["sido"] == sido) & (tx["sigungu"] == sigungu)]
        prod_agg = (
            region_tx.groupby(["sku_id", "product_name", "option_code"], as_index=False)["sales_amount"]
            .sum()
            .sort_values("sales_amount", ascending=False)
            .head(10)
        )
        top_products = [
            RegionProductItem(
                sku_id=row.sku_id,
                product_name=row.product_name,
                option_code=row.option_code,
                sales_amount=float(row.sales_amount),
            )
            for row in prod_agg.itertuples(index=False)
        ]

    return DailyRegionSalesResponse(
        center_id=center,
        date=basis_date.date(),
        total_sales_amount=total,
        top5=ranking[:5],
        ranking=ranking,
        top_products=top_products,
    )


# ── 일간 SKU별 당일 활동(대시보드 KPI modal 공용) ──────────────────
class ProductActivityItem(BaseModel):
    sku_id:         str
    product_name:   str
    barcode:        str
    option_code:    str
    sales_qty:      float
    sales_amount:   float
    return_qty:     float
    return_amount:  float
    record_count:   int


class DailyProductActivityResponse(BaseModel):
    center_id:  str
    date:        date
    items:       List[ProductActivityItem]


@router.get("/daily/product-activity", response_model=DailyProductActivityResponse)
def get_daily_product_activity(
    center:          str            = Query(..., description="A / B / ALL"),
    date_:           Optional[str]  = Query(None, alias="date", description="YYYY-MM-DD, 조회일"),
    category_large:  Optional[str]  = Query(None),
    category_middle: Optional[str]  = Query(None),
    category_small:  Optional[str]  = Query(None),
):
    """대시보드 KPI modal(판매금액/순판매금액/판매 SKU 수/반품률) 4개가 공유하는 SKU별
    당일 활동 목록. daily_transactions_sparse(입고/반출/판매/반품이 전부 0인 행은 제외된
    sparse 버전)를 그대로 사용하며 새 계산은 하지 않는다 — 각 modal은 이 목록을 받아
    자신에게 맞는 조건(sales_qty>0/return_amount>0 등)으로 필터링·정렬만 frontend에서 한다."""
    basis_date = _resolve_daily_basis_date(date_)

    pm = _filter_by_center(load_product_master(), center)
    if category_large:
        pm = pm[pm["KAN_대분류"] == category_large]
    if category_middle:
        pm = pm[pm["KAN_중분류"] == category_middle]
    if category_small:
        pm = pm[pm["KAN_소분류"] == category_small]

    dt = _filter_by_center(load_daily_transactions_sparse(), center)
    dt = dt[dt["date"] == basis_date]
    dt = dt.merge(pm[["center_id", "sku_id", "상품명", "barcode"]], on=["center_id", "sku_id"], how="inner")

    # 판매기록 수(record_count) — 원본 매출 기록(raw transaction) 행 수. daily_transactions_sparse는
    # SKU-day 단위로 이미 합쳐져 있어 몇 건의 판매 기록이 있었는지 알 수 없으므로, /daily/summary의
    # sales_record_count와 동일하게 raw 파일에서 SKU별로 센다(수량>0 행만, 새 계산 아님).
    valid_sku_keys = set(zip(pm["center_id"], pm["sku_id"]))
    raw = _filter_by_center(load_sales_transactions_raw(), center)
    raw = raw[(raw["거래일"] == basis_date) & (raw["수량"] > 0)]
    matched = pd.Series(
        [k in valid_sku_keys for k in zip(raw["center_id"], raw["sku_id"])], index=raw.index, dtype=bool,
    )
    raw = raw[matched]
    record_counts = raw.groupby("sku_id").size()

    items = [
        ProductActivityItem(
            sku_id=row.sku_id,
            product_name=row.상품명,
            barcode=row.barcode,
            option_code=row.option_code,
            sales_qty=float(row.sales_qty),
            sales_amount=float(row.sales_amount),
            return_qty=float(row.return_qty),
            return_amount=float(row.return_amount),
            record_count=int(record_counts.get(row.sku_id, 0)),
        )
        for row in dt.itertuples(index=False)
    ]

    return DailyProductActivityResponse(center_id=center, date=basis_date.date(), items=items)


# ── 일간 반품 TOP5(금액 기준) ──────────────────────────────────────
class DailyReturnsResponse(BaseModel):
    center_id:      str
    date:             date
    top5:              List[ReturnItem]
    ranking:           List[ReturnItem]


@router.get("/daily/returns", response_model=DailyReturnsResponse)
def get_daily_returns(
    center:          str            = Query(..., description="A / B"),
    date_:           Optional[str]  = Query(None, alias="date", description="YYYY-MM-DD, 조회일"),
    category_large:  Optional[str]  = Query(None),
    category_middle: Optional[str]  = Query(None),
    category_small:  Optional[str]  = Query(None),
):
    basis_date = _resolve_daily_basis_date(date_)

    # return_amount>0인 행은 daily_transactions_sparse(활동 있는 SKU-day만 남긴 sparse
    # 버전)에도 당연히 전부 포함돼 있으므로 canonical 대신 이 파일을 쓴다.
    dt = load_daily_transactions_sparse()
    dt = dt[(dt["center_id"] == center) & (dt["date"] == basis_date) & (dt["return_amount"] > 0)]

    pm = load_product_master()
    pm_center = pm[pm["center_id"] == center]
    if category_large:
        pm_center = pm_center[pm_center["KAN_대분류"] == category_large]
    if category_middle:
        pm_center = pm_center[pm_center["KAN_중분류"] == category_middle]
    if category_small:
        pm_center = pm_center[pm_center["KAN_소분류"] == category_small]
    dt = dt.merge(pm_center[["sku_id", "상품명", "barcode"]], on="sku_id", how="inner")
    dt = dt.sort_values("return_amount", ascending=False)

    ranking = [
        ReturnItem(
            center_id=center,
            sku_id=row.sku_id,
            product_name=row.상품명,
            barcode=row.barcode,
            option_code=row.option_code,
            return_qty=float(row.return_qty),
            return_amount=float(row.return_amount),
        )
        for row in dt.itertuples(index=False)
    ]

    return DailyReturnsResponse(
        center_id=center,
        date=basis_date.date(),
        top5=ranking[:5],
        ranking=ranking,
    )


# ── 일간 판매 증가 TOP5(전주 동일요일 대비, 전일 대비 아님) ─────────
# 요일별 판매 편차가 매우 커서(평일 수천 vs 토요일 133 vs 일요일 3, A센터 2024 실측)
# 전일 대비 비교는 요일 효과에 압도되어 의미가 없다. 반드시 date - 7일과 비교한다.
class DailySalesIncreaseResponse(BaseModel):
    center_id:              str
    date:                     date
    prev_date:                date
    increased_sku_count:      int
    top5:                      List[SalesIncreaseItem]
    ranking:                   List[SalesIncreaseItem]


@router.get("/daily/sales-surge", response_model=DailySalesIncreaseResponse)
def get_daily_sales_surge(
    center:          str            = Query(..., description="A / B"),
    date_:           Optional[str]  = Query(None, alias="date", description="YYYY-MM-DD, 조회일"),
    category_large:  Optional[str]  = Query(None),
    category_middle: Optional[str]  = Query(None),
    category_small:  Optional[str]  = Query(None),
    option_code:     Optional[str]  = Query(None, description="EA/BX/CS 등 — 지정 시 해당 단위 SKU만 반환(수량 ranking 단위 혼합 방지). 생략 시 기존과 동일."),
):
    basis_date = _resolve_daily_basis_date(date_)
    prev_date = basis_date - pd.Timedelta(days=7)

    # sparse에 없는 (sku,date)는 활동이 전혀 없었다는 뜻이므로 sales_qty=0으로 취급한다.
    # date 쪽에서 sales_qty=0인 SKU는 어차피 increase_qty>0을 만족할 수 없으므로(현재<=이전+0)
    # cur에서 빠져도 최종 ranking은 canonical과 동일하다.
    dd = load_daily_transactions_sparse()
    dd_center = dd[dd["center_id"] == center]

    cur = dd_center[dd_center["date"] == basis_date][["sku_id", "sales_qty"]].rename(
        columns={"sales_qty": "current_sales_qty"}
    )
    prev = dd_center[dd_center["date"] == prev_date][["sku_id", "sales_qty"]].rename(
        columns={"sales_qty": "prev_sales_qty"}
    )
    merged = cur.merge(prev, on="sku_id", how="left")
    merged["prev_sales_qty"] = merged["prev_sales_qty"].fillna(0.0)
    merged["increase_qty"] = merged["current_sales_qty"] - merged["prev_sales_qty"]
    merged = merged[merged["increase_qty"] > 0].copy()
    merged["increase_pct"] = merged.apply(
        lambda r: (r["increase_qty"] / r["prev_sales_qty"] * 100) if r["prev_sales_qty"] > 0 else None,
        axis=1,
    )
    merged = merged.sort_values("increase_qty", ascending=False)

    pm = load_product_master()
    pm_center = pm[pm["center_id"] == center]
    if category_large:
        pm_center = pm_center[pm_center["KAN_대분류"] == category_large]
    if category_middle:
        pm_center = pm_center[pm_center["KAN_중분류"] == category_middle]
    if category_small:
        pm_center = pm_center[pm_center["KAN_소분류"] == category_small]
    if option_code:
        pm_center = pm_center[pm_center["option_code"] == option_code]
    merged = merged.merge(pm_center[["sku_id", "상품명", "barcode", "option_code"]], on="sku_id", how="inner")
    merged = merged.sort_values("increase_qty", ascending=False)

    ranking = [
        SalesIncreaseItem(
            center_id=center,
            sku_id=row.sku_id,
            product_name=row.상품명,
            barcode=row.barcode,
            option_code=row.option_code,
            prev_sales_qty=float(row.prev_sales_qty),
            current_sales_qty=float(row.current_sales_qty),
            increase_qty=float(row.increase_qty),
            increase_pct=None if pd.isna(row.increase_pct) else float(row.increase_pct),
        )
        for row in merged.itertuples(index=False)
    ]

    return DailySalesIncreaseResponse(
        center_id=center,
        date=basis_date.date(),
        prev_date=prev_date.date(),
        increased_sku_count=len(ranking),
        top5=ranking[:5],
        ranking=ranking,
    )


# ── 일간 선택 SKU 입출고/판매/반품 이력(최근 30일) ──────────────────
class DailyTransactionHistoryPoint(BaseModel):
    date:                    date
    inbound_qty:             float
    outbound_qty:            float
    sales_qty:               float
    return_qty:              float
    inbound_amount:          float
    outbound_amount:         float
    inbound_total_amount:    float
    outbound_total_amount:   float
    sales_amount:            float
    return_amount:           float


class DailyTransactionsResponse(BaseModel):
    center_id:     str
    sku_id:         str
    product_name:   str
    option_code:    str
    date:            date
    history:         List[DailyTransactionHistoryPoint]


@router.get("/daily/transactions", response_model=DailyTransactionsResponse)
def get_daily_transactions(
    center:         str            = Query(..., description="A / B"),
    sku_id:         str            = Query(...),
    date_:          Optional[str]  = Query(None, alias="date", description="YYYY-MM-DD, 조회일"),
    history_days:   int            = Query(30, ge=1),
):
    pm = load_product_master()
    pm_match = pm[(pm["center_id"] == center) & (pm["sku_id"] == sku_id)]
    if pm_match.empty:
        raise HTTPException(status_code=404, detail="해당 center_id+sku_id 상품을 찾을 수 없습니다.")
    pm_row = pm_match.iloc[0]

    grid = load_daily_sku_grid()
    grid_match = grid[(grid["center_id"] == center) & (grid["sku_id"] == sku_id)]
    if grid_match.empty:
        raise HTTPException(status_code=404, detail="해당 SKU의 실적 데이터가 없습니다.")
    grid_start = grid_match.iloc[0]["grid_start"]

    basis_date = _resolve_daily_date_for_sku_grid_start(grid_start, date_)

    # daily_transactions_sparse에는 활동이 전혀 없는 날이 빠져 있으므로, canonical dense와
    # 동일한 history를 재현하기 위해 [range_start, basis_date] 전체 날짜를 만들고 없는 날은
    # (grid_start 이전이 아닌 한) 0으로 채운다 — history_days 규칙(가장 최근 N일, grid_start
    # 이전으로는 넘어가지 않음)은 canonical의 .head(history_days) 동작과 동일하다.
    range_start = max(grid_start, basis_date - pd.Timedelta(days=history_days - 1))
    full_range = pd.date_range(range_start, basis_date, freq="D")

    dt_sku = load_daily_transactions_sparse()
    dt_sku = dt_sku[(dt_sku["center_id"] == center) & (dt_sku["sku_id"] == sku_id)]
    dt_sku = dt_sku[(dt_sku["date"] >= range_start) & (dt_sku["date"] <= basis_date)]

    numeric_cols = [
        "inbound_qty", "outbound_qty", "sales_qty", "return_qty",
        "inbound_amount", "outbound_amount", "inbound_total_amount", "outbound_total_amount",
        "sales_amount", "return_amount",
    ]
    hist = (
        dt_sku.set_index("date")[numeric_cols]
        .reindex(full_range, fill_value=0.0)
        .rename_axis("date")
        .reset_index()
    )
    history = [
        DailyTransactionHistoryPoint(
            date=row.date.date(),
            inbound_qty=row.inbound_qty,
            outbound_qty=row.outbound_qty,
            sales_qty=row.sales_qty,
            return_qty=row.return_qty,
            inbound_amount=row.inbound_amount,
            outbound_amount=row.outbound_amount,
            inbound_total_amount=row.inbound_total_amount,
            outbound_total_amount=row.outbound_total_amount,
            sales_amount=row.sales_amount,
            return_amount=row.return_amount,
        )
        for row in hist.itertuples(index=False)
    ]

    return DailyTransactionsResponse(
        center_id=center,
        sku_id=sku_id,
        product_name=pm_row.상품명,
        option_code=pm_row.option_code,
        date=basis_date.date(),
        history=history,
    )


# ── 일간 선택 SKU 추정재고 ─────────────────────────────────────────
class DailyInventoryHistoryPoint(BaseModel):
    date:                  date
    inbound_qty:           float
    outbound_qty:          float
    sales_qty:             float
    return_qty:            float
    estimated_inventory:   Optional[float]


class DailyInventoryResponse(BaseModel):
    center_id:                     str
    sku_id:                         str
    product_name:                   str
    option_code:                     str
    date:                             date
    return_policy:                   str
    inventory_available:             bool
    estimated_inventory:             Optional[float]
    estimated_opening_inventory:     Optional[float]
    current_day:                      CurrentWeekTransaction
    history:                          List[DailyInventoryHistoryPoint]


@router.get("/daily/inventory", response_model=DailyInventoryResponse)
def get_daily_inventory(
    center:         str            = Query(..., description="A / B"),
    sku_id:         str            = Query(...),
    date_:          Optional[str]  = Query(None, alias="date", description="YYYY-MM-DD, 조회일"),
    return_policy:  str            = Query("excluded", description="excluded / included"),
    history_days:   int            = Query(30, ge=1),
):
    if return_policy not in ("excluded", "included"):
        raise HTTPException(status_code=400, detail="return_policy는 excluded 또는 included여야 합니다.")

    pm = load_product_master()
    pm_match = pm[(pm["center_id"] == center) & (pm["sku_id"] == sku_id)]
    if pm_match.empty:
        raise HTTPException(status_code=404, detail="해당 center_id+sku_id 상품을 찾을 수 없습니다.")
    pm_row = pm_match.iloc[0]

    dt_sku = load_daily_transactions()
    dt_sku = dt_sku[(dt_sku["center_id"] == center) & (dt_sku["sku_id"] == sku_id)]
    inv_sku = load_inventory_daily()
    inv_sku = inv_sku[(inv_sku["center_id"] == center) & (inv_sku["sku_id"] == sku_id)]
    if dt_sku.empty or inv_sku.empty:
        raise HTTPException(status_code=404, detail="해당 SKU의 실적/재고 데이터가 없습니다.")

    basis_date = _resolve_daily_date_for_sku(dt_sku["date"], date_)

    dt_row = dt_sku[dt_sku["date"] == basis_date].iloc[0]
    inv_row = inv_sku[inv_sku["date"] == basis_date].iloc[0]

    inventory_available = bool(inv_row["inventory_available"])
    inv_col = (
        "estimated_inventory_return_excluded"
        if return_policy == "excluded"
        else "estimated_inventory_return_included"
    )
    estimated_inventory = None if not inventory_available else float(inv_row[inv_col])
    estimated_opening_inventory = (
        None if not inventory_available else float(inv_row["estimated_opening_inventory"])
    )

    current_day = CurrentWeekTransaction(
        inbound_qty=float(dt_row["inbound_qty"]),
        outbound_qty=float(dt_row["outbound_qty"]),
        sales_qty=float(dt_row["sales_qty"]),
        return_qty=float(dt_row["return_qty"]),
        inbound_amount=float(dt_row["inbound_amount"]),
        outbound_amount=float(dt_row["outbound_amount"]),
        sales_amount=float(dt_row["sales_amount"]),
        return_amount=float(dt_row["return_amount"]),
    )

    hist_tx = (
        dt_sku[dt_sku["date"] <= basis_date]
        .sort_values("date", ascending=False)
        .head(history_days)
        .sort_values("date")
    )
    hist_inv = inv_sku[
        ["date", "estimated_inventory_return_excluded", "estimated_inventory_return_included", "inventory_available"]
    ]
    hist = hist_tx.merge(hist_inv, on="date", how="left")

    history = [
        DailyInventoryHistoryPoint(
            date=row.date.date(),
            inbound_qty=row.inbound_qty,
            outbound_qty=row.outbound_qty,
            sales_qty=row.sales_qty,
            return_qty=row.return_qty,
            estimated_inventory=(
                None if not bool(row.inventory_available) else float(getattr(row, inv_col))
            ),
        )
        for row in hist.itertuples(index=False)
    ]

    return DailyInventoryResponse(
        center_id=center,
        sku_id=sku_id,
        product_name=pm_row.상품명,
        option_code=pm_row.option_code,
        date=basis_date.date(),
        return_policy=return_policy,
        inventory_available=inventory_available,
        estimated_inventory=estimated_inventory,
        estimated_opening_inventory=estimated_opening_inventory,
        current_day=current_day,
        history=history,
    )
