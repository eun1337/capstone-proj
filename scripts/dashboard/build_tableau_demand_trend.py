"""
build_tableau_demand_trend.py

tableau_export/tableau_weekly_sku.csv를 읽어 "수요 추이" 차트용
long-format Fact 파일 tableau_export/tableau_demand_trend.csv를 생성한다.
기존 tableau_export CSV 4종은 읽기만 하며 수정하지 않는다.
"""

from pathlib import Path

import pandas as pd

SRC = Path("tableau_export/tableau_weekly_sku.csv")
OUT = Path("tableau_export/tableau_demand_trend.csv")

USECOLS = [
    "center_id", "sku_id", "week_st", "sales_qty",
    "h1_target_date", "h1_pred",
    "h2_target_date", "h2_pred",
    "h4_target_date", "h4_pred",
]
OUT_COLS = ["center_id", "sku_id", "plot_date", "value", "series", "forecast_origin", "horizon"]


def build_actual(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["center_id", "sku_id", "week_st", "sales_qty"]].rename(
        columns={"week_st": "plot_date", "sales_qty": "value"}
    )
    out["series"] = "actual"
    out["forecast_origin"] = pd.NA
    out["horizon"] = pd.NA
    return out


def build_forecast(df: pd.DataFrame, h: str) -> pd.DataFrame:
    sub = df[df[f"{h}_pred"].notna() & df[f"{h}_target_date"].notna()]
    out = sub[["center_id", "sku_id", f"{h}_target_date", f"{h}_pred", "week_st"]].rename(
        columns={f"{h}_target_date": "plot_date", f"{h}_pred": "value", "week_st": "forecast_origin"}
    )
    out["series"] = "forecast"
    out["horizon"] = h.upper()
    return out


def build_origin_anchor(df: pd.DataFrame) -> pd.DataFrame:
    has_pred = df["h1_pred"].notna() | df["h2_pred"].notna() | df["h4_pred"].notna()
    sub = df.loc[has_pred, ["center_id", "sku_id", "week_st", "sales_qty"]]
    out = sub.rename(columns={"week_st": "plot_date", "sales_qty": "value"})
    out["forecast_origin"] = sub["week_st"]
    out["series"] = "forecast"
    out["horizon"] = "origin"
    return out


def main() -> None:
    df = pd.read_csv(SRC, usecols=USECOLS)

    parts = [
        build_actual(df),
        build_origin_anchor(df),
        build_forecast(df, "h1"),
        build_forecast(df, "h2"),
        build_forecast(df, "h4"),
    ]
    out = pd.concat(parts, ignore_index=True)[OUT_COLS]
    out = out.sort_values(["center_id", "sku_id", "plot_date", "series"]).reset_index(drop=True)
    out.to_csv(OUT, index=False)

    print(f"input rows: {len(df):,}")
    for name, part in zip(["actual", "origin", "h1", "h2", "h4"], parts):
        print(f"{name} rows: {len(part):,}")
    print(f"final rows: {len(out):,}")
    print("null counts:", out[["center_id", "sku_id", "plot_date", "value"]].isna().sum().to_dict())
    print("series counts:\n", out["series"].value_counts())
    print("horizon counts:\n", out["horizon"].value_counts(dropna=False))
    print("plot_date range:", out["plot_date"].min(), out["plot_date"].max())
    fo = out["forecast_origin"].dropna()
    print("forecast_origin range:", fo.min(), fo.max())
    print("output size bytes:", OUT.stat().st_size)


if __name__ == "__main__":
    main()
