"""Tableau 라인차트용 최종본: 2021-2023 실적 + 2024(h1/h2/h4) 예측을 하나의 시계열로 이어붙인 파일.

설계 결정:
  - 2024 구간 예측은 h1(7일)/h2(14일)/h4(30일) 전부 포함, horizon_label로 구분.
  - 2021~2023 실적 구간은 h1/h2/h4 3벌로 복제(horizon_label별로 동일 값 반복) ->
    Tableau에서 horizon_label 단일값 퀵필터를 h1/h2/h4 중 아무거나 선택해도
    실적 구간이 항상 같이 조회됨. ('ALL' 단일값 태깅 방식은 단일선택 필터와
    바로 호환되지 않아 채택하지 않음 — 계산식 없이 바로 동작하는 쪽을 선택.)
  - 실적->예측 라인 연속성: 각 (center_id, sku_id)의 "마지막 실적 시점" 값을
    예측(series='예측') 쪽에도 동일 날짜/값으로 한 번 더 복제해 앵커 포인트로 삽입.
    두 계열이 그 지점에서 정확히 맞닿아 라인이 끊기지 않음.
    (h1은 갭 2주, h2는 3주, h4는 5주 정도 실제 데이터 없는 구간을 직선으로 건너뜀 —
    구간 자체가 없어서가 아니라 예측 산출물이 해당 target_date를 안 갖고 있기 때문.)

원천:
  - data/development_2021_2023.parquet -> 2021-2023 실적(qty)
  - data/tableau/dashboard_actual_vs_forecast.parquet -> 2024 예측(y_pred, h1/h2/h4 전부)

산출:
  - data/tableau/dashboard_actual_forecast_timeline.parquet
  - data/tableau/dashboard_actual_forecast_timeline.csv
"""
import os
import pandas as pd

HIST_COLS = ["center_id", "sku_id", "week_st", "qty",
             "상품명", "규격", "입수", "KAN_CODE", "KAN_대분류", "KAN_중분류", "KAN_소분류"]

DIM_COLS = ["center_id", "sku_id", "바코드", "옵션코드", "입수",
            "상품명", "규격", "KAN_CODE", "KAN_대분류", "KAN_중분류", "KAN_소분류"]

HORIZONS = ["h1", "h2", "h4"]

OUT_PARQUET = "data/tableau/dashboard_actual_forecast_timeline.parquet"
OUT_CSV = "data/tableau/dashboard_actual_forecast_timeline.csv"

# Tableau Cloud 웹 업로드 1.0GB 제한 때문에 센터별로도 분리 저장
CENTER_CSV_TEMPLATE = "data/tableau/timeline_center_{center}.csv"
# 센터만 나눠도 A가 1GB를 넘어서, 센터x horizon으로 한 번 더 분리
CENTER_HORIZON_CSV_TEMPLATE = "data/tableau/timeline_center_{center}_{horizon}.csv"


def build_history() -> pd.DataFrame:
    hist = pd.read_parquet("data/development_2021_2023.parquet", columns=HIST_COLS)
    hist["KAN_CODE"] = hist["KAN_CODE"].astype("Int64").astype(str).str.zfill(6)
    parsed = hist["sku_id"].str.split("*", expand=True)
    hist["바코드"] = parsed[0]
    hist["옵션코드"] = parsed[1]
    hist = hist.rename(columns={"week_st": "date", "qty": "value"})
    hist["series"] = "실적"
    return hist[DIM_COLS + ["date", "series", "value"]]


def replicate_by_horizon(df: pd.DataFrame) -> pd.DataFrame:
    """horizon_label 없는 df를 h1/h2/h4 3벌로 복제."""
    reps = []
    for h in HORIZONS:
        r = df.copy()
        r["horizon_label"] = h
        reps.append(r)
    return pd.concat(reps, ignore_index=True)


def build_forecast() -> pd.DataFrame:
    fc = pd.read_parquet("data/tableau/dashboard_actual_vs_forecast.parquet")
    fc = fc[fc["horizon_label"].isin(HORIZONS)].copy()
    fc = fc.rename(columns={"target_date": "date", "y_pred": "value"})
    fc["series"] = "예측"
    return fc[DIM_COLS + ["date", "series", "horizon_label", "value"]]


def build_bridge(hist: pd.DataFrame) -> pd.DataFrame:
    """각 (center_id, sku_id)의 마지막 실적 포인트를 예측 계열에도 동일하게 삽입 -> 라인 연속."""
    last_actual = (
        hist.sort_values("date")
        .groupby(["center_id", "sku_id"], as_index=False)
        .tail(1)
        .copy()
    )
    last_actual["series"] = "예측"
    return replicate_by_horizon(last_actual)


def main():
    hist_raw = build_history()
    hist = replicate_by_horizon(hist_raw)
    forecast = build_forecast()
    bridge = build_bridge(hist_raw)

    combined = pd.concat([hist, bridge, forecast], ignore_index=True)
    combined = combined.sort_values(["center_id", "sku_id", "horizon_label", "date"]).reset_index(drop=True)
    combined = combined[DIM_COLS + ["date", "horizon_label", "series", "value"]]

    os.makedirs("data/tableau", exist_ok=True)
    combined.to_parquet(OUT_PARQUET, index=False)
    combined.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")

    center_csv_sizes = {}
    for center in sorted(combined["center_id"].unique()):
        path = CENTER_CSV_TEMPLATE.format(center=center)
        combined[combined["center_id"] == center].to_csv(path, index=False, encoding="utf-8-sig")
        center_csv_sizes[path] = round(os.path.getsize(path) / 1024 / 1024, 2)

    center_horizon_csv_sizes = {}
    for center in sorted(combined["center_id"].unique()):
        for horizon in HORIZONS:
            path = CENTER_HORIZON_CSV_TEMPLATE.format(center=center, horizon=horizon)
            sub = combined[(combined["center_id"] == center) & (combined["horizon_label"] == horizon)]
            sub.to_csv(path, index=False, encoding="utf-8-sig")
            center_horizon_csv_sizes[path] = round(os.path.getsize(path) / 1024 / 1024, 2)

    print("=== 완료 ===")
    print("총 행수:", len(combined))
    print("  실적(복제 3벌):", len(hist), " / 앵커(브릿지, 3벌):", len(bridge), " / 예측:", len(forecast))
    print("date 범위:", combined["date"].min(), "~", combined["date"].max())
    print("\nhorizon_label x series 교차표:")
    print(pd.crosstab(combined["horizon_label"], combined["series"]))

    print("\n=== 연속성 확인: h1 기준 (center=A, 임의 sku 1개) 경계 구간 ===")
    sample_sku = hist_raw[hist_raw["center_id"] == "A"]["sku_id"].iloc[0]
    check = combined[
        (combined["center_id"] == "A") & (combined["sku_id"] == sample_sku)
        & (combined["horizon_label"] == "h1")
    ].sort_values("date")
    print(check[["date", "series", "value"]].tail(6).to_string())

    print("\nparquet 크기(MB):", round(os.path.getsize(OUT_PARQUET) / 1024 / 1024, 2))
    print("csv 크기(MB):", round(os.path.getsize(OUT_CSV) / 1024 / 1024, 2))
    print("\n센터별 분리 CSV:")
    for path, size_mb in center_csv_sizes.items():
        print(f"  {path}: {size_mb}MB")
    print("\n센터x horizon 분리 CSV:")
    for path, size_mb in center_horizon_csv_sizes.items():
        print(f"  {path}: {size_mb}MB")


if __name__ == "__main__":
    main()
