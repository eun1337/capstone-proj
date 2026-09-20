"""Tableau 대시보드용 실적(y_true) vs 예측(y_pred) 단일 flat 파일 생성.

원천:
  - data/final_feature_table.parquet          -> (center_id, sku_id) 상품 디멘션
  - outputs/final_holdout/lightgbm_hurdle/prediction_lightgbm_hurdle_h1/h2/h4.parquet -> 실적+예측 팩트

산출:
  - data/tableau/dashboard_actual_vs_forecast.parquet
"""
import pandas as pd

DIM_COLS = ["center_id", "sku_id", "상품명", "규격", "입수",
            "KAN_CODE", "KAN_대분류", "KAN_중분류", "KAN_소분류"]

PRED_FILES = {
    1: "outputs/final_holdout/lightgbm_hurdle/prediction_lightgbm_hurdle_h1.parquet",
    2: "outputs/final_holdout/lightgbm_hurdle/prediction_lightgbm_hurdle_h2.parquet",
    4: "outputs/final_holdout/lightgbm_hurdle/prediction_lightgbm_hurdle_h4.parquet",
}

OUT_PATH = "data/tableau/dashboard_actual_vs_forecast.parquet"
OUT_PATH_CSV = "data/tableau/dashboard_actual_vs_forecast.csv"


def build_dimension() -> pd.DataFrame:
    dim = pd.read_parquet("data/final_feature_table.parquet", columns=DIM_COLS)
    dim = dim.drop_duplicates(subset=["center_id", "sku_id"], keep="last").copy()

    # KAN_CODE가 float으로 저장되어 앞자리 0이 소실된 문제 복구 (예: 10101.0 -> '010101')
    dim["KAN_CODE"] = dim["KAN_CODE"].astype("Int64").astype(str).str.zfill(6)

    parsed = dim["sku_id"].str.split("*", expand=True)
    dim["바코드"] = parsed[0]
    dim["옵션코드"] = parsed[1]

    return dim


def build_facts() -> pd.DataFrame:
    parts = []
    for h, path in PRED_FILES.items():
        df = pd.read_parquet(path)
        assert df["horizon"].nunique() == 1 and int(df["horizon"].iloc[0]) == h, \
            f"{path}: horizon 컬럼 값이 파일명(h{h})과 불일치"
        parts.append(df)
    facts = pd.concat(parts, ignore_index=True)
    facts["horizon_label"] = "h" + facts["horizon"].astype(str)
    return facts


def main():
    dim = build_dimension()
    facts = build_facts()

    before = len(facts)
    merged = facts.merge(dim, on=["center_id", "sku_id"], how="left", indicator=True)
    unmatched = (merged["_merge"] == "left_only").sum()
    assert len(merged) == before, "머지 중 행수 변화 발생 (fan-out 의심)"
    assert unmatched == 0, f"디멘션 매칭 안 된 행 {unmatched}건 발생"
    merged = merged.drop(columns=["_merge"])

    merged["error"] = merged["y_pred"] - merged["y_true"]
    merged["abs_error"] = merged["error"].abs()

    ordered = [
        "center_id", "sku_id", "바코드", "옵션코드", "입수",
        "상품명", "규격", "KAN_CODE", "KAN_대분류", "KAN_중분류", "KAN_소분류",
        "week_st", "horizon", "horizon_label", "target_date",
        "y_true", "y_pred", "sale_probability", "conditional_prediction",
        "error", "abs_error",
    ]
    merged = merged[ordered]

    import os
    os.makedirs("data/tableau", exist_ok=True)
    merged.to_parquet(OUT_PATH, index=False)
    merged.to_csv(OUT_PATH_CSV, index=False, encoding="utf-8-sig")

    print("=== 완료 ===")
    print("행수:", len(merged))
    print("컬럼수:", len(merged.columns))
    print("저장 경로(parquet):", OUT_PATH, f"({round(os.path.getsize(OUT_PATH) / 1024 / 1024, 2)}MB)")
    print("저장 경로(csv):", OUT_PATH_CSV, f"({round(os.path.getsize(OUT_PATH_CSV) / 1024 / 1024, 2)}MB)")
    print("\ndtypes:")
    print(merged.dtypes)
    print("\nhorizon_label 분포:")
    print(merged["horizon_label"].value_counts())
    print("\ncenter_id 분포:")
    print(merged["center_id"].value_counts())
    print("\n표본 5행:")
    print(merged.head(5).to_string())


if __name__ == "__main__":
    main()
