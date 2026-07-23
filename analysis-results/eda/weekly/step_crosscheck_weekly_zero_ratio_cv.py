"""
주 단위 데이터(aggregated_weekly_demand.parquet)로 Zero-Ratio/CV를 센터별로 재계산해서,
별도로 진행된 일 단위 분석과 방법론을 맞춰 비교 가능하게 만든다.

[Zero-Ratio - SKU 활동기간 기준 (주 방법론)]
센터별로 상품키(바코드+옵션코드+상품클러스터) 단위로 묶어서:
  - span_weeks = 해당 SKU의 첫 활동주(min)~마지막 활동주(max) 사이 존재해야 할 주차 수
  - count = 그 구간 내 실제 데이터가 존재하는 주 수(distinct 주시작일 - 같은 주에 여러
    지역(시도/시군구)에서 거래돼도 1주로만 센다)
  - zero_ratio = 1 - (전체 SKU의 count 합) / (전체 SKU의 span_weeks 합)  [풀링 방식]

[Zero-Ratio - 그리드 확장 기준 (Step2 방법론, 이번엔 센터별로 분리)]
Step2는 센터 합산 기준이었으나, 이번엔 (시도+시군구+상품키) x (전체 주차) 이론상
전체 그리드 대비 실제 존재 행 비율을 센터별로 따로 계산한다.

[CV]
센터별 주간 총판매수량 시계열의 std/mean.
  - A: 학습기간(21-23) 전체
  - B: 레짐전(21-2023.06) / 레짐후(23.07-23.12, 26주) / 레짐전체(21-23, 참고)
  - 위 전부 전체기간(21-24) 버전도 병행 계산(비교/참고용, leakage 없음을 명시)

[Data Leakage 방지]
학습기간(21-23) 수치가 본 기준이다. 전체기간(21-24) 수치는 참고/비교용으로만 병기하고
모델링 의사결정 근거로 쓰지 않는다.
"""

import pandas as pd

SRC_PATH = "data/final/aggregated_weekly_demand.parquet"
OUT_CSV = "analysis-results/eda/step_crosscheck_weekly_zero_ratio_cv.csv"

TRAIN_CUTOFF = pd.Timestamp("2024-01-01")
REGIME_SHIFT_DATE = pd.Timestamp("2023-07-01")

PRODUCT_KEY_COLS = ["바코드", "옵션코드", "상품클러스터"]
REGION_PRODUCT_KEY_COLS = ["시도", "시군구"] + PRODUCT_KEY_COLS
CENTERS = ["A", "B"]

# 별도(일 단위) 분석에서 이미 산출된 참고 수치 - 이번 스크립트에서 계산하지 않고 비교용으로만 사용
DAILY_ZERO_RATIO = {"A": 89.43, "B": 93.23}
DAILY_CV = {"A": 0.338, "B": 0.871}


def load_data() -> pd.DataFrame:
    return pd.read_parquet(SRC_PATH)


def sku_activity_zero_ratio(df_period: pd.DataFrame) -> float:
    g = df_period.groupby(PRODUCT_KEY_COLS)["주시작일"]
    first, last = g.min(), g.max()
    count = g.nunique()
    span_weeks = ((last - first).dt.days // 7) + 1
    return (1 - count.sum() / span_weeks.sum()) * 100


def grid_expansion_zero_ratio(df_period: pd.DataFrame) -> float:
    n_weeks = df_period["주시작일"].nunique()
    n_keys = df_period.drop_duplicates(subset=REGION_PRODUCT_KEY_COLS).shape[0]
    theoretical = n_keys * n_weeks
    actual = len(df_period)
    return (theoretical - actual) / theoretical * 100


def cv(series: pd.Series) -> float:
    return series.std() / series.mean()


def weekly_totals(df_period: pd.DataFrame) -> pd.Series:
    return df_period.groupby("주시작일")["총판매수량"].sum().sort_index()


def print_markdown_table(df: pd.DataFrame, float_cols: list) -> None:
    header = "| " + " | ".join(df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    print(header)
    print(sep)
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            v = row[col]
            if col in float_cols and isinstance(v, (int, float)):
                cells.append(f"{v:.2f}")
            else:
                cells.append(str(v))
        print("| " + " | ".join(cells) + " |")


def main() -> None:
    print(f"Loading {SRC_PATH} ...")
    df = load_data()
    print(f"Loaded {len(df):,} rows\n")

    # ---------- Zero-Ratio ----------
    zr_rows = []
    for center in CENTERS:
        d = df[df["센터"] == center]
        d_train = d[d["주시작일"] < TRAIN_CUTOFF]
        zr_rows.append({
            "센터": center,
            "SKU활동기간_학습기간(%)": sku_activity_zero_ratio(d_train),
            "SKU활동기간_전체(%)": sku_activity_zero_ratio(d),
            "그리드확장_학습기간(%)": grid_expansion_zero_ratio(d_train),
            "그리드확장_전체(%)": grid_expansion_zero_ratio(d),
        })
    zr_df = pd.DataFrame(zr_rows)

    print("=" * 70)
    print("Zero-Ratio 비교표 (센터별, SKU 활동기간 기준 vs 그리드 확장 기준)")
    print("=" * 70)
    print_markdown_table(zr_df, float_cols=[c for c in zr_df.columns if c != "센터"])

    # ---------- CV ----------
    a_train = df[(df["센터"] == "A") & (df["주시작일"] < TRAIN_CUTOFF)]
    a_full = df[df["센터"] == "A"]

    b = df[df["센터"] == "B"]
    b_train = b[b["주시작일"] < TRAIN_CUTOFF]

    b_pre_train = b_train[b_train["주시작일"] < REGIME_SHIFT_DATE]
    b_pre_full = b[b["주시작일"] < REGIME_SHIFT_DATE]  # 정의상 학습기간과 동일(레짐전은 항상 <2024)

    b_post_train = b_train[b_train["주시작일"] >= REGIME_SHIFT_DATE]  # 26주, leakage 없음
    b_post_full = b[b["주시작일"] >= REGIME_SHIFT_DATE]  # 78주, 2024 포함(참고용)

    cv_rows = [
        {"구간": "A 전체", "학습기간": cv(weekly_totals(a_train)), "전체기간": cv(weekly_totals(a_full))},
        {"구간": "B 레짐전(~23.06)", "학습기간": cv(weekly_totals(b_pre_train)), "전체기간": cv(weekly_totals(b_pre_full))},
        {"구간": "B 레짐후(23.07~, 26주/78주)", "학습기간": cv(weekly_totals(b_post_train)), "전체기간": cv(weekly_totals(b_post_full))},
        {"구간": "B 레짐전체(참고)", "학습기간": cv(weekly_totals(b_train)), "전체기간": cv(weekly_totals(b))},
    ]
    cv_df = pd.DataFrame(cv_rows)

    print("\n" + "=" * 70)
    print("CV(변동계수) 비교표 (학습기간 vs 전체기간)")
    print("=" * 70)
    print_markdown_table(cv_df, float_cols=["학습기간", "전체기간"])

    # ---------- CSV 저장 (long format으로 두 표 합쳐서 저장) ----------
    csv_rows = []
    for _, row in zr_df.iterrows():
        csv_rows.append({"지표": "Zero-Ratio(SKU활동기간)", "센터/구간": row["센터"],
                          "학습기간": row["SKU활동기간_학습기간(%)"], "전체기간": row["SKU활동기간_전체(%)"]})
        csv_rows.append({"지표": "Zero-Ratio(그리드확장)", "센터/구간": row["센터"],
                          "학습기간": row["그리드확장_학습기간(%)"], "전체기간": row["그리드확장_전체(%)"]})
    for _, row in cv_df.iterrows():
        csv_rows.append({"지표": "CV", "센터/구간": row["구간"], "학습기간": row["학습기간"], "전체기간": row["전체기간"]})
    pd.DataFrame(csv_rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료 -> {OUT_CSV}")

    # ---------- 일 단위 대비 요약 ----------
    print("\n" + "=" * 70)
    print("일 단위 vs 주 단위 비교 (학습기간 기준, SKU활동기간 방법론)")
    print("=" * 70)
    for center in CENTERS:
        weekly_zr = zr_df.loc[zr_df["센터"] == center, "SKU활동기간_학습기간(%)"].iloc[0]
        daily_zr = DAILY_ZERO_RATIO[center]
        weekly_cv = cv_df.loc[cv_df["구간"] == ("A 전체" if center == "A" else "B 레짐전체(참고)"), "학습기간"].iloc[0]
        daily_cv = DAILY_CV[center]
        print(f"\n[{center}센터]")
        print(f"  Zero-Ratio: 일 단위 {daily_zr:.2f}% -> 주 단위 {weekly_zr:.2f}% "
              f"({weekly_zr - daily_zr:+.2f}%p)")
        print(f"  CV: 일 단위 {daily_cv:.3f} -> 주 단위 {weekly_cv:.3f} "
              f"({weekly_cv / daily_cv:.2f}배)")


if __name__ == "__main__":
    main()
