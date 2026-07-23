"""
Step_crosscheck 확장: 주 단위 집계 데이터를 N주(1주=주 단위, 2주=14일/격주 단위)로 묶어
Zero-Ratio(SKU 활동기간 기준)/CV를 센터별로 계산하고, 일/주/14일 세 단위를 한 표로 비교한다.

14일(격주) 단위를 검토하는 이유: 7의 배수라 요일 정렬이 유지되면서(주중/주말 패턴이 안
깨짐) 노이즈를 더 줄일 수 있는 후보. A센터는 Step6 ACF/PACF에서 단기 lag(1~3주)
자기상관이 관측된 바 있어 이 주기와 맞아떨어질 가능성이 있다.

[버킷 구성 방법]
센터별로 그 센터의 첫 관측 주부터 N주씩 묶어서 버킷을 만든다. 원본 일 단위로 다시
내려가지 않고, 이미 있는 주 단위 집계(aggregated_weekly_demand.parquet)를 그대로
N개씩 합산한다. 주차 수가 N의 배수가 아니면 마지막 불완전 버킷은 계산에서 제외한다.

[Zero-Ratio - SKU 활동기간 기준]
이전 주 단위 스크립트(step_crosscheck_weekly_zero_ratio_cv.py)와 동일한 방법론을
버킷 단위로 그대로 적용: span = 첫 활동버킷~마지막 활동버킷 사이 버킷 수,
count = 실제 활동 버킷 수(distinct), zero_ratio = 1 - sum(count)/sum(span) (풀링).

[레짐 컷오프 - 버킷 경계 근사]
2주 버킷에서는 2023-07-01(레짐 전환)이 버킷 경계와 정확히 안 맞을 수 있다. 이 경우
레짐 전환 이후 첫 주가 속한 버킷 전체를 "레짐후"로 분류한다(그 버킷의 시작일이 레짐
전환일에 가장 가까운 구간 경계이기 때문 - B센터 실측 결과 해당 버킷은 2023-06-26주
+2023-07-03주가 섞여 있고, 후자를 포함하는 쪽이 더 가까운 경계라 "레짐후"로 처리).

[Data Leakage 방지]
학습기간(21-23, 주시작일 < 2024-01-01)이 기준이고, 전체기간(21-24)은 참고/비교용으로만
병기한다.
"""

import pandas as pd

SRC_PATH = "data/final/aggregated_weekly_demand.parquet"
OUT_CSV = "analysis-results/eda/step_crosscheck_biweekly_zero_ratio_cv.csv"

TRAIN_CUTOFF = pd.Timestamp("2024-01-01")
REGIME_SHIFT_DATE = pd.Timestamp("2023-07-01")

PRODUCT_KEY_COLS = ["바코드", "옵션코드", "상품클러스터"]
CENTERS = ["A", "B"]

# 별도(일 단위) 분석 참고 수치 - 이번 스크립트에서 계산하지 않고 비교용으로만 사용
DAILY_ZERO_RATIO = {"A": 89.43, "B": 93.23}
DAILY_CV = {"A": 0.338, "B": 0.871}


def load_data() -> pd.DataFrame:
    return pd.read_parquet(SRC_PATH)


def build_buckets(df_period: pd.DataFrame, bucket_weeks: int) -> dict:
    """센터 period 데이터를 bucket_weeks주씩 묶는다. 마지막 불완전 버킷은 제외."""
    weeks_sorted = sorted(df_period["주시작일"].unique())
    n_weeks = len(weeks_sorted)
    week_to_bucket = {w: i // bucket_weeks for i, w in enumerate(weeks_sorted)}
    n_full_buckets = n_weeks // bucket_weeks
    remainder = n_weeks % bucket_weeks

    d = df_period.copy()
    d["_bucket"] = d["주시작일"].map(week_to_bucket)
    d_valid = d[d["_bucket"] < n_full_buckets]

    return {
        "df": d_valid, "weeks_sorted": weeks_sorted, "week_to_bucket": week_to_bucket,
        "n_weeks": n_weeks, "n_full_buckets": n_full_buckets, "remainder_weeks": remainder,
    }


def sku_activity_zero_ratio(df_valid: pd.DataFrame) -> float:
    g = df_valid.groupby(PRODUCT_KEY_COLS)["_bucket"]
    first, last = g.min(), g.max()
    count = g.nunique()
    span = last - first + 1
    return (1 - count.sum() / span.sum()) * 100


def bucket_totals(df_valid: pd.DataFrame) -> pd.Series:
    return df_valid.groupby("_bucket")["총판매수량"].sum().sort_index()


def cv(series: pd.Series) -> float:
    return series.std() / series.mean()


def regime_split(bucket_info: dict) -> int:
    weeks_sorted = bucket_info["weeks_sorted"]
    first_post_week = next(w for w in weeks_sorted if pd.Timestamp(w) >= REGIME_SHIFT_DATE)
    return bucket_info["week_to_bucket"][first_post_week]


def print_markdown_table(df: pd.DataFrame, float_cols: list) -> None:
    print("| " + " | ".join(df.columns) + " |")
    print("|" + "|".join(["---"] * len(df.columns)) + "|")
    for _, row in df.iterrows():
        cells = []
        for col in df.columns:
            v = row[col]
            cells.append(f"{v:.2f}" if col in float_cols and isinstance(v, (int, float)) else str(v))
        print("| " + " | ".join(cells) + " |")


def compute_for_unit(df: pd.DataFrame, bucket_weeks: int, unit_label: str) -> dict:
    result = {"unit": unit_label}

    zr_rows = []
    cv_rows_train = {}
    cv_rows_full = {}
    b_post_bucket_count = None

    for center in CENTERS:
        d = df[df["센터"] == center]
        d_train = d[d["주시작일"] < TRAIN_CUTOFF]

        train_info = build_buckets(d_train, bucket_weeks)
        full_info = build_buckets(d, bucket_weeks)

        zr_train = sku_activity_zero_ratio(train_info["df"])
        zr_full = sku_activity_zero_ratio(full_info["df"])
        zr_rows.append({"센터": center, "학습기간(%)": zr_train, "전체기간(%)": zr_full})

        if center == "A":
            cv_rows_train["A 전체"] = cv(bucket_totals(train_info["df"]))
            cv_rows_full["A 전체"] = cv(bucket_totals(full_info["df"]))
        else:
            regime_bucket_train = regime_split(train_info)
            regime_bucket_full = regime_split(full_info)

            d_train_b = train_info["df"]
            d_full_b = full_info["df"]

            pre_train = d_train_b[d_train_b["_bucket"] < regime_bucket_train]
            post_train = d_train_b[d_train_b["_bucket"] >= regime_bucket_train]
            pre_full = d_full_b[d_full_b["_bucket"] < regime_bucket_full]
            post_full = d_full_b[d_full_b["_bucket"] >= regime_bucket_full]

            cv_rows_train["B 레짐전(~23.06)"] = cv(bucket_totals(pre_train))
            cv_rows_train["B 레짐후(23.07~)"] = cv(bucket_totals(post_train))
            cv_rows_train["B 레짐전체(참고)"] = cv(bucket_totals(d_train_b))
            cv_rows_full["B 레짐전(~23.06)"] = cv(bucket_totals(pre_full))
            cv_rows_full["B 레짐후(23.07~)"] = cv(bucket_totals(post_full))
            cv_rows_full["B 레짐전체(참고)"] = cv(bucket_totals(d_full_b))

            b_post_bucket_count = post_train["_bucket"].nunique()
            b_post_week_count = train_info["n_weeks"] - regime_bucket_train * bucket_weeks

    zr_df = pd.DataFrame(zr_rows)
    cv_df = pd.DataFrame([
        {"구간": k, "학습기간": cv_rows_train[k], "전체기간": cv_rows_full[k]}
        for k in ["A 전체", "B 레짐전(~23.06)", "B 레짐후(23.07~)", "B 레짐전체(참고)"]
    ])

    result["zr_df"] = zr_df
    result["cv_df"] = cv_df
    result["b_post_bucket_count"] = b_post_bucket_count
    result["b_post_week_count"] = b_post_week_count
    return result


def main() -> None:
    print(f"Loading {SRC_PATH} ...")
    df = load_data()
    print(f"Loaded {len(df):,} rows\n")

    weekly_result = compute_for_unit(df, 1, "주(7일)")
    biweekly_result = compute_for_unit(df, 2, "14일(격주)")

    print("=" * 70)
    print("[검증용] 주(7일) 단위 재계산 - 이전 스크립트 결과와 동일해야 함")
    print("=" * 70)
    print_markdown_table(weekly_result["zr_df"], float_cols=["학습기간(%)", "전체기간(%)"])
    print()
    print_markdown_table(weekly_result["cv_df"], float_cols=["학습기간", "전체기간"])

    print("\n" + "=" * 70)
    print("[신규] 14일(격주) 단위 Zero-Ratio 비교표 (SKU 활동기간 기준)")
    print("=" * 70)
    print_markdown_table(biweekly_result["zr_df"], float_cols=["학습기간(%)", "전체기간(%)"])

    print("\n" + "=" * 70)
    print("[신규] 14일(격주) 단위 CV 비교표")
    print("=" * 70)
    print_markdown_table(biweekly_result["cv_df"], float_cols=["학습기간", "전체기간"])

    print("\n" + "=" * 70)
    print("[중요] B센터 레짐후 구간 표본 수 확인")
    print("=" * 70)
    print(f"주(7일) 단위: 26개 구간(주)")
    print(f"14일(격주) 단위: {biweekly_result['b_post_bucket_count']}개 구간 "
          f"(원 주차 기준 {biweekly_result['b_post_week_count']}주 해당)")
    if biweekly_result["b_post_bucket_count"] < 15:
        print(f"-> 표본 {biweekly_result['b_post_bucket_count']}개는 CV/분포 추정에 상당히 얇은 수준.")
        print("   Step6에서 26주(주 단위)도 이미 ACF lag=52 계산 불가 수준이었는데, 14일 단위로")
        print("   묶으면 표본이 절반(13개 안팍)으로 더 줄어 통계적 신뢰도가 한층 더 낮아진다.")

    # ---------- 통합 비교표 (일/주/14일) ----------
    print("\n" + "=" * 70)
    print("[통합] 일 / 주(7일) / 14일(격주) Zero-Ratio 비교 (학습기간, SKU 활동기간 기준)")
    print("=" * 70)
    combined_zr = pd.DataFrame([
        {
            "센터": c,
            "일(1일)": DAILY_ZERO_RATIO[c],
            "주(7일)": weekly_result["zr_df"].loc[weekly_result["zr_df"]["센터"] == c, "학습기간(%)"].iloc[0],
            "14일(격주)": biweekly_result["zr_df"].loc[biweekly_result["zr_df"]["센터"] == c, "학습기간(%)"].iloc[0],
        }
        for c in CENTERS
    ])
    print_markdown_table(combined_zr, float_cols=["일(1일)", "주(7일)", "14일(격주)"])

    print("\n" + "=" * 70)
    print("[통합] 일 / 주(7일) / 14일(격주) CV 비교 (학습기간)")
    print("=" * 70)
    weekly_cv_map = weekly_result["cv_df"].set_index("구간")["학습기간"]
    biweekly_cv_map = biweekly_result["cv_df"].set_index("구간")["학습기간"]
    combined_cv = pd.DataFrame([
        {"구간": "A 전체", "일(1일)": DAILY_CV["A"], "주(7일)": weekly_cv_map["A 전체"], "14일(격주)": biweekly_cv_map["A 전체"]},
        {"구간": "B 레짐전체(참고)", "일(1일)": DAILY_CV["B"], "주(7일)": weekly_cv_map["B 레짐전체(참고)"], "14일(격주)": biweekly_cv_map["B 레짐전체(참고)"]},
        {"구간": "B 레짐전(~23.06)", "일(1일)": None, "주(7일)": weekly_cv_map["B 레짐전(~23.06)"], "14일(격주)": biweekly_cv_map["B 레짐전(~23.06)"]},
        {"구간": "B 레짐후(23.07~)", "일(1일)": None, "주(7일)": weekly_cv_map["B 레짐후(23.07~)"], "14일(격주)": biweekly_cv_map["B 레짐후(23.07~)"]},
    ])
    print_markdown_table(combined_cv, float_cols=["일(1일)", "주(7일)", "14일(격주)"])

    # ---------- CSV 저장 ----------
    csv_rows = []
    for _, row in biweekly_result["zr_df"].iterrows():
        csv_rows.append({"지표": "Zero-Ratio(SKU활동기간,14일)", "센터/구간": row["센터"],
                          "학습기간": row["학습기간(%)"], "전체기간": row["전체기간(%)"]})
    for _, row in biweekly_result["cv_df"].iterrows():
        csv_rows.append({"지표": "CV(14일)", "센터/구간": row["구간"], "학습기간": row["학습기간"], "전체기간": row["전체기간"]})
    csv_rows.append({"지표": "B레짐후_구간수(14일)", "센터/구간": "B", "학습기간": biweekly_result["b_post_bucket_count"], "전체기간": None})
    pd.DataFrame(csv_rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료 -> {OUT_CSV}")


if __name__ == "__main__":
    main()
