"""
Step3에서 식별된 "수량 부호 반전 의심 상품"(175개, 주 단위·센터무관·거래건수 기준)을
일 단위·센터별로 재식별하고, 두 가지 처리안(제외/부호수정)이 일/주/14일 단위
Zero-Ratio·CV에 미치는 영향을 정량적으로 비교한다.

[재식별 방법론 차이 - 기존 대비]
- 기존(step3_anomalous_product_candidates.csv): 바코드+옵션코드+상품클러스터 단위로
  센터 구분 없이 통합 집계, 기준은 "수량<0인 거래 건수 비율 >= 90%"(건수 기준).
- 이번(신규): 센터+상품키 단위로 완전히 분리 집계, 기준은
  "반품수량/(총판매수량+반품수량) >= 90%"(수량 기준)로 변경.
  주의: 상품키는 센터 내에서만 고유하다는 원칙(Step4)에 따라, 같은 바코드라도 A/B에서
  서로 다른 상품일 수 있어 센터별로 완전히 분리해서 봐야 한다. 기존 방식은 이 원칙을
  지키지 않고 두 센터를 합쳐서 판정했으므로, 한쪽 센터에서만 이상 패턴이 있어도 다른
  쪽 센터의 정상 거래까지 같이 제외됐을 가능성이 있다.

[Zero-Ratio·CV가 어느 컬럼에 영향받는지]
집계 파이프라인(aggregate_demand_data.py)에서 총판매수량 = sum(수량>0인 거래),
반품수량 = sum(abs(수량<0인 거래))로 분리 저장된다. 지금까지의 zero-ratio(SKU
활동기간 기준)·CV 계산은 전부 "총판매수량" 컬�럼을 기준으로 했다(순수량 아님).
따라서 "반품비율이 높다"는 것 자체가 zero-ratio/CV에 직접 영향을 주는 게 아니라,
그 상품이 원래는 정상 판매였는데 수량 부호가 반전되어 반품으로 잘못 기록됐다면
총판매수량이 실제보다 과소평가되고, 그만큼 "활동 없음(0)"으로 잘못 잡히는 주가
늘어나 zero-ratio가 과대평가된다.

[처리안 정의]
(A) 제외안: 신규 식별 리스트(구 리스트와의 합집합)에 해당하는 (센터,상품키) 조합의
    모든 원본 거래 행을 제거하고 재집계.
(B) 부호수정안: 같은 대상 (센터,상품키) 조합에 한해서만, 수량<0인 행의 부호를
    양수로 되돌린다(수량=abs(수량), 금액=abs(금액)). Step3 진단 결론(EA+특정
    카테고리 조합에서 정상 판매가 반품으로 반전 기록되는 파이프라인 버그로 추정)에
    가장 직접적으로 대응하는 보정이다 - "반품"이 아예 없었던 것처럼 지우는 대신,
    반전된 값을 원래 있었어야 할 판매로 되돌린다.

[Data Leakage 방지]
학습기간(21-23, 거래일/주시작일 < 2024-01-01)이 기준이다. B센터 레짐 컷오프는
기존과 동일하게 2023-07-01을 사용한다.
"""

import pandas as pd

DAILY_SRC = "data/final/cleaned_main_joined_cleaned_for_pred.parquet"
OLD_CANDIDATES_PATH = "analysis-results/preprocessing/step3_anomalous_product_candidates.csv"
OUT_CANDIDATES = "analysis-results/preprocessing/step_daily_anomalous_candidates.csv"
OUT_COMPARISON = "analysis-results/preprocessing/step_return_treatment_impact_comparison.csv"

TRAIN_CUTOFF = pd.Timestamp("2024-01-01")
REGIME_SHIFT_DATE = pd.Timestamp("2023-07-01")
PRODUCT_KEY_COLS = ["바코드", "옵션코드", "상품클러스터"]
CENTERS = ["A", "B"]
MIN_TXN = 10
RATIO_THRESHOLD = 0.9


def load_daily() -> pd.DataFrame:
    df = pd.read_parquet(DAILY_SRC)
    df["거래일"] = pd.to_datetime(df["거래일"])
    df["주시작일"] = df["거래일"] - pd.to_timedelta(df["거래일"].dt.dayofweek, unit="D")
    return df


# ---------------------------------------------------------------------------
# Step 1: 일 단위 재식별
# ---------------------------------------------------------------------------
def identify_daily_candidates(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["센터"] + PRODUCT_KEY_COLS)
    summary = g.agg(
        거래건수=("수량", "count"),
        총판매수량=("수량", lambda s: s[s > 0].sum()),
        반품수량=("수량", lambda s: (-s[s < 0]).sum()),
    ).reset_index()
    summary["반품비율"] = summary["반품수량"] / (summary["총판매수량"] + summary["반품수량"])
    cand = summary[(summary["거래건수"] >= MIN_TXN) & (summary["반품비율"] >= RATIO_THRESHOLD)].copy()
    return cand.sort_values("거래건수", ascending=False), summary


def compare_with_old(df: pd.DataFrame, new_cand: pd.DataFrame) -> set:
    """반환값은 (센터,바코드,옵션코드,상품클러스터) 튜플 집합 - 상품키는 센터 내에서만
    고유하다는 원칙에 따라, 실제 처리(제외/부호수정)는 항상 센터를 포함해 적용한다."""
    old = pd.read_csv(OLD_CANDIDATES_PATH, dtype={"바코드": str})
    old_keys = set(map(tuple, old[PRODUCT_KEY_COLS].values))
    new_keys_no_center = set(map(tuple, new_cand[PRODUCT_KEY_COLS].values))
    new_keys_with_center = set(map(tuple, new_cand[["센터"] + PRODUCT_KEY_COLS].values))

    overlap = old_keys & new_keys_no_center
    only_old = old_keys - new_keys_no_center
    only_new = new_keys_no_center - old_keys

    print(f"기존(주 단위, 센터무관) 후보: {len(old_keys)}개")
    print(f"신규(일 단위, 센터별) 후보: {len(new_cand)}개 (고유 상품키 {len(new_keys_no_center)}개, "
          f"센터별 분포: {dict(new_cand['센터'].value_counts())})")
    print(f"겹치는 상품키: {len(overlap)}개")
    print(f"기존에만 있고 신규엔 없음: {len(only_old)}개 (건수 기준->수량 기준 전환 시 기준 미달)")
    print(f"신규에만 있고 기존엔 없음: {len(only_new)}개 (센터 분리 덕분에 새로 드러남)")

    # 기존(센터무관) 리스트는 실제 데이터에 존재하는 센터로 확장한다(원래 파이프라인이
    # 센터 구분 없이 두 센터 모두에서 제외했던 것과 동일하게 취급).
    existing_center_key = set(map(tuple, df[["센터"] + PRODUCT_KEY_COLS].drop_duplicates().values))
    old_keys_expanded = {ck for ck in existing_center_key if ck[1:] in old_keys}

    union_keys = old_keys_expanded | new_keys_with_center
    print(f"최종 처리 대상(합집합, 센터+상품키 기준): {len(union_keys)}개")
    return union_keys


# ---------------------------------------------------------------------------
# Step 2: 영향도 정량화 (처리 전)
# ---------------------------------------------------------------------------
def quantify_impact(df: pd.DataFrame, union_keys: set) -> None:
    df = df.copy()
    df["_flagged"] = df.set_index(["센터"] + PRODUCT_KEY_COLS).index.isin(union_keys)

    print("\n영향도 (센터별, 처리 대상 상품키 = 구+신규 합집합 기준):")
    for center in CENTERS:
        d = df[df["센터"] == center]
        flagged = d[d["_flagged"]]
        row_pct = len(flagged) / len(d) * 100
        sales_before = d.loc[d["수량"] > 0, "수량"].sum()
        sales_flagged = flagged.loc[flagged["수량"] > 0, "수량"].sum()
        returns_before = (-d.loc[d["수량"] < 0, "수량"]).sum()
        returns_flagged = (-flagged.loc[flagged["수량"] < 0, "수량"]).sum()
        print(f"  [{center}센터] 대상 행 비율: {len(flagged):,}/{len(d):,} ({row_pct:.3f}%) | "
              f"총판매수량 중 대상 비중: {sales_flagged / sales_before:.3%} | "
              f"반품수량 중 대상 비중: {returns_flagged / returns_before:.3%}")


# ---------------------------------------------------------------------------
# 처리안 적용
# ---------------------------------------------------------------------------
def apply_exclude(df: pd.DataFrame, union_keys: set) -> pd.DataFrame:
    mask = df.set_index(["센터"] + PRODUCT_KEY_COLS).index.isin(union_keys)
    return df[~mask].copy()


def apply_sign_fix(df: pd.DataFrame, union_keys: set) -> pd.DataFrame:
    df = df.copy()
    mask = df.set_index(["센터"] + PRODUCT_KEY_COLS).index.isin(union_keys) & (df["수량"] < 0)
    df.loc[mask, "수량"] = df.loc[mask, "수량"].abs()
    df.loc[mask, "금액"] = df.loc[mask, "금액"].abs()
    return df


# ---------------------------------------------------------------------------
# Zero-Ratio / CV 계산 (일/주/14일 공통 로직)
# ---------------------------------------------------------------------------
def build_buckets(df_period: pd.DataFrame, date_col: str, bucket_size: int) -> dict:
    """date_col의 distinct 값을 정렬해 bucket_size개씩 묶는다. 마지막 불완전 버킷은 제외."""
    vals_sorted = sorted(df_period[date_col].unique())
    n = len(vals_sorted)
    val_to_bucket = {v: i // bucket_size for i, v in enumerate(vals_sorted)}
    n_full = n // bucket_size

    d = df_period.copy()
    d["_bucket"] = d[date_col].map(val_to_bucket)
    d_valid = d[d["_bucket"] < n_full]
    return {"df": d_valid, "vals_sorted": vals_sorted, "val_to_bucket": val_to_bucket, "n_full": n_full}


def sku_activity_zero_ratio(df_valid: pd.DataFrame) -> float:
    g = df_valid.groupby(PRODUCT_KEY_COLS)["_bucket"]
    first, last = g.min(), g.max()
    count = g.nunique()
    span = last - first + 1
    return (1 - count.sum() / span.sum()) * 100


def bucket_totals(df_valid: pd.DataFrame) -> pd.Series:
    df_valid = df_valid.copy()
    df_valid["_sale"] = df_valid["수량"].where(df_valid["수량"] > 0, 0)
    return df_valid.groupby("_bucket")["_sale"].sum().sort_index()


def cv(series: pd.Series) -> float:
    return series.std() / series.mean()


def regime_bucket_index(bucket_info: dict, date_col_is_week: bool) -> int:
    vals_sorted = bucket_info["vals_sorted"]
    first_post = next(v for v in vals_sorted if pd.Timestamp(v) >= REGIME_SHIFT_DATE)
    return bucket_info["val_to_bucket"][first_post]


def compute_metrics(df: pd.DataFrame, unit: str) -> dict:
    """unit: 'daily'(1일), 'weekly'(7일), 'biweekly'(14일)"""
    date_col = "거래일" if unit == "daily" else "주시작일"
    bucket_size = {"daily": 1, "weekly": 1, "biweekly": 2}[unit]
    # 주/14일 단위는 거래일이 아닌 주시작일 기준 그리드에서 1주/2주씩 묶는다.
    # 일 단위는 거래일 자체를 1일 버킷으로 그대로 쓴다.

    zr_rows = []
    cv_rows = {}
    for center in CENTERS:
        d = df[df["센터"] == center]
        d_train = d[d[date_col] < TRAIN_CUTOFF] if unit == "daily" else d[d["주시작일"] < TRAIN_CUTOFF]

        info_train = build_buckets(d_train, date_col, bucket_size)
        zr_train = sku_activity_zero_ratio(info_train["df"])
        zr_rows.append({"센터": center, "학습기간(%)": zr_train})

        if center == "A":
            cv_rows["A 전체"] = cv(bucket_totals(info_train["df"]))
        else:
            regime_idx = regime_bucket_index(info_train, unit != "daily")
            b_df = info_train["df"]
            pre = b_df[b_df["_bucket"] < regime_idx]
            post = b_df[b_df["_bucket"] >= regime_idx]
            cv_rows["B 레짐전(~23.06)"] = cv(bucket_totals(pre))
            cv_rows["B 레짐후(23.07~)"] = cv(bucket_totals(post))
            cv_rows["B 레짐전체(참고)"] = cv(bucket_totals(b_df))

    return {"zr": {r["센터"]: r["학습기간(%)"] for r in zr_rows}, "cv": cv_rows}


def print_markdown_table(rows: list, columns: list) -> None:
    print("| " + " | ".join(columns) + " |")
    print("|" + "|".join(["---"] * len(columns)) + "|")
    for r in rows:
        cells = []
        for c in columns:
            v = r.get(c)
            cells.append(f"{v:.2f}" if isinstance(v, float) else ("" if v is None else str(v)))
        print("| " + " | ".join(cells) + " |")


def main() -> None:
    print(f"Loading {DAILY_SRC} ...")
    df = load_daily()
    print(f"Loaded {len(df):,} rows\n")

    print("=" * 70)
    print("[Step 1] 일 단위·센터별 재식별")
    print("=" * 70)
    new_cand, _ = identify_daily_candidates(df)
    new_cand.to_csv(OUT_CANDIDATES, index=False, encoding="utf-8-sig")
    print(f"저장 완료 -> {OUT_CANDIDATES}\n")
    union_keys = compare_with_old(df, new_cand)

    print("\n" + "=" * 70)
    print("[Step 2] 영향도 정량화 (처리 전)")
    print("=" * 70)
    print("Zero-Ratio/CV 계산에 쓰인 컬럼: 총판매수량 (수량>0인 거래의 합, 순수량 아님)")
    print("-> 부호반전으로 정상판매가 반품 처리되면 총판매수량이 과소평가되어 zero-ratio가 과대평가됨\n")
    quantify_impact(df, union_keys)

    print("\n" + "=" * 70)
    print("[Step 3] 시나리오별 재계산")
    print("=" * 70)
    scenarios = {
        "기존(처리안함)": df,
        "A_제외안": apply_exclude(df, union_keys),
        "B_부호수정안": apply_sign_fix(df, union_keys),
    }

    all_results = {}
    for name, sdf in scenarios.items():
        all_results[name] = {
            "daily": compute_metrics(sdf, "daily"),
            "weekly": compute_metrics(sdf, "weekly"),
            "biweekly": compute_metrics(sdf, "biweekly"),
        }
        print(f"{name} 계산 완료")

    # ---------------- 비교표 출력 ----------------
    print("\n" + "=" * 70)
    print("[Step 4] Zero-Ratio 비교표 (학습기간, SKU 활동기간 기준)")
    print("=" * 70)
    for unit, unit_label in [("daily", "일(1일)"), ("weekly", "주(7일)"), ("biweekly", "14일(격주)")]:
        print(f"\n--- {unit_label} ---")
        rows = []
        for center in CENTERS:
            rows.append({
                "센터": center,
                "기존(처리안함)": all_results["기존(처리안함)"][unit]["zr"][center],
                "A_제외안": all_results["A_제외안"][unit]["zr"][center],
                "B_부호수정안": all_results["B_부호수정안"][unit]["zr"][center],
            })
        print_markdown_table(rows, ["센터", "기존(처리안함)", "A_제외안", "B_부호수정안"])

    print("\n" + "=" * 70)
    print("[Step 4] CV 비교표 (학습기간)")
    print("=" * 70)
    for unit, unit_label in [("daily", "일(1일)"), ("weekly", "주(7일)"), ("biweekly", "14일(격주)")]:
        print(f"\n--- {unit_label} ---")
        rows = []
        for seg in ["A 전체", "B 레짐전(~23.06)", "B 레짐후(23.07~)", "B 레짐전체(참고)"]:
            rows.append({
                "구간": seg,
                "기존(처리안함)": all_results["기존(처리안함)"][unit]["cv"].get(seg),
                "A_제외안": all_results["A_제외안"][unit]["cv"].get(seg),
                "B_부호수정안": all_results["B_부호수정안"][unit]["cv"].get(seg),
            })
        print_markdown_table(rows, ["구간", "기존(처리안함)", "A_제외안", "B_부호수정안"])

    # ---------------- CSV 저장 ----------------
    csv_rows = []
    for unit, unit_label in [("daily", "일(1일)"), ("weekly", "주(7일)"), ("biweekly", "14일(격주)")]:
        for center in CENTERS:
            csv_rows.append({
                "지표": "Zero-Ratio", "단위": unit_label, "센터/구간": center,
                "기존(처리안함)": all_results["기존(처리안함)"][unit]["zr"][center],
                "A_제외안": all_results["A_제외안"][unit]["zr"][center],
                "B_부호수정안": all_results["B_부호수정안"][unit]["zr"][center],
            })
        for seg in ["A 전체", "B 레짐전(~23.06)", "B 레짐후(23.07~)", "B 레짐전체(참고)"]:
            csv_rows.append({
                "지표": "CV", "단위": unit_label, "센터/구간": seg,
                "기존(처리안함)": all_results["기존(처리안함)"][unit]["cv"].get(seg),
                "A_제외안": all_results["A_제외안"][unit]["cv"].get(seg),
                "B_부호수정안": all_results["B_부호수정안"][unit]["cv"].get(seg),
            })
    pd.DataFrame(csv_rows).to_csv(OUT_COMPARISON, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료 -> {OUT_COMPARISON}")


if __name__ == "__main__":
    main()
