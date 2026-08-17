"""
econ_indicator_screening.py

설명: 외부 경제지표 후보를 생성하고 train 데이터에서 통계적으로 스크리닝함.
- CPI와 소비자심리지수(CCSI)의 시점별 후보 피처를 생성함.
- 센터별 주간 총수요를 타깃으로 Spearman 상관계수와 Mutual Information을 계산함.
- 원본, 추세 제거(detrend), 전년동월대비(YoY) 결과를 비교하여 후보를 평가함.
- 평가 데이터 누수를 방지하기 위해 train 구간만 스크리닝에 사용함.
- 후보 피처 테이블과 스크리닝 결과를 parquet 및 CSV로 저장함.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression

BASE_DIR = Path(__file__).resolve().parents[4]
FEATURE_TABLE_PATH = BASE_DIR / "data" / "ml" / "splits" / "feature_table_final.parquet"
CPI_PATH = BASE_DIR / "data" / "external" / "economic" / "consumer_price_index_national.csv"
CCSI_PATH = BASE_DIR / "data" / "external" / "economic" / "consumer_sentiment_index_national.csv"

EXP_DIR = BASE_DIR / "data" / "ml" / "experiments" / "economic_indicators"
REPORT_DIR = EXP_DIR / "reports"
CANDIDATES_OUT_PATH = EXP_DIR / "econ_candidates_train.parquet"
REPORT_OUT_PATH = REPORT_DIR / "econ_indicator_screening_report.csv"

CENTER_COL = "center_id"
WEEK_COL = "week_st"
QTY_COL = "qty"

# 각 경제지표 후보가 현재 시점에서 몇 년 전, 몇 개월 전 값을 사용할지 정의함.
# 예: y1_prev는 현재 연도에서 1년 전으로 이동한 뒤, 해당 월의 전월 값을 사용함.
FEATURE_SPECS = [
    ("lag_m1", 0, -1),
    ("y1_prev", -1, -1),
    ("y1_curr", -1, 0),
    ("y1_next", -1, 1),
    ("y2_prev", -2, -1),
    ("y2_curr", -2, 0),
    ("y2_next", -2, 1),
    ("y3_prev", -3, -1),
    ("y3_curr", -3, 0),
    ("y3_next", -3, 1),
]

RNG_SEED = 42

# 2024 Final Holdout 기간의 경제지표가 screening 통계에 들어가지 않도록 상한을 둠.
SERIES_CUTOFF_YM = (2023, 12)


def shift_year_month(year: int, month: int, offset: int) -> tuple[int, int]:
    total = year * 12 + (month - 1) + offset
    return total // 12, total % 12 + 1


def load_monthly_series(path: Path, value_col: str) -> pd.Series:
    """(연,월) -> 값 Series(MultiIndex 아닌 (year,month) 튜플 인덱스)를 시간순 정렬해 반환."""
    df = pd.read_csv(path)
    df = df.sort_values(["년", "월"]).reset_index(drop=True)
    idx = list(zip(df["년"].astype(int), df["월"].astype(int)))
    s = pd.Series(df[value_col].astype(float).to_numpy(), index=pd.Index(idx))
    if s.index.duplicated().any():
        raise ValueError(f"{path}: (연,월) 중복 존재")
    return s


def linear_detrend(s: pd.Series) -> pd.Series:
    """시간순 정렬된 월별 시계열을 t=0..N-1 선형회귀로 적합, 잔차를 반환."""
    t = np.arange(len(s), dtype=float)
    coef = np.polyfit(t, s.to_numpy(), deg=1)
    fitted = np.polyval(coef, t)
    resid = s.to_numpy() - fitted
    return pd.Series(resid, index=s.index)


def yoy_pct_change(s: pd.Series) -> pd.Series:
    """전년동월대비 증감률(%). 첫 12개월(비교 대상 없음)은 NaN."""
    vals = s.to_numpy()
    out = np.full(len(vals), np.nan)
    out[12:] = (vals[12:] - vals[:-12]) / vals[:-12] * 100
    return pd.Series(out, index=s.index)


def build_candidate_features(series: pd.Series, prefix: str, year_months: pd.DataFrame) -> pd.DataFrame:
    """year_months(고유 (year,month) 행)마다 10개 후보 컬럼을 계산."""
    lookup = series.to_dict()
    out = {}
    for suffix, year_off, month_off in FEATURE_SPECS:
        col = f"{prefix}_{suffix}"
        vals = []
        for year, month in zip(year_months["year"], year_months["month"]):
            base_year = year + year_off
            y, m = shift_year_month(base_year, month, month_off)
            vals.append(lookup.get((y, m), np.nan))
        out[col] = vals
    return pd.DataFrame(out, index=year_months.index)


def build_candidates() -> pd.DataFrame:
    print("[1/3] feature_table_final.parquet 로드 (split=='train'만, 필요 컬럼만)")
    df = pd.read_parquet(
        FEATURE_TABLE_PATH, columns=[CENTER_COL, WEEK_COL, QTY_COL, "split"]
    )
    df = df[df["split"] == "train"].copy()
    print(f"  train 행수: {len(df):,}  (A/B 센터: {sorted(df[CENTER_COL].unique())})")

    print("[2/3] 센터x주 Roll-up 타겟(qty 합계) 집계 (P4 공식과 동일: groupby(center,week).sum())")
    target = df.groupby([CENTER_COL, WEEK_COL], as_index=False, observed=True)[QTY_COL].sum()
    target = target.rename(columns={QTY_COL: "qty_rollup"})
    target["year"] = target[WEEK_COL].dt.year
    target["month"] = target[WEEK_COL].dt.month
    print(f"  센터x주 행수: {len(target):,}  기간: {target[WEEK_COL].min().date()} ~ {target[WEEK_COL].max().date()}")

    print("[3/3] CPI/소비자심리지수 로드 및 20개 후보 피처 생성 (원본/선형detrend잔차/YoY 3버전)")
    cpi_raw = load_monthly_series(CPI_PATH, "소비자물가지수")
    ccsi_raw = load_monthly_series(CCSI_PATH, "소비자심리지수")
    cpi_raw = cpi_raw[[ym <= SERIES_CUTOFF_YM for ym in cpi_raw.index]]
    ccsi_raw = ccsi_raw[[ym <= SERIES_CUTOFF_YM for ym in ccsi_raw.index]]

    year_months = target[["year", "month"]].drop_duplicates().reset_index(drop=True)

    versions = {
        "": {"cpi": cpi_raw, "ccsi": ccsi_raw},
        "_detrend": {"cpi": linear_detrend(cpi_raw), "ccsi": linear_detrend(ccsi_raw)},
        "_yoy": {"cpi": yoy_pct_change(cpi_raw), "ccsi": yoy_pct_change(ccsi_raw)},
    }

    feat_blocks = [year_months]
    for suffix, series_map in versions.items():
        for prefix, series in series_map.items():
            block = build_candidate_features(series, prefix, year_months)
            block.columns = [f"{c}{suffix}" for c in block.columns]
            feat_blocks.append(block)
    year_month_features = pd.concat(feat_blocks, axis=1)

    candidates = target.merge(year_month_features, on=["year", "month"], how="left")
    return candidates


def pairwise_spearman(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 3:
        return np.nan, np.nan, n
    rho, p = spearmanr(x[mask], y[mask])
    return float(rho), float(p), n


def pairwise_mi(x: pd.Series, y: pd.Series) -> tuple[float, int]:
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 3:
        return np.nan, n
    X = x[mask].to_numpy().reshape(-1, 1)
    yy = y[mask].to_numpy()
    mi = mutual_info_regression(X, yy, random_state=RNG_SEED)[0]
    return float(mi), n


def screen(candidates: pd.DataFrame) -> pd.DataFrame:
    raw_cols = [f"{p}_{s}" for p in ("cpi", "ccsi") for s, _, _ in FEATURE_SPECS]
    groups = {
        "pooled(A+B)": candidates,
        "A": candidates[candidates[CENTER_COL] == "A"],
        "B": candidates[candidates[CENTER_COL] == "B"],
    }

    rows = []
    for group_name, gdf in groups.items():
        target = gdf["qty_rollup"]
        for col in raw_cols:
            indicator = "cpi" if col.startswith("cpi_") else "ccsi"
            rho_raw, p_raw, n_raw = pairwise_spearman(gdf[col], target)
            rho_dt, p_dt, n_dt = pairwise_spearman(gdf[f"{col}_detrend"], target)
            rho_yoy, p_yoy, n_yoy = pairwise_spearman(gdf[f"{col}_yoy"], target)
            mi_raw, n_mi = pairwise_mi(gdf[col], target)
            rows.append({
                "indicator": indicator,
                "candidate": col,
                "group": group_name,
                "n": n_raw,
                "spearman_raw": round(rho_raw, 4) if pd.notna(rho_raw) else np.nan,
                "spearman_raw_p": round(p_raw, 4) if pd.notna(p_raw) else np.nan,
                "spearman_detrend_linear": round(rho_dt, 4) if pd.notna(rho_dt) else np.nan,
                "spearman_detrend_linear_p": round(p_dt, 4) if pd.notna(p_dt) else np.nan,
                "n_detrend": n_dt,
                "spearman_yoy": round(rho_yoy, 4) if pd.notna(rho_yoy) else np.nan,
                "spearman_yoy_p": round(p_yoy, 4) if pd.notna(p_yoy) else np.nan,
                "n_yoy": n_yoy,
                "mutual_info": round(mi_raw, 4) if pd.notna(mi_raw) else np.nan,
            })

    report = pd.DataFrame(rows)
    col_order = [
        "indicator", "candidate", "group", "n",
        "spearman_raw", "spearman_raw_p",
        "spearman_detrend_linear", "spearman_detrend_linear_p", "n_detrend",
        "spearman_yoy", "spearman_yoy_p", "n_yoy",
        "mutual_info",
    ]
    return report[col_order]


def summarize_recommendations(report: pd.DataFrame) -> None:
    pooled = report[report["group"] == "pooled(A+B)"].copy()
    pooled["abs_spearman_raw"] = pooled["spearman_raw"].abs()
    pooled["abs_spearman_detrend"] = pooled["spearman_detrend_linear"].abs()
    pooled["abs_spearman_yoy"] = pooled["spearman_yoy"].abs()

    # 단순한 시간 추세 때문에 상관이 높게 나타난 후보를 줄이기 위해,
    # 원본 값과 수요의 상관이 유의하고 detrend 또는 YoY에서도 유의한 후보만 남김.
    survives_detrend = (
        (pooled["spearman_detrend_linear_p"] < 0.05)
        | (pooled["spearman_yoy_p"] < 0.05)
    )
    significant_raw = pooled["spearman_raw_p"] < 0.05

    candidate_pool = pooled[significant_raw & survives_detrend].copy()
    candidate_pool["score"] = (
        candidate_pool["abs_spearman_raw"]
        + candidate_pool[["abs_spearman_detrend", "abs_spearman_yoy"]].max(axis=1)
        + candidate_pool["mutual_info"]
    )
    top = candidate_pool.sort_values("score", ascending=False).head(5)

    print()
    print("=" * 80)
    print("[추천 최종 후보] pooled(A+B) 기준, 원본상관 유의(p<0.05) AND (detrend상관 또는 YoY상관 중 하나 이상 p<0.05)")
    print("  선정 기준: |spearman_raw| + max(|spearman_detrend|, |spearman_yoy|) + mutual_info 합산 점수 상위")
    if len(top) == 0:
        print("  -> 위 조건을 만족하는 후보 없음(원본 상관이 순수 시간추세 혼입으로 의심되거나 통계적으로 유의하지 않음)")
    else:
        for _, r in top.iterrows():
            print(
                f"  - {r['candidate']:15s} raw={r['spearman_raw']:+.3f}(p={r['spearman_raw_p']:.3f})  "
                f"detrend={r['spearman_detrend_linear']:+.3f}(p={r['spearman_detrend_linear_p']:.3f})  "
                f"yoy={r['spearman_yoy']:+.3f}(p={r['spearman_yoy_p']:.3f})  MI={r['mutual_info']:.3f}"
            )
    print("=" * 80)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    candidates = build_candidates()
    EXP_DIR.mkdir(parents=True, exist_ok=True)
    candidates.to_parquet(CANDIDATES_OUT_PATH, index=False)
    print(f"\n[저장] 후보 피처 테이블 -> {CANDIDATES_OUT_PATH} ({len(candidates):,}행 x {candidates.shape[1]}컬럼)")

    print("\n[스크리닝] 스피어만 상관(원본/선형detrend잔차/YoY) + mutual information (모델 미사용, pairwise)")
    report = screen(candidates)
    report.to_csv(REPORT_OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"[저장] 스크리닝 리포트 -> {REPORT_OUT_PATH} ({len(report)}행 = 20후보 x 3그룹)")

    summarize_recommendations(report)


if __name__ == "__main__":
    main()
