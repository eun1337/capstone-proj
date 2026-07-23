"""
Step 6: 시계열 파라미터(Lag / 계절 주기) 탐색 - ACF/PACF, ADF 정상성 검정, STL 분해.

이 단계는 사전 탐색(exploration)이지 모델 확정이 아니다. "이거다"라고 단정하지 않고
각 후보의 장단점을 정리하는 톤으로 결과를 낸다.

[센터별 완전 분리 원칙]
A/B 센터를 분리해서 각각 분석한다. 두 센터의 lag/계절 주기가 다르게 나올 수 있다는
전제로 접근한다 - 특히 B센터는 레짐 전/후 자체가 통계적으로 다른 성격(Step4/5에서 확인:
반품률 급증, SKU 25% 재편, 톱니->평탄 패턴 전환)이라 같은 주기를 가정하지 않는다.

[B센터 레짐 구간 정의 - 사용자 확인 완료]
ACF/PACF·계절성 주기 탐색을 21-23 전체로 하면 레짐전/후가 섞여 가짜 주기성(mixture
artifact)이 나올 위험이 있어, B센터는 레짐후(2023-07-01~2023-12-25, 26주, 학습기간
내로 엄격히 제한 - 2024년 데이터 미포함)를 1차 분석 대상으로 하고, 레짐 전체(21-23,
157주)를 참고용으로 병기해 비교한다.
주의: 26주는 lag=52(1년) ACF/PACF 계산 자체가 불가능한 길이다(nlags는 nobs보다 작아야
함). B 레짐후 분석에서 lag=52, 계절주기 s=52는 "계산 불가"로 명시하고, 참고용
레짐전체(21-23) 결과에서만 확인한다. s=13도 26주 대비 2주기뿐이라 추정이 불안정하다.

[Data Leakage 방지]
모든 분석은 학습기간(주시작일 < 2024-01-01)만 사용한다. 2024년 데이터는 이후 모델
검증 단계에서 쓸 것이므로 이번 파라미터 탐색에는 전혀 포함하지 않는다.
A센터는 레짐 구분이 없으므로 학습기간(21-23) 전체를 그대로 사용한다.
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import acf, adfuller

SRC_PATH = "data/final/aggregated_weekly_demand.parquet"
OUT_DIR = "analysis-results/eda"

TRAIN_CUTOFF = pd.Timestamp("2024-01-01")
REGIME_SHIFT_DATE = pd.Timestamp("2023-07-01")

CANDIDATE_LAGS = [1, 2, 3, 4, 13, 26, 52]
SEASONAL_CANDIDATES = [52, 13, 4]

KOREAN_FONT_CANDIDATES = [
    "/mnt/c/Windows/Fonts/malgun.ttf",
    "C:/Windows/Fonts/malgun.ttf",
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",
]


def setup_korean_font() -> None:
    for path in KOREAN_FONT_CANDIDATES:
        if os.path.exists(path):
            fm.fontManager.addfont(path)
            plt.rcParams["font.family"] = fm.FontProperties(fname=path).get_name()
            break
    else:
        plt.rcParams["font.family"] = "AppleGothic"
    plt.rcParams["axes.unicode_minus"] = False


def load_data() -> pd.DataFrame:
    return pd.read_parquet(SRC_PATH)


def weekly_series(df_center: pd.DataFrame) -> pd.Series:
    return df_center.groupby("주시작일")["총판매수량"].sum().sort_index()


def max_reliable_nlags(n: int) -> int:
    # statsmodels 요구사항: nlags < nobs. 통상적 권장치(nobs//2)로 제한해 신뢰도 확보.
    return max(1, min(n - 2, n // 2))


def run_adf(series: pd.Series) -> dict:
    stat, pvalue, *_ = adfuller(series, autolag="AIC")
    return {"통계량": round(stat, 4), "p-value": round(pvalue, 4), "정상성(p<0.05)": pvalue < 0.05}


def candidate_lag_significance(series: pd.Series) -> pd.DataFrame:
    n = len(series)
    max_lag = min(max(CANDIDATE_LAGS), n - 1) if n > 1 else 0
    rows = []
    if max_lag >= 1:
        acf_vals, confint = acf(series, nlags=max_lag, alpha=0.05, fft=False)
    for lag in CANDIDATE_LAGS:
        if lag > max_lag or lag >= n:
            rows.append({"lag": lag, "ACF": None, "유의성": "계산불가(관측치부족)"})
            continue
        val = acf_vals[lag]
        lo, hi = confint[lag]
        # confint는 (acf±margin) 형태라 0을 포함하는 폭으로 환산해서 유의성 판단
        margin = hi - val
        significant = abs(val) > margin
        rows.append({
            "lag": lag, "ACF": round(val, 4),
            "유의성": "유의함(95% CI 초과)" if significant else "유의하지 않음",
        })
    return pd.DataFrame(rows)


def plot_acf_pacf_pair(series: pd.Series, title: str, out_path: str) -> int:
    n = len(series)
    nlags = max_reliable_nlags(n)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    plot_acf(series, lags=nlags, ax=axes[0])
    axes[0].set_title(f"{title} - ACF (n={n}, nlags={nlags})", fontsize=12)
    plot_pacf(series, lags=nlags, ax=axes[1], method="ywm")
    axes[1].set_title(f"{title} - PACF (n={n}, nlags={nlags})", fontsize=12)
    for ax in axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"저장 완료 -> {out_path}")
    return nlags


def seasonal_candidate_eval(series: pd.Series, s: int) -> dict:
    n = len(series)
    # STL/계절차분 모두 최소 2주기 이상 데이터가 있어야 의미 있는 추정이 된다.
    if n < 2 * s + 1:
        return {"적용가능": False, "계절차분_분산감소율(%)": None, "STL_계절성강도": None,
                "비고": f"n={n} < 2*s+1={2*s+1}, 데이터 부족으로 추정 불안정/불가"}

    diffed = series.diff(s).dropna()
    var_reduction = (1 - diffed.var() / series.var()) * 100

    try:
        # robust=True는 이상치에 강건하지만, 주기 수가 적을 때(예: n=156, s=52 -> 3주기뿐)
        # 반복 재가중 과정에서 경계 구간의 residual이 거의 0으로 수렴하는 퇴화(degenerate)
        # 현상이 실제로 관측됨(검증됨: A센터 s=52에서 156개 중 104개 residual이 1e-9 수준).
        # 짧은 시리즈에서는 robust=False가 더 안정적이라 기본값으로 사용한다.
        stl = STL(series, period=s, robust=False).fit()
        resid_var = stl.resid.var()
        seasonal_resid_var = (stl.seasonal + stl.resid).var()
        strength = max(0.0, 1 - resid_var / seasonal_resid_var) if seasonal_resid_var else 0.0
        note = "" if n >= 3 * s else f"2주기 남짓(n={n}, s={s})이라 추정이 약함"
    except Exception as e:
        strength = None
        note = f"STL 실패: {e}"

    return {
        "적용가능": True,
        "계절차분_분산감소율(%)": round(var_reduction, 2),
        "STL_계절성강도": round(strength, 4) if strength is not None else None,
        "비고": note,
    }


def plot_stl(series: pd.Series, s: int, title: str, out_path: str) -> None:
    stl = STL(series, period=s, robust=True).fit()
    fig = stl.plot()
    fig.set_size_inches(12, 9)
    fig.suptitle(f"{title} (s={s})", fontsize=13, fontweight="bold", y=1.01)
    for ax in fig.axes:
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"저장 완료 -> {out_path}")


def analyze_segment(series: pd.Series, center: str, segment_label: str,
                     acf_pacf_path: str, save_stl_path: str | None) -> dict:
    print(f"\n=== [{center}센터] {segment_label} (n={len(series)}) ===")

    adf_raw = run_adf(series)
    diffed = series.diff().dropna()
    adf_diff = run_adf(diffed) if len(diffed) > 3 else {"통계량": None, "p-value": None, "정상성(p<0.05)": None}
    print(f"ADF(원본): 통계량={adf_raw['통계량']}, p-value={adf_raw['p-value']}, 정상성={adf_raw['정상성(p<0.05)']}")
    print(f"ADF(1차차분): 통계량={adf_diff['통계량']}, p-value={adf_diff['p-value']}, 정상성={adf_diff['정상성(p<0.05)']}")

    nlags = plot_acf_pacf_pair(series, f"[{center}센터] {segment_label}", acf_pacf_path)
    lag_table = candidate_lag_significance(series)
    sig_lags = lag_table.loc[lag_table["유의성"].str.startswith("유의함", na=False), "lag"].tolist()
    print(f"유의미한 lag(원본 시리즈): {sig_lags if sig_lags else '없음'}")
    print(lag_table.to_string(index=False))

    # 차분 후 ACF도 참고용으로 별도 계산(플롯은 저장하지 않고 유의 lag만 콘솔 출력)
    if len(diffed) > 3:
        diff_lag_table = candidate_lag_significance(diffed)
        sig_lags_diff = diff_lag_table.loc[diff_lag_table["유의성"].str.startswith("유의함", na=False), "lag"].tolist()
        print(f"유의미한 lag(1차차분 후): {sig_lags_diff if sig_lags_diff else '없음'}")

    seasonal_results = {}
    for s in SEASONAL_CANDIDATES:
        res = seasonal_candidate_eval(series, s)
        seasonal_results[s] = res
        note = f" ({res['비고']})" if res["비고"] else ""
        if res["적용가능"]:
            print(f"s={s}: 계절차분 분산감소율={res['계절차분_분산감소율(%)']}%, "
                  f"STL 계절성강도={res['STL_계절성강도']}{note}")
        else:
            print(f"s={s}: 적용 불가 - {res['비고']}")

    feasible = {s: r for s, r in seasonal_results.items() if r["적용가능"] and r["STL_계절성강도"] is not None}
    if feasible:
        recommended_s = max(feasible, key=lambda s: feasible[s]["STL_계절성강도"])
        reason = (
            f"적용 가능한 후보 중 STL 계절성강도가 가장 높음({feasible[recommended_s]['STL_계절성강도']})"
        )
        if feasible[recommended_s]["비고"]:
            reason += f" - 단, {feasible[recommended_s]['비고']}"
    else:
        recommended_s = None
        reason = "적용 가능한 계절 주기 후보 없음(데이터 부족)"
    print(f"권장 s: {recommended_s} ({reason})")

    if save_stl_path and recommended_s:
        plot_stl(series, recommended_s, f"[{center}센터] {segment_label} STL 분해", save_stl_path)

    return {
        "센터": center, "구간": segment_label, "n": len(series),
        "ADF_통계량_원본": adf_raw["통계량"], "ADF_pvalue_원본": adf_raw["p-value"],
        "정상성_원본": adf_raw["정상성(p<0.05)"],
        "ADF_통계량_1차차분": adf_diff["통계량"], "ADF_pvalue_1차차분": adf_diff["p-value"],
        "정상성_1차차분": adf_diff["정상성(p<0.05)"],
        "유의미한_lag_원본": ",".join(map(str, sig_lags)) if sig_lags else "",
        **{
            f"s={s}_적용가능": seasonal_results[s]["적용가능"] for s in SEASONAL_CANDIDATES
        },
        **{
            f"s={s}_분산감소율(%)": seasonal_results[s]["계절차분_분산감소율(%)"] for s in SEASONAL_CANDIDATES
        },
        **{
            f"s={s}_STL계절성강도": seasonal_results[s]["STL_계절성강도"] for s in SEASONAL_CANDIDATES
        },
        "권장_s": recommended_s, "권장근거": reason,
    }


def main() -> None:
    setup_korean_font()
    print(f"Loading {SRC_PATH} ...")
    df = load_data()
    print(f"Loaded {len(df):,} rows")

    summary_rows = []

    # --- A센터: 레짐 구분 없음, 학습기간(21-23) 전체 ---
    a_train = df[(df["센터"] == "A") & (df["주시작일"] < TRAIN_CUTOFF)]
    a_series = weekly_series(a_train)
    summary_rows.append(analyze_segment(
        a_series, "A", "학습기간 전체(21-23)",
        f"{OUT_DIR}/step6_acf_pacf_A.png", f"{OUT_DIR}/step6_stl_decomposition_A.png",
    ))

    # --- B센터: 레짐후(1차 분석, 26주) ---
    b_train = df[(df["센터"] == "B") & (df["주시작일"] < TRAIN_CUTOFF)]
    b_full_series = weekly_series(b_train)
    b_post_series = b_full_series[b_full_series.index >= REGIME_SHIFT_DATE]
    summary_rows.append(analyze_segment(
        b_post_series, "B", "레짐후(2023-07~2023-12, 1차 분석)",
        f"{OUT_DIR}/step6_acf_pacf_B_regime_post.png", f"{OUT_DIR}/step6_stl_decomposition_B.png",
    ))

    # --- B센터: 레짐 전체(21-23, 참고용) ---
    summary_rows.append(analyze_segment(
        b_full_series, "B", "레짐 전체(21-23, 참고용 - 레짐전/후 혼재)",
        f"{OUT_DIR}/step6_acf_pacf_B_regime_full_ref.png", None,
    ))

    print("\n" + "=" * 70)
    print("[B센터] 레짐후(26주) vs 레짐전체(157주, 참고) 비교 요약")
    print("=" * 70)
    post_row = summary_rows[1]
    full_row = summary_rows[2]
    print(f"정상성(원본): 레짐후={post_row['정상성_원본']} / 레짐전체(참고)={full_row['정상성_원본']}")
    print(f"권장 s: 레짐후={post_row['권장_s']} / 레짐전체(참고)={full_row['권장_s']}")
    print("주의: 레짐전체(참고)는 레짐전(톱니 패턴)과 레짐후(평탄 패턴)가 섞여 있어 나온 주기성이")
    print("      실제 계절성이 아니라 두 레짐의 평균적 차이(mixture artifact)를 반영했을 수 있다.")
    print("      레짐후(26주) 결과가 '현재 레짐'을 더 잘 대표하지만, 26주는 약 0.5~2주기 분량이라")
    print("      s=52는 물론 s=13조차 추정이 통계적으로 약하다는 점을 감안해야 한다.")

    summary_df = pd.DataFrame(summary_rows)
    out_path = f"{OUT_DIR}/step6_stationarity_and_seasonality_summary.csv"
    summary_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료 -> {out_path}")


if __name__ == "__main__":
    main()
