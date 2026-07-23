"""
B센터 레짐 체인지(regime change) 정확한 시점 확정.

지금까지 "2023-05"(일 단위 Top5 상품 그래프 관찰)와 "2023-07"(주 단위 반품률/SKU재편/
boxplot 기준)로 다르게 언급된 시점을, 4가지 독립 지표로 교차검증해 통계적으로 재확인한다.

[데이터]
`daily_center_sku.csv`가 프로젝트에 실제로 존재하지 않아(확인함), 원본
cleaned_main_joined_cleaned_for_pred.parquet에서 B센터만 필터링해 매번 새로 만든다.

[Changepoint 탐지 방법]
ruptures 라이브러리(PELT/Binseg)를 쓰려 했으나 이 환경에 C 컴파일러가 없어 설치가
안 됨(pip install 실패, gcc 없음). 대신 단일 변화점에 대한 동등한 최대우도 방식을
직접 구현했다: 각 후보 분할점 t에 대해 전/후 평균 차이를 표본크기로 정규화한
CUSUM류 통계량 |mean_before-mean_after| * sqrt(t*(n-t)/n) 이 최대인 지점을 찾는다
(PELT/Binseg의 l2-cost 단일 변화점 탐지와 동일한 목적함수).

[월말 배치 아티팩트]
이전 분석(Step3)에서 일부 상품이 월말(예: 09-30, 10-31)에 배치성으로 대량 기록되는
패턴이 발견된 바 있다. 이게 changepoint 탐지를 왜곡할 수 있어 월말(각 달의 마지막
날) 제외 버전도 병행 계산해서 비교한다.
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SRC_PATH = "data/final/cleaned_main_joined_cleaned_for_pred.parquet"
OUT_PNG = "analysis-results/preprocessing/step_regime_changepoint_verification.png"
OUT_CSV = "analysis-results/preprocessing/step_regime_changepoint_summary.csv"

SEARCH_START = pd.Timestamp("2023-01-01")
SEARCH_END = pd.Timestamp("2023-12-31")
ROLL_WINDOW = 7

# Step4에서 확인된 B센터 학습기간 기준 Top5 상품
TOP5_B = {
    "8801048959044": "참이슬프레쉬병360ml*24(지)",
    "8801048101887": "진로이즈병360ml*24(지)",
    "8801858011123": "(인상)카스프레쉬캔355ml*24",
    "8801094016203": "코카콜라(슈퍼용)355ml",
    "8801094013004": "코카콜라250ml",
}

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


def load_b_daily() -> pd.DataFrame:
    df = pd.read_parquet(SRC_PATH)
    df["거래일"] = pd.to_datetime(df["거래일"])
    return df[df["센터"] == "B"].copy()


def find_single_changepoint(series: pd.Series, margin: int = 5) -> tuple:
    """전/후 평균차 정규화 통계량이 최대인 지점 (PELT/Binseg l2-cost 단일 변화점과 동등)."""
    values = series.values.astype(float)
    n = len(values)
    if n < margin * 2 + 1:
        return None, None
    cumsum = np.cumsum(values)
    total = cumsum[-1]
    best_score, best_t = -1.0, None
    for t in range(margin, n - margin):
        mean_before = cumsum[t - 1] / t
        mean_after = (total - cumsum[t - 1]) / (n - t)
        score = abs(mean_before - mean_after) * np.sqrt(t * (n - t) / n)
        if score > best_score:
            best_score, best_t = score, t
    return series.index[best_t], best_score


# ---------------------------------------------------------------------------
# 1. 반품률 급변 시점
# ---------------------------------------------------------------------------
def detect_return_rate_change(df: pd.DataFrame, exclude_month_end: bool = False) -> dict:
    d = df.copy()
    if exclude_month_end:
        d = d[d["거래일"] != (d["거래일"] + pd.offsets.MonthEnd(0))]

    daily = d.groupby("거래일")["수량"].agg(
        총판매수량=lambda s: s[s > 0].sum(), 반품수량=lambda s: (-s[s < 0]).sum(),
    )
    daily["반품률"] = daily["반품수량"] / (daily["총판매수량"] + daily["반품수량"])
    rolling = daily["반품률"].rolling(ROLL_WINDOW, min_periods=1).mean()

    window = rolling[(rolling.index >= SEARCH_START) & (rolling.index <= SEARCH_END)]
    cp_date, score = find_single_changepoint(window)
    return {"series": rolling, "changepoint": cp_date, "score": score}


# ---------------------------------------------------------------------------
# 2. SKU 구성 변화 시점 (월별 turnover)
# ---------------------------------------------------------------------------
def detect_sku_turnover(df: pd.DataFrame) -> dict:
    key_cols = ["바코드", "옵션코드", "상품클러스터"]
    d = df.copy()
    d["년월"] = d["거래일"].dt.to_period("M")
    monthly_sets = d.groupby("년월").apply(
        lambda g: set(map(tuple, g[key_cols].drop_duplicates().values)), include_groups=False
    )
    months = sorted(monthly_sets.index)

    rows = []
    for i in range(1, len(months)):
        prev_set, cur_set = monthly_sets[months[i - 1]], monthly_sets[months[i]]
        union = prev_set | cur_set
        inter = prev_set & cur_set
        jaccard_dist = 1 - len(inter) / len(union) if union else 0
        new_skus = cur_set - prev_set
        churned_skus = prev_set - cur_set
        turnover = (len(new_skus) + len(churned_skus)) / len(union) if union else 0
        rows.append({"년월": str(months[i]), "jaccard_거리": jaccard_dist, "turnover율": turnover,
                      "신규SKU수": len(new_skus), "소멸SKU수": len(churned_skus)})
    turnover_df = pd.DataFrame(rows)

    turnover_2023 = turnover_df[turnover_df["년월"].str.startswith("2023")]
    max_row = turnover_2023.loc[turnover_2023["turnover율"].idxmax()]
    return {"df": turnover_df, "changepoint_month": max_row["년월"], "turnover": max_row["turnover율"]}


# ---------------------------------------------------------------------------
# 3. 판매 패턴(톱니->평탄) 변동성 전환 시점
# ---------------------------------------------------------------------------
def detect_volatility_change(df: pd.DataFrame, exclude_month_end: bool = False) -> dict:
    d = df.copy()
    if exclude_month_end:
        d = d[d["거래일"] != (d["거래일"] + pd.offsets.MonthEnd(0))]

    daily_sales = d.groupby("거래일")["수량"].apply(lambda s: s[s > 0].sum())
    rolling_std = daily_sales.rolling(ROLL_WINDOW, min_periods=1).std()

    window = rolling_std[(rolling_std.index >= SEARCH_START) & (rolling_std.index <= SEARCH_END)].dropna()
    cp_date, score = find_single_changepoint(window)
    return {"series": rolling_std, "changepoint": cp_date, "score": score}


# ---------------------------------------------------------------------------
# 4. Top5 SKU 첫 유의미한 거래일
# ---------------------------------------------------------------------------
def detect_top5_first_significant(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for barcode, name in TOP5_B.items():
        sub = df[df["바코드"] == barcode]
        daily_sales = sub[sub["수량"] > 0].groupby("거래일")["수량"].sum()
        significant_days = daily_sales[daily_sales >= 10]
        first_day = significant_days.index.min() if len(significant_days) else None
        first_txn_day = sub["거래일"].min() if len(sub) else None
        rows.append({"바코드": barcode, "상품명": name, "첫유의미거래일(일판매>=10)": first_day,
                      "최초거래일(전체)": first_txn_day})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 시각화
# ---------------------------------------------------------------------------
def plot_timeline(return_rate_info: dict, volatility_info: dict, sku_turnover_info: dict,
                   top5_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    rr = return_rate_info["series"]
    rr_2023 = rr[(rr.index >= "2022-11-01") & (rr.index <= "2024-01-31")]
    axes[0].plot(rr_2023.index, rr_2023.values * 100, color="#ea580c", linewidth=1.5, label="반품률(7일 이동평균, %)")
    axes[0].set_ylabel("반품률 (%)")
    axes[0].legend(loc="upper left", frameon=False)

    vol = volatility_info["series"]
    vol_2023 = vol[(vol.index >= "2022-11-01") & (vol.index <= "2024-01-31")]
    ax0b = axes[0].twinx()
    ax0b.plot(vol_2023.index, vol_2023.values, color="#2563eb", linewidth=1.2, alpha=0.6, label="판매량 변동성(7일 std)")
    ax0b.set_ylabel("판매량 변동성(std)", color="#2563eb")
    ax0b.legend(loc="upper right", frameon=False)

    for ax in axes:
        if return_rate_info["changepoint"] is not None:
            ax.axvline(return_rate_info["changepoint"], color="#ea580c", linestyle="--", linewidth=1.5, alpha=0.8)
        if volatility_info["changepoint"] is not None:
            ax.axvline(volatility_info["changepoint"], color="#2563eb", linestyle=":", linewidth=1.5, alpha=0.8)
        ax.axvline(pd.Timestamp("2023-05-01"), color="#6b7280", linestyle="-", linewidth=1, alpha=0.5)
        ax.axvline(pd.Timestamp("2023-07-01"), color="#16a34a", linestyle="-", linewidth=1, alpha=0.5)

    turnover_month = pd.Timestamp(sku_turnover_info["changepoint_month"] + "-01")
    for ax in axes:
        ax.axvline(turnover_month, color="#a855f7", linestyle="-.", linewidth=1.5, alpha=0.8)

    axes[1].set_title("Top5 SKU 첫 유의미 거래일(일 판매량>=10)", fontsize=11, loc="left")
    for i, row in top5_df.iterrows():
        if row["첫유의미거래일(일판매>=10)"] is not None:
            axes[1].scatter(row["첫유의미거래일(일판매>=10)"], i, color="#dc2626", s=60, zorder=3)
            axes[1].text(row["첫유의미거래일(일판매>=10)"], i, f"  {row['상품명'][:14]}", fontsize=8, va="center")
    axes[1].set_yticks([])
    axes[1].set_ylim(-1, len(top5_df))

    fig.suptitle(
        "B센터 레짐 체인지 시점 교차검증 (주황 점선=반품률 변화점, 파랑 점선=변동성 변화점,\n"
        "보라 1점쇄선=SKU turnover 최대월, 회색=2023-05 참고선, 초록=2023-07 참고선)",
        fontsize=12, fontweight="bold",
    )
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"저장 완료 -> {OUT_PNG}")


def main() -> None:
    setup_korean_font()
    print(f"Loading {SRC_PATH} (B센터만 필터링) ...")
    df = load_b_daily()
    print(f"B센터 거래 행 수: {len(df):,}\n")

    print("=" * 70)
    print("[1] 반품률 급변 시점")
    print("=" * 70)
    rr_info = detect_return_rate_change(df, exclude_month_end=False)
    rr_info_ex = detect_return_rate_change(df, exclude_month_end=True)
    print(f"월말 포함: {rr_info['changepoint']} (score={rr_info['score']:.4f})")
    print(f"월말 제외: {rr_info_ex['changepoint']} (score={rr_info_ex['score']:.4f})")

    print("\n" + "=" * 70)
    print("[2] SKU 구성 변화 시점 (월별 turnover)")
    print("=" * 70)
    sku_info = detect_sku_turnover(df)
    print(sku_info["df"].to_string(index=False))
    print(f"\n2023년 중 turnover 최대월: {sku_info['changepoint_month']} (turnover율={sku_info['turnover']:.3f})")

    print("\n" + "=" * 70)
    print("[3] 판매 패턴 변동성 전환 시점")
    print("=" * 70)
    vol_info = detect_volatility_change(df, exclude_month_end=False)
    vol_info_ex = detect_volatility_change(df, exclude_month_end=True)
    print(f"월말 포함: {vol_info['changepoint']} (score={vol_info['score']:.4f})")
    print(f"월말 제외: {vol_info_ex['changepoint']} (score={vol_info_ex['score']:.4f})")

    print("\n" + "=" * 70)
    print("[4] Top5 SKU 첫 유의미 거래일")
    print("=" * 70)
    top5_df = detect_top5_first_significant(df)
    print(top5_df.to_string(index=False))

    plot_timeline(rr_info, vol_info, sku_info, top5_df)

    print("\n" + "=" * 70)
    print("[종합] 지표별 탐지 시점 비교")
    print("=" * 70)
    summary_rows = [
        {"지표": "반품률 변화점(월말포함)", "탐지일자": rr_info["changepoint"], "근거/신뢰도": f"score={rr_info['score']:.4f}"},
        {"지표": "반품률 변화점(월말제외)", "탐지일자": rr_info_ex["changepoint"], "근거/신뢰도": f"score={rr_info_ex['score']:.4f}"},
        {"지표": "SKU turnover 최대월", "탐지일자": sku_info["changepoint_month"], "근거/신뢰도": f"turnover율={sku_info['turnover']:.3f}"},
        {"지표": "판매변동성 전환(월말포함)", "탐지일자": vol_info["changepoint"], "근거/신뢰도": f"score={vol_info['score']:.4f}"},
        {"지표": "판매변동성 전환(월말제외)", "탐지일자": vol_info_ex["changepoint"], "근거/신뢰도": f"score={vol_info_ex['score']:.4f}"},
    ]
    for _, row in top5_df.iterrows():
        summary_rows.append({
            "지표": f"Top5_{row['상품명'][:12]}_첫유의미거래일",
            "탐지일자": row["첫유의미거래일(일판매>=10)"], "근거/신뢰도": "일판매량>=10 최초일",
        })
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료 -> {OUT_CSV}")


if __name__ == "__main__":
    main()
