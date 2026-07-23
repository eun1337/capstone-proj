# 02_center_b_regime_change_and_outlier_deepdive.py
# (1) 센터 B 레짐 체인지 정확한 전환 시점 특정
# (2) 상위 이상치 SKU들의 실제 거래 내역을 까봐서 오류 vs 정상 수요 급변 구분
# 기준 파일: data/final/cleaned_main_joined_cleaned_for_pred.parquet
# 01_row_level_baseline_profiling.py에서 확정한 SKU 키/타겟 변수/분석 단위를 그대로 사용

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
INPUT_PATH = DATA_DIR / "final" / "cleaned_main_joined_cleaned_for_pred.parquet"
OUT_DIR = BASE_DIR / "analysis-results" / "eda" / "overall" / "02_center_b_regime_change_and_outlier_deepdive"
OUT_DIR.mkdir(parents=True, exist_ok=True)


BARCODE_COL = "바코드"
OPTION_COL = "옵션코드"
CLUSTER_COL = "상품클러스터"
DATE_COL = "거래일"
CENTER_COL = "센터"
QTY_COL = "수량"
TARGET_COL = "수량"
HOLIDAY_COL = "공휴일"          
COVID_COL = "covid_영향여부"

_KOREAN_FONT_CANDIDATES = ["AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"]
_installed = {f.name for f in fm.fontManager.ttflist}
for _font in _KOREAN_FONT_CANDIDATES:
    if _font in _installed:
        plt.rcParams["font.family"] = _font
        break
plt.rcParams["axes.unicode_minus"] = False


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(INPUT_PATH)
    df[DATE_COL] = pd.to_datetime(df[DATE_COL])
    df["SKU_KEY"] = (
        df[BARCODE_COL].astype(str) + "_" +
        df[OPTION_COL].astype(str) + "_" +
        df[CLUSTER_COL].astype(str)
    )
    return df


def build_center_sku_daily(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby([CENTER_COL, "SKU_KEY", DATE_COL])[TARGET_COL]
        .sum()
        .reset_index()
    )



def detect_center_b_regime_change(df: pd.DataFrame, center="B", spike_threshold=3.0):
    print("=" * 60)
    print(f"[1] 센터 {center} 레짐 체인지 시점 특정")
    print("=" * 60)

    daily_total = (
        df[df[CENTER_COL] == center]
        .groupby(DATE_COL)[TARGET_COL].sum()
        .asfreq("D", fill_value=0)          # 거래 없는 날도 0으로 채워서 월별 통계 왜곡 방지
        .sort_index()
    )

    # 월별 max/mean 비율 = "그 달에 스파이크가 있었는가"를 나타내는 지표
    monthly = daily_total.groupby(pd.Grouper(freq="ME")).agg(["max", "mean"])
    monthly.columns = ["monthly_max", "monthly_mean"]
    monthly["spike_ratio"] = monthly["monthly_max"] / monthly["monthly_mean"].replace(0, np.nan)

    print("\n- 월별 spike_ratio (max/mean, 클수록 그 달에 스파이크 있음):")
    print(monthly["spike_ratio"].round(2))
    monthly.to_csv(OUT_DIR / f"center_{center}_monthly_spike_ratio.csv", encoding="utf-8-sig")

    # 마지막으로 spike_ratio가 threshold를 넘은 달 다음 달 = 전환 시점 후보
    spiky_months = monthly.index[monthly["spike_ratio"] >= spike_threshold]
    if len(spiky_months) > 0:
        last_spiky_month = spiky_months.max()
        transition_candidate = (last_spiky_month + pd.offsets.MonthBegin(1))
        print(f"\n- 마지막으로 스파이크(spike_ratio>={spike_threshold})가 관측된 달: "
              f"{last_spiky_month.strftime('%Y-%m')}")
        print(f"- 레짐 체인지 전환 시점 후보: {transition_candidate.strftime('%Y-%m')} 부터 안정화된 것으로 추정")
    else:
        transition_candidate = None
        print("\n- threshold 기준으로 스파이크 달을 찾지 못했습니다. spike_threshold를 낮춰서 재확인하세요.")

    # 스파이크 발생일이 월중 언제인지 (월말 배치설 검증용)
    # 주의: 레짐 체인지 전/후 베이스라인이 완전히 다르므로, 전체 기간 median을 쓰면
    # 레짐 체인지 이후의 평범한 날들까지 "스파이크"로 잘못 잡힌다.
    # → 레짐 체인지 이전 구간만 떼어서, 그 구간 자체의 median으로 재판정한다.
    if transition_candidate is not None:
        pre_regime = daily_total[daily_total.index < transition_candidate]
    else:
        pre_regime = daily_total

    pre_median = pre_regime.median()
    spike_days = pre_regime[pre_regime > pre_median * spike_threshold]

    if len(spike_days) > 0:
        day_of_month = spike_days.index.day
        print(f"\n- [레짐 체인지 이전 구간만] 스파이크 발생일의 '일(day-of-month)' 분포:")
        print(pd.Series(day_of_month).value_counts().sort_index())
        late_month_ratio = (day_of_month >= 25).mean()
        print(f"- 스파이크가 매월 25일 이후에 발생한 비율: {late_month_ratio:.1%}")
        print(f"- (참고) 이전 구간 스파이크 총 {len(spike_days)}건 "
              f"(약 {len(pre_regime)//30}개월 기준 월 1회 안팎이면 '월 1회 배치' 가설과 부합)")
    else:
        spike_days = daily_total[daily_total > daily_total.median() * spike_threshold]

    # 시각화: 전체 기간 + 전환 시점 표시 + 스파이크일 강조
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily_total.index, daily_total.values, linewidth=0.7, label=f"센터 {center} 일별 합계")
    ax.scatter(spike_days.index, spike_days.values, color="red", s=15, zorder=5, label="스파이크일")
    if transition_candidate is not None:
        ax.axvline(transition_candidate, color="black", linestyle="--",
                    label=f"전환 시점 후보 ({transition_candidate.strftime('%Y-%m')})")
    ax.set_title(f"센터 {center} 레짐 체인지 시점 확인")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"center_{center}_regime_change.png", dpi=150)
    plt.close(fig)

    print(f"\n- 저장 완료: center_{center}_monthly_spike_ratio.csv, center_{center}_regime_change.png")
    return transition_candidate, spike_days


# ══════════════════════════════════════════════════════════════
# [2] 상위 이상치 SKU 실제 거래 내역 딥다이브
# ══════════════════════════════════════════════════════════════
def compute_group_outliers(daily: pd.DataFrame) -> pd.DataFrame:
    grouped = daily.groupby([CENTER_COL, "SKU_KEY"])[TARGET_COL]
    q1 = grouped.transform(lambda x: x.quantile(0.25))
    q3 = grouped.transform(lambda x: x.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return daily[(daily[TARGET_COL] < lower) | (daily[TARGET_COL] > upper)]


def investigate_top_outlier_skus(df: pd.DataFrame, daily: pd.DataFrame, top_n=10, rank_by="frequency"):
    print("\n" + "=" * 60)
    print(f"[2] 상위 이상치 SKU 딥다이브 (기준: {rank_by})")
    print("=" * 60)

    outliers = compute_group_outliers(daily)

    if rank_by == "frequency":
        # 이상치가 자주 발생하는 SKU (intermittent demand 후보)
        top_groups = (
            outliers.groupby([CENTER_COL, "SKU_KEY"]).size()
            .sort_values(ascending=False).head(top_n)
        )
    elif rank_by == "magnitude":
        # 이상치 값이 그룹 평소 median 대비 몇 배나 튀는지 (진짜 이상 사건 후보)
        group_median = daily.groupby([CENTER_COL, "SKU_KEY"])[TARGET_COL].transform("median")
        outliers = outliers.copy()
        outliers["배율"] = outliers[TARGET_COL].abs() / group_median.loc[outliers.index].abs().clip(lower=1)
        top_groups = (
            outliers.groupby([CENTER_COL, "SKU_KEY"])["배율"].max()
            .sort_values(ascending=False).head(top_n)
        )
    else:
        raise ValueError("rank_by는 'frequency' 또는 'magnitude'여야 합니다")

    summary_rows = []
    fig, axes = plt.subplots(top_n, 1, figsize=(12, 2.5 * top_n), squeeze=False)

    for i, ((center, sku), _rank_value) in enumerate(top_groups.items()):
        group_daily = daily[(daily[CENTER_COL] == center) & (daily["SKU_KEY"] == sku)].sort_values(DATE_COL)
        group_outlier_dates = outliers[
            (outliers[CENTER_COL] == center) & (outliers["SKU_KEY"] == sku)
        ][DATE_COL]
        outlier_count = len(group_outlier_dates)

        # 이상치 날짜에 해당하는 row-level 상세 (공휴일/covid 여부, 거래건수 등)
        barcode, option, cluster = sku.split("_", 2)
        detail = df[
            (df[CENTER_COL] == center) &
            (df[BARCODE_COL].astype(str) == barcode) &
            (df[OPTION_COL].astype(str) == option) &
            (df[CLUSTER_COL].astype(str) == cluster) &
            (df[DATE_COL].isin(group_outlier_dates))
        ]

        holiday_hit_ratio = detail[HOLIDAY_COL].mean() if len(detail) else np.nan
        covid_hit_ratio = detail[COVID_COL].mean() if len(detail) else np.nan
        group_median = group_daily[TARGET_COL].median()
        outlier_values = group_daily[group_daily[DATE_COL].isin(group_outlier_dates)][TARGET_COL]
        magnitude_ratio = (outlier_values.abs().median() / max(abs(group_median), 1))

        summary_rows.append({
            "센터": center, "SKU_KEY": sku, "이상치_건수": outlier_count,
            "그룹_중앙값": group_median,
            "이상치_중앙값": outlier_values.median(),
            "이상치/중앙값_배율": round(magnitude_ratio, 1),
            "공휴일_적중률": round(holiday_hit_ratio, 2) if pd.notna(holiday_hit_ratio) else None,
            "covid_적중률": round(covid_hit_ratio, 2) if pd.notna(covid_hit_ratio) else None,
            "최초_이상치일": group_outlier_dates.min().date() if len(group_outlier_dates) else None,
            "최종_이상치일": group_outlier_dates.max().date() if len(group_outlier_dates) else None,
        })

        # 개별 시계열 + 이상치일 강조
        ax = axes[i, 0]
        ax.plot(group_daily[DATE_COL], group_daily[TARGET_COL], linewidth=0.7)
        outlier_plot = group_daily[group_daily[DATE_COL].isin(group_outlier_dates)]
        ax.scatter(outlier_plot[DATE_COL], outlier_plot[TARGET_COL], color="red", s=12, zorder=5)
        ax.set_title(f"센터{center} / {sku} (이상치 {outlier_count}건, 공휴일적중 {holiday_hit_ratio:.0%})"
                     if pd.notna(holiday_hit_ratio) else f"센터{center} / {sku} (이상치 {outlier_count}건)",
                     fontsize=9)

    plt.tight_layout()
    fig.savefig(OUT_DIR / f"top_outlier_sku_detail_{rank_by}.png", dpi=150)
    plt.close(fig)

    summary_df = pd.DataFrame(summary_rows)
    print("\n- 상위 이상치 SKU 요약:")
    print(summary_df.to_string(index=False))
    summary_df.to_csv(OUT_DIR / f"top_outlier_sku_summary_{rank_by}.csv", index=False, encoding="utf-8-sig")

    print(f"\n- 저장 완료: top_outlier_sku_summary_{rank_by}.csv, top_outlier_sku_detail_{rank_by}.png")
    print("  ※ '공휴일_적중률'이 높으면 설/추석 등 명절 수요 급증일 가능성,")
    print("    '이상치/중앙값_배율'이 비정상적으로 크면(수십~수백 배) 오탈력/오류 의심 필요")

    return summary_df


def main():
    df = load_data()
    daily = build_center_sku_daily(df)

    detect_center_b_regime_change(df, center="B")

    # 빈도 기준(자주 튀는 SKU=intermittent demand 후보)과
    # 배율 기준(한 번이라도 크게 튄 SKU=진짜 이상 사건 후보)을 둘 다 확인
    investigate_top_outlier_skus(df, daily, top_n=10, rank_by="frequency")
    investigate_top_outlier_skus(df, daily, top_n=10, rank_by="magnitude")

    print(f"\n모든 딥다이브 결과가 {OUT_DIR} 에 저장되었습니다.")


if __name__ == "__main__":
    main()
