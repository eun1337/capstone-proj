# 03_train_test_redesign_and_external_corr_followups.py
# 후속 체크 5종:
#   [1] 센터 A 레짐 안정성 재검증 (센터 B와 동일한 방식으로 정량 검증)
#   [2] magnitude 기준 상위 이상치 SKU의 원본 row 직접 확인
#   [3] Train/Test 스플릿 옵션별 데이터 규모 비교 (의사결정 지원용, 자동 확정 아님)
#   [4] 옵션코드(EA/BX/CS)별 수량 분포 재확인
#   [5] 외부 변수(기상/거시경제/공휴일/covid)와 타겟 변수 상관관계
# 기준 파일: data/final/cleaned_main_joined_cleaned_for_pred.parquet

from pathlib import Path
import sys
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
INPUT_PATH = DATA_DIR / "final" / "cleaned_main_joined_cleaned_for_pred.parquet"
OUT_DIR = BASE_DIR / "analysis-results" / "eda" / "overall" / "03_train_test_redesign_and_external_corr_followups"
OUT_DIR.mkdir(parents=True, exist_ok=True)

class _Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data)
    def flush(self):
        for s in self.streams:
            s.flush()

_LOG_PATH = OUT_DIR / f"log_followups_{datetime.now():%Y%m%d_%H%M%S}.txt"
_log_file = open(_LOG_PATH, "w", encoding="utf-8")
sys.stdout = _Tee(sys.stdout, _log_file)
print(f"[로그 저장 위치] {_LOG_PATH}")

BARCODE_COL = "바코드"
OPTION_COL = "옵션코드"
CLUSTER_COL = "상품클러스터"
DATE_COL = "거래일"
CENTER_COL = "센터"
REGION_COLS = ["시도", "시군구"]
QTY_COL = "수량"
TARGET_COL = "수량"
AMOUNT_COL = "금액"
HOLIDAY_COL = "공휴일"
COVID_COL = "covid_영향여부"
EXTERNAL_COLS = ["평균온도", "총강수량", "cpi", "경상지수", "불변지수"]

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


def compute_group_outliers(daily: pd.DataFrame) -> pd.DataFrame:
    grouped = daily.groupby([CENTER_COL, "SKU_KEY"])[TARGET_COL]
    q1 = grouped.transform(lambda x: x.quantile(0.25))
    q3 = grouped.transform(lambda x: x.quantile(0.75))
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return daily[(daily[TARGET_COL] < lower) | (daily[TARGET_COL] > upper)]


def check_regime_stability(df: pd.DataFrame, center: str, spike_threshold=3.0):
    print("=" * 60)
    print(f"[1] 센터 {center} 레짐 안정성 검증")
    print("=" * 60)

    daily_total = (
        df[df[CENTER_COL] == center]
        .groupby(DATE_COL)[TARGET_COL].sum()
        .asfreq("D", fill_value=0)
        .sort_index()
    )
    monthly = daily_total.groupby(pd.Grouper(freq="ME")).agg(["max", "mean"])
    monthly.columns = ["monthly_max", "monthly_mean"]
    monthly["spike_ratio"] = monthly["monthly_max"] / monthly["monthly_mean"].replace(0, np.nan)

    print(f"\n- 센터 {center} 월별 spike_ratio 요약: "
          f"min={monthly['spike_ratio'].min():.2f}, max={monthly['spike_ratio'].max():.2f}, "
          f"std={monthly['spike_ratio'].std():.2f}")

    diffs = monthly["spike_ratio"].diff().abs()
    big_jumps = diffs[diffs > spike_threshold]
    if len(big_jumps) > 0:
        print(f"\n⚠ 센터 {center}: 아래 시점에서 spike_ratio가 급변함 (레짐 체인지 의심):")
        print(big_jumps)
    else:
        print(f"\n- 센터 {center}: spike_ratio가 큰 변화 없이 안정적 (레짐 체인지 없음으로 판단)")

    monthly.to_csv(OUT_DIR / f"center_{center}_monthly_spike_ratio.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(monthly.index, monthly["spike_ratio"], marker="o", markersize=3)
    ax.axhline(spike_threshold, color="red", linestyle="--", linewidth=0.8, label=f"threshold={spike_threshold}")
    ax.set_title(f"센터 {center} 월별 spike_ratio 추이 (급변 없으면 안정 레짐)")
    ax.legend()
    plt.tight_layout()
    fig.savefig(OUT_DIR / f"center_{center}_spike_ratio_trend.png", dpi=150)
    plt.close(fig)

    print(f"- 저장 완료: center_{center}_monthly_spike_ratio.csv, center_{center}_spike_ratio_trend.png")
    return monthly


def dump_raw_rows_for_magnitude_outliers(df: pd.DataFrame, daily: pd.DataFrame, top_n=5):
    print("\n" + "=" * 60)
    print("[2] magnitude 상위 이상치 원본 row 확인")
    print("=" * 60)

    outliers = compute_group_outliers(daily)
    group_median = daily.groupby([CENTER_COL, "SKU_KEY"])[TARGET_COL].transform("median")
    outliers = outliers.copy()
    outliers["배율"] = outliers[TARGET_COL].abs() / group_median.loc[outliers.index].abs().clip(lower=1)

    top_groups = (
        outliers.groupby([CENTER_COL, "SKU_KEY"])["배율"].max()
        .sort_values(ascending=False).head(top_n)
    )

    detail_cols = [CENTER_COL] + REGION_COLS + [BARCODE_COL, OPTION_COL, CLUSTER_COL,
                    DATE_COL, QTY_COL, AMOUNT_COL, HOLIDAY_COL, COVID_COL]
    detail_cols = [c for c in detail_cols if c in df.columns]

    all_details = []
    for (center, sku), ratio in top_groups.items():
        barcode, option, cluster = sku.split("_", 2)
        outlier_dates = outliers[
            (outliers[CENTER_COL] == center) & (outliers["SKU_KEY"] == sku)
        ][DATE_COL]

        rows = df[
            (df[CENTER_COL] == center) &
            (df[BARCODE_COL].astype(str) == barcode) &
            (df[OPTION_COL].astype(str) == option) &
            (df[CLUSTER_COL].astype(str) == cluster) &
            (df[DATE_COL].isin(outlier_dates))
        ][detail_cols].sort_values(DATE_COL)

        print(f"\n--- 센터{center} / {sku} (배율 {ratio:.1f}배, 원본 row {len(rows)}건) ---")
        print(rows.to_string(index=False))
        print(f"  → 지역(시도/시군구)이 여러 개면 '합산된 날'이라 개별 거래는 정상 크기일 수 있음.")
        print(f"    지역이 1개뿐인데 값이 크면 단일 거래 자체가 큰 것 — 대량발주/오류 여부 직접 판단 필요.")

        rows = rows.copy()
        rows["SKU_KEY"] = sku
        rows["배율"] = ratio
        all_details.append(rows)

    if all_details:
        pd.concat(all_details).to_csv(
            OUT_DIR / "magnitude_outlier_raw_rows.csv", index=False, encoding="utf-8-sig"
        )
        print(f"\n- 저장 완료: magnitude_outlier_raw_rows.csv")


def compare_split_options(df: pd.DataFrame, regime_transition="2023-05-01"):
    print("\n" + "=" * 60)
    print("[3] Train/Test 스플릿 옵션 비교")
    print("=" * 60)

    transition = pd.Timestamp(regime_transition)

    for center in sorted(df[CENTER_COL].unique()):
        sub = df[df[CENTER_COL] == center]
        total_days = (sub[DATE_COL].max() - sub[DATE_COL].min()).days
        post_regime = sub[sub[DATE_COL] >= transition]
        post_days = (post_regime[DATE_COL].max() - post_regime[DATE_COL].min()).days if len(post_regime) else 0

        print(f"\n- 센터 {center}")
        print(f"  옵션 A (전체 기간 사용, 레짐 플래그 피처 추가): "
              f"{sub[DATE_COL].min().date()} ~ {sub[DATE_COL].max().date()} ({total_days}일, row {len(sub):,})")
        print(f"  옵션 B (레짐 체인지 이후만 사용, {regime_transition}~): "
              f"{post_days}일, row {len(post_regime):,} "
              f"({len(post_regime)/max(len(sub),1):.1%} 사용)")

    print("\n※ 이 함수는 규모만 비교해서 보여줍니다. 최종 선택(A/B/센터별 다르게)은 "
          "레짐 체인지가 '데이터 기록 방식' 문제인지 '실제 수요 패턴' 변화인지에 대한 "
          "판단이 선행되어야 합니다 — [2]의 원본 row 확인 결과와 함께 결정하는 걸 추천합니다.")


def option_code_distribution(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("[4] 옵션코드별 수량 분포")
    print("=" * 60)

    desc = df.groupby(OPTION_COL)[TARGET_COL].describe()
    print(desc)
    desc.to_csv(OUT_DIR / "option_code_qty_describe.csv", encoding="utf-8-sig")

    option_codes = df[OPTION_COL].unique()
    fig, axes = plt.subplots(1, len(option_codes), figsize=(5 * len(option_codes), 4))
    if len(option_codes) == 1:
        axes = [axes]
    for ax, opt in zip(axes, option_codes):
        vals = df.loc[df[OPTION_COL] == opt, TARGET_COL]
        p1, p99 = vals.quantile([0.01, 0.99])
        ax.hist(vals.clip(p1, p99), bins=50)
        ax.set_title(f"옵션코드={opt} (n={len(vals):,})")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "qty_distribution_by_option.png", dpi=150)
    plt.close(fig)


    for opt in option_codes:
        vals = df.loc[(df[OPTION_COL] == opt) & (df[QTY_COL] > 0), QTY_COL]
        round_ratio = (vals % 10 == 0).mean()
        print(f"- 옵션코드={opt}: 10의 배수 수량 비율 = {round_ratio:.1%}")

    print("- 저장 완료: option_code_qty_describe.csv, qty_distribution_by_option.png")


def external_variable_correlation(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("[5] 외부 변수 상관관계")
    print("=" * 60)

    agg_dict = {TARGET_COL: "sum"}
    for col in EXTERNAL_COLS:
        if col in df.columns:
            agg_dict[col] = "mean"
    for col in [HOLIDAY_COL, COVID_COL]:
        if col in df.columns:
            agg_dict[col] = "max"

    daily_ext = df.groupby([CENTER_COL, DATE_COL]).agg(agg_dict).reset_index()

    corr_cols = [TARGET_COL] + [c for c in EXTERNAL_COLS + [HOLIDAY_COL, COVID_COL] if c in daily_ext.columns]
    corr = daily_ext[corr_cols].corr()

    print("\n- 일별(센터 합산 전) 상관행렬:")
    print(corr[[TARGET_COL]].sort_values(TARGET_COL, ascending=False))
    corr.to_csv(OUT_DIR / "external_var_correlation.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)))
    ax.set_yticklabels(corr.columns)
    for i in range(len(corr.columns)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    ax.set_title(f"{TARGET_COL} vs 외부 변수 상관관계 (센터+일별 집계 기준)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "external_var_correlation_heatmap.png", dpi=150)
    plt.close(fig)

    print("- 저장 완료: external_var_correlation.csv, external_var_correlation_heatmap.png")
    print("  ※ 상관계수는 '전체 기간'을 뭉쳐서 본 값이라, 레짐 체인지로 왜곡될 수 있음 —")
    print("    필요하면 레짐 체인지 이후 구간만 잘라서 다시 보는 것도 고려")


def main():
    df = load_data()
    daily = build_center_sku_daily(df)

    for center in sorted(df[CENTER_COL].unique()):
        check_regime_stability(df, center=center)

    dump_raw_rows_for_magnitude_outliers(df, daily, top_n=5)
    compare_split_options(df, regime_transition="2023-05-01")
    option_code_distribution(df)
    external_variable_correlation(df)

    print(f"\n모든 후속 체크 결과가 {OUT_DIR} 에 저장되었습니다.")


if __name__ == "__main__":
    main()
