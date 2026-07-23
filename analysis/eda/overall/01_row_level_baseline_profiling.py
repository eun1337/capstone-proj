# 01_row_level_baseline_profiling.py
# row-level 기초 통계 / 타겟 변수 시계열 플롯 / 이상치 후보 탐지
# 기준 파일: data/final/cleaned_main_joined_cleaned_for_pred.parquet
# SKU 키: 바코드 + 옵션코드 + 상품클러스터
# 타겟 변수: 수량 (반품은 수량<0이라 자동으로 순수량화됨)
# 분석 단위: 센터 + SKU (시도/시군구는 합산)

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
INPUT_PATH = DATA_DIR / "final" / "cleaned_main_joined_cleaned_for_pred.parquet"
OUT_DIR = BASE_DIR / "analysis-results" / "eda" / "overall" / "01_row_level_baseline_profiling"
OUT_DIR.mkdir(parents=True, exist_ok=True)


BARCODE_COL = "바코드"
OPTION_COL = "옵션코드"
CLUSTER_COL = "상품클러스터"
DATE_COL = "거래일"
CENTER_COL = "센터"
QTY_COL = "수량"
TARGET_COL = "수량"


_KOREAN_FONT_CANDIDATES = ["AppleGothic", "Malgun Gothic", "NanumGothic", "Noto Sans CJK KR"]
_installed = {f.name for f in fm.fontManager.ttflist}
for _font in _KOREAN_FONT_CANDIDATES:
    if _font in _installed:
        plt.rcParams["font.family"] = _font
        break
else:
    print("⚠ 한글 폰트를 찾지 못했습니다. 그림의 한글 라벨이 깨질 수 있습니다.")
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


def basic_stats(df: pd.DataFrame):
    print("=" * 60)
    print("[1] 기초 통계")
    print("=" * 60)

    print(f"- 전체 row 수: {len(df):,}")
    print(f"- 고유 SKU(바코드+옵션코드+상품클러스터) 수: {df['SKU_KEY'].nunique():,}")
    print(f"- 센터+SKU 조합 수: {df.groupby([CENTER_COL, 'SKU_KEY']).ngroups:,}")
    print(f"- 기간: {df[DATE_COL].min().date()} ~ {df[DATE_COL].max().date()}")
    print(f"- 센터별 row 수:\n{df[CENTER_COL].value_counts()}")

    print("\n- 결측치 비율 (%):")
    print((df.isna().mean() * 100).round(2).sort_values(ascending=False))

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    desc = df[numeric_cols].describe().T
    print("\n- 수치형 컬럼 기술통계:")
    print(desc)
    desc.to_csv(OUT_DIR / "describe_numeric.csv", encoding="utf-8-sig")

    sale_ratio = (df[QTY_COL] > 0).mean()
    return_ratio = (df[QTY_COL] < 0).mean()
    print(f"\n- 판매 row 비율: {sale_ratio:.2%} / 반품 row 비율: {return_ratio:.2%}")

    p1, p99 = df[TARGET_COL].quantile([0.01, 0.99])
    clipped = df[TARGET_COL].clip(p1, p99)
    n_excluded = ((df[TARGET_COL] < p1) | (df[TARGET_COL] > p99)).sum()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(clipped, bins=60)
    axes[0].set_title(f"{TARGET_COL} 분포 (1~99 백분위, {p1:.0f}~{p99:.0f}, 반품 포함)")

    sales_only = df.loc[df[QTY_COL] > 0, TARGET_COL]
    s_p99 = sales_only.quantile(0.99)
    axes[1].hist(sales_only.clip(upper=s_p99), bins=60)
    axes[1].set_title(f"{TARGET_COL} 분포 (판매 row만, ~{s_p99:.0f}까지)")

    plt.tight_layout()
    fig.savefig(OUT_DIR / "target_distribution.png", dpi=150)
    plt.close(fig)

    print(f"\n- {TARGET_COL} 1~99 백분위 범위 밖(극단값) row 수: {n_excluded:,} "
          f"({n_excluded/len(df):.2%}) — 그래프에서는 경계값으로 뭉쳐서 표시됨")

    return desc

def build_center_sku_daily(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby([CENTER_COL, "SKU_KEY", DATE_COL])[TARGET_COL]
        .sum()
        .reset_index()
    )



def target_timeseries_plot(daily: pd.DataFrame):
    print("\n" + "=" * 60)
    print("[2] 타겟 변수 시계열 플롯 (센터+SKU 합산 기준)")
    print("=" * 60)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    for center, g in daily.groupby(CENTER_COL):
        s = g.groupby(DATE_COL)[TARGET_COL].sum().sort_index()
        axes[0].plot(s.index, s.values, label=f"센터 {center}", linewidth=0.8)
        axes[0].plot(s.index, s.rolling(30).mean(), linestyle="--",
                     label=f"센터 {center} 30일 이동평균")
    axes[0].set_title(f"센터별 일별 {TARGET_COL} 합계 + 30일 이동평균 (추세)")
    axes[0].legend(fontsize=8)

    for center, g in daily.groupby(CENTER_COL):
        s = g.groupby(DATE_COL)[TARGET_COL].sum().sort_index().resample("W").sum()
        axes[1].plot(s.index, s.values, marker="o", markersize=3, label=f"센터 {center}")
    axes[1].set_title(f"센터별 주별 {TARGET_COL} 합계 (계절성/주기 확인)")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    fig.savefig(OUT_DIR / "target_timeseries_by_center.png", dpi=150)
    plt.close(fig)

    top_skus = (
        daily.groupby([CENTER_COL, "SKU_KEY"])[TARGET_COL].sum()
        .sort_values(ascending=False).head(6)
    )
    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    for ax, ((center, sku), _) in zip(axes.flat, top_skus.items()):
        g = daily[(daily[CENTER_COL] == center) & (daily["SKU_KEY"] == sku)]
        s = g.set_index(DATE_COL)[TARGET_COL].sort_index()
        ax.plot(s.index, s.values, linewidth=0.8)
        ax.set_title(f"센터{center} / {sku}", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUT_DIR / "top_sku_timeseries_sample.png", dpi=150)
    plt.close(fig)

    print("- 저장 완료: target_timeseries_by_center.png, top_sku_timeseries_sample.png")



def outlier_detection(df: pd.DataFrame, daily: pd.DataFrame):
    print("\n" + "=" * 60)
    print("[3] 이상치 후보 탐지")
    print("=" * 60)


    numeric_cols = [c for c in [QTY_COL, "금액", "평균온도", "총강수량", "입수"]
                     if c in df.columns]
    fig, axes = plt.subplots(1, len(numeric_cols), figsize=(3 * len(numeric_cols), 6))
    for ax, col in zip(axes, numeric_cols):
        ax.boxplot(df[col].dropna())
        ax.set_title(col)
        ax.set_xticks([])
    fig.suptitle("주요 수치형 컬럼 박스플롯 (컬럼별 개별 스케일, row-level)")
    plt.tight_layout()
    fig.savefig(OUT_DIR / "boxplot_numeric.png", dpi=150)
    plt.close(fig)


    p1, p99 = df[TARGET_COL].quantile([0.01, 0.99])
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.boxplot(df[TARGET_COL].clip(p1, p99))
    ax.set_title(f"{TARGET_COL} 박스플롯 (1~99 백분위 확대)")
    ax.set_xticks([])
    plt.tight_layout()
    fig.savefig(OUT_DIR / "boxplot_target_zoomed.png", dpi=150)
    plt.close(fig)


    grouped = daily.groupby([CENTER_COL, "SKU_KEY"])[TARGET_COL]
    q1 = grouped.transform(lambda x: x.quantile(0.25))
    q3 = grouped.transform(lambda x: x.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr

    is_outlier = (daily[TARGET_COL] < lower) | (daily[TARGET_COL] > upper)
    outliers = daily[is_outlier]

    print(f"- 센터+SKU 그룹 내부 IQR 기준 이상치 후보: {len(outliers):,} / {len(daily):,} "
          f"({len(outliers)/len(daily):.2%})")

    top_outlier_groups = (
        outliers.groupby([CENTER_COL, "SKU_KEY"]).size()
        .sort_values(ascending=False).head(20)
    )
    print("\n- 이상치 발생 빈도 상위 센터+SKU:")
    print(top_outlier_groups)

    outliers.to_csv(OUT_DIR / "outlier_candidates.csv", index=False, encoding="utf-8-sig")
    return outliers


def main():
    df = load_data()
    basic_stats(df)
    daily = build_center_sku_daily(df)
    target_timeseries_plot(daily)
    outlier_detection(df, daily)
    print(f"\n모든 EDA 결과가 {OUT_DIR} 에 저장되었습니다.")


if __name__ == "__main__":
    main()
