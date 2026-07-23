"""
Step 4: 센터(A/B)별 주 단위 타겟 변수(총판매수량) 시계열 시각화.

대상: data/final/aggregated_weekly_demand.parquet (21-24년, 210주)
타겟 변수: 총판매수량 (반품이 차감되지 않은 순수 판매 수량)

[확인된 사실 - 재작업 불필요]
- 총판매수량은 aggregate_demand_data.py에서 수량>0인 거래만 합산해 만든 컬럼이라
  이미 항상 0 이상이다(검증됨, min=0). abs()나 별도 양수화 처리는 필요 없다.
- 수량 부호 반전이 의심되던 175개 품목(step3_anomalous_product_candidates.csv)은
  같은 파이프라인에서 이미 집계 대상에서 제외되어 있다. 여기서 다시 필터링하지 않는다.

[상품키 고유성]
- 상품키(바코드+옵션코드+상품클러스터)는 센터 내에서만 고유하다. A/B 센터에 같은
  상품키가 있어도 서로 다른 상품일 수 있어, 모든 집계·순위 산출(Top5 카테고리,
  Top5 상품)은 센터별로 완전히 분리해서 수행한다. A+B를 합쳐 순위를 매기지 않는다.

[Data Leakage 방지]
- Top5 카테고리/상품 선정은 학습기간(주시작일 < 2024-01-01)만 사용한다. 2024년
  데이터는 선정된 항목의 추세를 함께 그리는 용도로만 쓰고 순위 산출엔 넣지 않는다.
"""

import os

import matplotlib
matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

SRC_PATH = "data/final/aggregated_weekly_demand.parquet"
OUT_DIR = "analysis-results/eda"

REGIME_SHIFT_DATE = pd.Timestamp("2023-07-01")
TRAIN_CUTOFF = pd.Timestamp("2024-01-01")

PRODUCT_KEY_COLS = ["바코드", "옵션코드", "상품클러스터"]
CENTERS = ["A", "B"]
CENTER_COLORS = {"A": "#2563eb", "B": "#f59e0b"}

KOREAN_FONT_CANDIDATES = [
    "/mnt/c/Windows/Fonts/malgun.ttf",  # WSL에서 마운트된 Windows 폰트
    "C:/Windows/Fonts/malgun.ttf",  # 네이티브 Windows
    "/System/Library/Fonts/Supplemental/AppleGothic.ttf",  # Mac
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


def style_axes(ax) -> None:
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))


def mark_regime_and_test_period(ax, label_y_frac: float = 0.92) -> None:
    # axvspan/axvline은 patch 추가 시 autoscale을 재계산하면서 날짜 축 xlim이 실제
    # 데이터 범위 밖으로 크게 튀는 경우가 있어(확인된 버그), 추가 전 xlim을 저장해뒀다가
    # 마지막에 강제로 복원한다.
    xlo, xhi = ax.get_xlim()
    ax.axvspan(TRAIN_CUTOFF, xhi, color="#9ca3af", alpha=0.08, zorder=0)
    ax.axvline(REGIME_SHIFT_DATE, color="#dc2626", linewidth=1.5, linestyle="--", alpha=0.8, zorder=2)
    ylo, yhi = ax.get_ylim()
    ax.text(
        REGIME_SHIFT_DATE, ylo + (yhi - ylo) * label_y_frac, " 2023-07 레짐 전환",
        color="#dc2626", fontsize=9, va="top",
    )
    ax.set_xlim(xlo, xhi)


def compute_seasonality(df_center: pd.DataFrame) -> tuple:
    # 2020-12-28 시작 주(실제 거래일은 2021-01-02~03뿐인 반쪽 주)는 12월 평균을
    # 왜곡시키므로 제외하고 계산한다.
    clean = df_center[df_center["주시작일"] >= "2021-01-01"]
    yearly_monthly = (
        clean.assign(년=clean["주시작일"].dt.year, 월=clean["주시작일"].dt.month)
        .groupby(["년", "월"])["총판매수량"].sum().reset_index()
    )
    monthly_avg = yearly_monthly.groupby("월")["총판매수량"].mean()
    return monthly_avg.idxmax(), monthly_avg.idxmin()


def plot_total_trend(df_center: pd.DataFrame, center: str) -> dict:
    weekly = (
        df_center.groupby("주시작일")
        .agg(총판매수량=("총판매수량", "sum"), 반품수량=("반품수량", "sum"))
        .reset_index()
    )
    weekly["반품률(%)"] = weekly["반품수량"] / weekly["총판매수량"] * 100

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(14, 8), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    fig.suptitle(f"[{center}센터] 주별 총 매출 수량 및 반품률 추이 (2021-2024)", fontsize=15, fontweight="bold")

    ax1.plot(weekly["주시작일"], weekly["총판매수량"], color=CENTER_COLORS[center], linewidth=1.8)
    ax1.set_ylabel("총판매수량 (개)")
    ax1.set_title("주간 총 매출 수량", fontsize=11, loc="left", color="#374151")

    ax2.plot(weekly["주시작일"], weekly["반품률(%)"], color="#ea580c", linewidth=1.8)
    ax2.set_ylabel("반품률 (%)")
    ax2.set_title("주간 반품률", fontsize=11, loc="left", color="#374151")

    for ax in (ax1, ax2):
        style_axes(ax)
        mark_regime_and_test_period(ax)

    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    out_path = f"{OUT_DIR}/step4_total_weekly_demand_and_return_trend_{center}.png"
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"저장 완료 -> {out_path}")

    peak_row = weekly.loc[weekly["총판매수량"].idxmax()]
    pre_shift = weekly[weekly["주시작일"] < REGIME_SHIFT_DATE]["반품률(%)"].mean()
    post_shift = weekly[weekly["주시작일"] >= REGIME_SHIFT_DATE]["반품률(%)"].mean()
    peak_month, trough_month = compute_seasonality(df_center)

    return {
        "peak_week": peak_row["주시작일"].date(),
        "peak_value": peak_row["총판매수량"],
        "pre_shift_return_rate": pre_shift,
        "post_shift_return_rate": post_shift,
        "peak_month": peak_month,
        "trough_month": trough_month,
    }


def plot_combined_total_trend(df: pd.DataFrame) -> None:
    # 참고 비교용 차트 - Top5 선정 등 어떤 순위 산출에도 쓰이지 않는다.
    weekly = df.groupby(["주시작일", "센터"])["총판매수량"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(14, 6))
    fig.suptitle("[A/B 비교용] 주별 총 매출 수량 (참고용, 순위 산출 미사용)", fontsize=14, fontweight="bold")
    for center in CENTERS:
        sub = weekly[weekly["센터"] == center]
        ax.plot(sub["주시작일"], sub["총판매수량"], label=f"{center}센터", linewidth=1.6, color=CENTER_COLORS[center])
    ax.set_ylabel("총판매수량 (개)")
    ax.legend(loc="upper right", frameon=False)
    style_axes(ax)
    mark_regime_and_test_period(ax)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    out_path = f"{OUT_DIR}/step4_total_weekly_demand_and_return_trend_combined.png"
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"저장 완료 -> {out_path}")


def plot_top_category(df_center: pd.DataFrame, center: str) -> list:
    train = df_center[df_center["주시작일"] < TRAIN_CUTOFF]
    top5 = train.groupby("KAN_대분류")["총판매수량"].sum().sort_values(ascending=False).head(5).index.tolist()

    weekly = (
        df_center[df_center["KAN_대분류"].isin(top5)]
        .groupby(["주시작일", "KAN_대분류"])["총판매수량"].sum().reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(
        f"[{center}센터] Top 5 KAN_대분류별 주간 총 매출 수량 (Top5는 학습기간 21-23 기준)",
        fontsize=13, fontweight="bold",
    )

    palette = sns.color_palette("colorblind", n_colors=len(top5))
    for color, category in zip(palette, top5):
        sub = weekly[weekly["KAN_대분류"] == category]
        ax.plot(sub["주시작일"], sub["총판매수량"], label=category, linewidth=1.6, color=color)

    ax.set_ylabel("총판매수량 (개)")
    ax.legend(loc="upper right", frameon=False, ncol=2, fontsize=9)
    style_axes(ax)
    mark_regime_and_test_period(ax)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    out_path = f"{OUT_DIR}/step4_demand_by_top_category_{center}.png"
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"저장 완료 -> {out_path}")

    return top5


def plot_top_products(df_center: pd.DataFrame, center: str, full_weeks: pd.DatetimeIndex) -> list:
    train = df_center[df_center["주시작일"] < TRAIN_CUTOFF]
    # 상품명은 groupby 키에서 제외 - 같은 상품키인데 표기가 미세하게 갈리는 경우
    # 상품명까지 키에 넣으면 같은 상품이 여러 줄로 쪼개져 잘못 합산된다.
    totals = train.groupby(PRODUCT_KEY_COLS)["총판매수량"].sum().sort_values(ascending=False)
    top5_keys = totals.head(5).index.tolist()

    name_map = (
        df_center.drop_duplicates(subset=PRODUCT_KEY_COLS)
        .set_index(PRODUCT_KEY_COLS)["상품명"].to_dict()
    )

    weekly = (
        df_center.merge(pd.DataFrame(top5_keys, columns=PRODUCT_KEY_COLS), on=PRODUCT_KEY_COLS)
        .groupby(["주시작일"] + PRODUCT_KEY_COLS)["총판매수량"].sum().reset_index()
    )

    fig, ax = plt.subplots(figsize=(14, 8))
    fig.suptitle(
        f"[{center}센터] Top 5 상품 주간 매출 수량 추이 (Top5는 학습기간 21-23 기준)",
        fontsize=13, fontweight="bold",
    )

    # 거래가 없던 주는 행 자체가 존재하지 않으므로(간헐적 수요), 전체 주차 범위로
    # reindex해서 결측 주차를 명시적 NaN으로 만든다. matplotlib은 NaN 구간에서
    # 선을 자동으로 끊으므로(기본 동작, 별도 옵션 불필요) 이렇게 하면 "값이 이어지는
    # 것처럼" 보이던 직선(예: A센터 참소주G/카스355의 26주 공백)이 실제로 끊겨 보인다.
    for color, key in zip(sns.color_palette("colorblind", n_colors=5), top5_keys):
        mask = (
            (weekly["바코드"] == key[0])
            & (weekly["옵션코드"] == key[1])
            & (weekly["상품클러스터"] == key[2])
        )
        sub = weekly[mask].set_index("주시작일")["총판매수량"].reindex(full_weeks)
        label = str(name_map.get(key, key))[:20]
        ax.plot(sub.index, sub.values, label=label, linewidth=1.6, color=color, marker="o", markersize=2.5)

    ax.set_ylabel("총판매수량 (개)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    style_axes(ax)
    mark_regime_and_test_period(ax)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    plt.tight_layout()
    out_path = f"{OUT_DIR}/step4_top5_products_trend_{center}.png"
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"저장 완료 -> {out_path}")

    return [(key, name_map.get(key, "")) for key in top5_keys]


def print_insights(center: str, total_info: dict, top5_categories: list, top5_products: list) -> None:
    print(f"\n=== [{center}센터] 인사이트 요약 ===")
    print(f"- 최대 매출 주차: {total_info['peak_week']} (총판매수량 {total_info['peak_value']:,}개)")
    print(f"- 반품률 레짐 전환 전(~2023-06) 평균: {total_info['pre_shift_return_rate']:.2f}%")
    print(f"- 반품률 레짐 전환 후(2023-07~) 평균: {total_info['post_shift_return_rate']:.2f}%")
    print(f"- 계절성(연도별 집계 후 월평균): 최대 {total_info['peak_month']}월, 최소 {total_info['trough_month']}월")
    if total_info["peak_month"] <= 6 and total_info["trough_month"] >= 7:
        print("  (주의: 레짐 전환이 2023-07이라 상반기/하반기 비교에 레짐 효과가 섞였을 수 있음 - 참고용)")
    print(f"- Top 5 KAN_대분류(학습기간 기준): {', '.join(top5_categories)}")
    print("- Top 5 상품(학습기간 기준):")
    for key, name in top5_products:
        print(f"    {key[0]} ({key[1]}, 클러스터{key[2]}) {name}")


def main() -> None:
    setup_korean_font()
    print(f"Loading {SRC_PATH} ...")
    df = load_data()
    print(f"Loaded {len(df):,} rows\n")

    plot_combined_total_trend(df)
    full_weeks = pd.date_range(df["주시작일"].min(), df["주시작일"].max(), freq="7D")

    for center in CENTERS:
        df_center = df[df["센터"] == center]
        total_info = plot_total_trend(df_center, center)
        top5_categories = plot_top_category(df_center, center)
        top5_products = plot_top_products(df_center, center, full_weeks)
        print_insights(center, total_info, top5_categories, top5_products)


if __name__ == "__main__":
    main()
