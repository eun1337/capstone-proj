"""
02_full_daily_eda_by_center.py

일 단위 EDA — 기초통계/Zero-Ratio/요일패턴 + 시계열 플롯 + 이상치·특수일 탐지
+ Lag·ACF 파라미터 탐색 (센터 A/B 완전 분리 버전)

센터A와 센터B는 CV(변동계수), lag 자기상관 구조, 레짐체인지 유무 등에서
구조적으로 다르다는 게 확인되어, 합산 기준으로 보면
서로 다른 두 패턴이 평균으로 뭉개져 어느 쪽에도 안 맞는 결론이 나옴 → 모든
분석 항목을 센터별로 각각 수행한다.

입력: data/csv/daily_center_sku.csv
      (01_center_sku_csv_conversion.py 결과물, 센터+SKU+일 단위,
      지역(시도/시군구)은 합산되어 빠진 상태)

수행 항목 (센터 A, 센터 B, 각각 동일하게 반복)
  1. 기초 통계량 및 분포 확인
  2. 타겟 변수 시계열 플롯
  3. 이상치 및 특수일 탐지
  4. 시계열 파라미터 사전 탐색 (Lag, ACF)
마지막에 센터간 비교 요약(CV, lag, zero-ratio 등 핵심 지표만 표로 정리)도 출력.

출력: analysis-results/eda/daily/02_full_daily_eda_by_center/ 아래
  - center_A/  : 센터A 플롯 + summary.txt
  - center_B/  : 센터B 플롯 + summary.txt
  - summary_compare.txt : 센터간 비교 요약
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["axes.unicode_minus"] = False
for _font in ["Malgun Gothic", "AppleGothic", "NanumGothic", "Noto Sans CJK JP", "Noto Sans CJK KR"]:
    if _font in [f.name for f in matplotlib.font_manager.fontManager.ttflist]:
        plt.rcParams["font.family"] = _font
        break

BASE_DIR = Path(__file__).resolve().parents[3]
INPUT_PATH = BASE_DIR / "data" / "csv" / "daily_center_sku.csv"
ROOT_OUT = BASE_DIR / "analysis-results" / "eda" / "daily" / "02_full_daily_eda_by_center"
ROOT_OUT.mkdir(parents=True, exist_ok=True)

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAY_KO = {"Monday": "월", "Tuesday": "화", "Wednesday": "수", "Thursday": "목",
              "Friday": "금", "Saturday": "토", "Sunday": "일"}


def run_eda_for_center(df_all, center, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    log_lines = []

    def log(msg=""):
        print(f"[센터{center}] {msg}" if msg else "")
        log_lines.append(str(msg))

    def section(title):
        log("\n" + "=" * 70)
        log(f"[센터 {center}] {title}")
        log("=" * 70)

    df = df_all[df_all["센터"] == center].copy()
    df["요일"] = df["거래일"].dt.day_name()
    df["월말"] = df["거래일"].dt.is_month_end

    section("0. 데이터 개요")
    log(f"행 수: {len(df):,}")
    log(f"기간: {df['거래일'].min().date()} ~ {df['거래일'].max().date()}")
    log(f"고유 SKU 수: {df['SKU'].nunique():,}")

   
    section("1-1. 일별 총판매수량/반품수량/순수량 기술통계 (행 단위)")
    log(df[["총판매수량", "반품수량", "순수량"]].describe().to_string())

    section("1-2. 일 단위 Zero-Ratio")
    date_min, date_max = df["거래일"].min(), df["거래일"].max()
    n_days = (date_max - date_min).days + 1
    n_sku = df["SKU"].nunique()
    zero_ratio_fixed = 1 - len(df) / (n_sku * n_days)
    log(f"[전체기간 고정그리드 기준] SKU x 전체일수 = {n_sku:,} x {n_days:,} = {n_sku*n_days:,}")
    log(f"[전체기간 고정그리드 기준] Zero-Ratio = {zero_ratio_fixed:.4f} ({zero_ratio_fixed*100:.2f}%)")

    grp_span = df.groupby("SKU")["거래일"].agg(["min", "max", "count"])
    grp_span["span_days"] = (grp_span["max"] - grp_span["min"]).dt.days + 1
    zero_ratio_active = 1 - grp_span["count"].sum() / grp_span["span_days"].sum()
    log(f"[SKU별 활동기간 기준] Zero-Ratio = {zero_ratio_active:.4f} ({zero_ratio_active*100:.2f}%)")

    section("1-3. 요일별 평균 판매량/반품량")
    daily_total = df.groupby("거래일")[["총판매수량", "반품수량", "순수량"]].sum().reset_index()
    daily_total["요일"] = daily_total["거래일"].dt.day_name()
    daily_total["월말"] = daily_total["거래일"].dt.is_month_end

    log("\n[월말 포함 - 참고용]")
    log(daily_total.groupby("요일")[["총판매수량", "반품수량"]].mean().reindex(WEEKDAY_ORDER).to_string())
    log("\n[월말 제외 - 실제 요일 패턴]")
    wk_clean = daily_total[~daily_total["월말"]].groupby("요일")[["총판매수량", "반품수량"]].mean().reindex(WEEKDAY_ORDER)
    log(wk_clean.to_string())

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(7)
    width = 0.35
    ax.bar(x - width/2, wk_clean["총판매수량"], width, label="판매수량", color="#2563eb")
    ax.bar(x + width/2, wk_clean["반품수량"] * 10, width, label="반품수량(x10)", color="#dc2626")
    ax.set_xticks(x)
    ax.set_xticklabels([WEEKDAY_KO[d] for d in WEEKDAY_ORDER])
    ax.set_title(f"[센터{center}] 요일별 평균 판매량/반품량 (월말 제외)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "01_weekday_pattern.png", dpi=130)
    plt.close()


    section("2. 시계열 플롯 생성")
    daily_idx = df.groupby("거래일")[["총판매수량", "반품수량"]].sum()

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(daily_idx.index, daily_idx["총판매수량"], lw=0.6, color="#2563eb")
    ax.set_title(f"[센터{center}] 일별 총판매수량")
    plt.tight_layout()
    plt.savefig(out_dir / "02_total_daily_series.png", dpi=130)
    plt.close()

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(daily_idx.index, daily_idx["반품수량"], lw=0.6, color="#dc2626")
    ax.set_title(f"[센터{center}] 일별 반품수량")
    plt.tight_layout()
    plt.savefig(out_dir / "03_total_return_series.png", dpi=130)
    plt.close()

    top_cat = df.groupby("KAN_대분류")["총판매수량"].sum().sort_values(ascending=False).head(5).index
    cat_daily = df[df["KAN_대분류"].isin(top_cat)].groupby(["거래일", "KAN_대분류"])["총판매수량"].sum().unstack()
    fig, ax = plt.subplots(figsize=(14, 5))
    for c in cat_daily.columns:
        ax.plot(cat_daily.index, cat_daily[c], lw=0.5, alpha=0.8, label=c)
    ax.set_title(f"[센터{center}] 대분류별(상위5) 일별 총판매수량")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "04_category_daily_series.png", dpi=130)
    plt.close()
    log(f"대분류 상위5: {list(top_cat)}")

    top5_sku = df.groupby("SKU")["총판매수량"].sum().sort_values(ascending=False).head(5).index
    sku_daily = df[df["SKU"].isin(top5_sku)].groupby(["거래일", "SKU"])["총판매수량"].sum().unstack()
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True)
    for i, sku in enumerate(top5_sku):
        axes[i].plot(sku_daily.index, sku_daily[sku], lw=0.5, color="#059669")
        name = df.loc[df["SKU"] == sku, "상품명"].iloc[0]
        axes[i].set_title(f"{sku} - {name}", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "05_top5_sku_series.png", dpi=130)
    plt.close()
    log(f"Top5 SKU: {list(top5_sku)}")


    section("3-1. Boxplot")
    fig, ax = plt.subplots(figsize=(6, 5))
    data = df.loc[df["총판매수량"] > 0, "총판매수량"]
    ax.boxplot(np.log1p(data))
    ax.set_xticklabels(["총판매수량 (log1p)"])
    ax.set_title(f"[센터{center}] 일별 총판매수량 분포")
    plt.tight_layout()
    plt.savefig(out_dir / "06_boxplot_qty.png", dpi=130)
    plt.close()

    section("3-2. 스파이크일 탐지 (평균+3표준편차 초과)")
    mu, sigma = daily_idx["총판매수량"].mean(), daily_idx["총판매수량"].std()
    sales_spikes = daily_idx[daily_idx["총판매수량"] > mu + 3*sigma].sort_values("총판매수량", ascending=False)
    log(f"판매 스파이크일 ({len(sales_spikes)}건):")
    log(sales_spikes.to_string())

    mu_r, sigma_r = daily_idx["반품수량"].mean(), daily_idx["반품수량"].std()
    return_spikes = daily_idx[daily_idx["반품수량"] > mu_r + 3*sigma_r].sort_values("반품수량", ascending=False)
    log(f"\n반품 스파이크일 ({len(return_spikes)}건):")
    log(return_spikes.to_string())

    section("3-3. 공휴일 전후 및 날씨 극단일 영향 (월말 제외)")
    daily_ext = df.groupby("거래일").agg(
        총판매수량=("총판매수량", "sum"), 공휴일=("공휴일", "max"),
        평균온도=("평균온도", "first"), 총강수량=("총강수량", "first"),
    ).sort_index()
    daily_ext["월말"] = daily_ext.index.is_month_end
    clean = daily_ext[~daily_ext["월말"]].copy()
    clean["공휴일_D-1"] = clean["공휴일"].shift(-1).fillna(0)
    clean["공휴일_D+1"] = clean["공휴일"].shift(1).fillna(0)

    log("공휴일 D0 vs 평시:")
    log(clean.groupby("공휴일")["총판매수량"].agg(["mean", "count"]).to_string())
    log("\n공휴일 D-1 vs 평시:")
    log(clean.groupby("공휴일_D-1")["총판매수량"].agg(["mean", "count"]).to_string())
    log("\n공휴일 D+1 vs 평시:")
    log(clean.groupby("공휴일_D+1")["총판매수량"].agg(["mean", "count"]).to_string())

    rain_thresh = clean["총강수량"].quantile(0.95)
    log(f"\n강수량 상위5%(임계값 {rain_thresh:.1f}mm) vs 나머지:")
    log(clean.groupby(clean["총강수량"] >= rain_thresh)["총판매수량"].agg(["mean", "count"]).to_string())

    t_lo, t_hi = clean["평균온도"].quantile([0.05, 0.95])
    log(f"\n기온 극단일 (한파<= {t_lo:.1f}℃ / 폭염>= {t_hi:.1f}℃) vs 전체평균 {clean['총판매수량'].mean():.1f}:")
    log(f"  한파일 평균: {clean.loc[clean['평균온도'] <= t_lo, '총판매수량'].mean():.1f}")
    log(f"  폭염일 평균: {clean.loc[clean['평균온도'] >= t_hi, '총판매수량'].mean():.1f}")


    section("4-1. Lag 자기상관계수 (월말 제외 버전 기준)")
    daily_sales = df.groupby("거래일")["총판매수량"].sum()
    daily_sales_clean = daily_sales[~daily_sales.index.is_month_end]
    full_idx = pd.date_range(daily_sales.index.min(), daily_sales.index.max(), freq="D")
    series_clean = daily_sales_clean.reindex(full_idx).fillna(0)
    series_raw = daily_sales.asfreq("D").fillna(0)

    log("(a) 원본(월말 포함) - 참고용:")
    for lag in [1, 7, 14, 30]:
        log(f"  Lag {lag:>2}일: {series_raw.autocorr(lag=lag):.4f}")
    log("\n(b) 월말 제외 - feature 설계 근거로 사용할 버전:")
    for lag in [1, 7, 14, 30]:
        log(f"  Lag {lag:>2}일: {series_clean.autocorr(lag=lag):.4f}")

    cv = daily_sales_clean.std() / daily_sales_clean.mean()
    log(f"\n변동계수(CV) = std/mean = {cv:.4f}  (일평균 {daily_sales_clean.mean():.1f}, std {daily_sales_clean.std():.1f})")

    section("4-2. 요일 계절성(s=7) 확인용 ACF 플롯")
    try:
        from statsmodels.graphics.tsaplots import plot_acf
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))
        plot_acf(series_raw, lags=45, ax=axes[0])
        axes[0].set_title(f"[센터{center}] (a) 원본(월말 포함) ACF - 참고용")
        plot_acf(series_clean, lags=45, ax=axes[1])
        axes[1].set_title(f"[센터{center}] (b) 월말 제외 ACF - 이 버전 기준으로 s=7 판단")
        plt.tight_layout()
        plt.savefig(out_dir / "07_acf_plot.png", dpi=130)
        plt.close()
        log("ACF 플롯 저장 완료 (07_acf_plot.png)")
    except ImportError:
        log("statsmodels 미설치로 ACF 플롯 건너뜀")

    (out_dir / "summary.txt").write_text("\n".join(log_lines), encoding="utf-8")
    print(f"\n[센터{center}] summary.txt 및 플롯 저장 완료: {out_dir}")

    return {
        "센터": center,
        "zero_ratio_active": zero_ratio_active,
        "CV": cv,
        "lag1": series_clean.autocorr(1),
        "lag7": series_clean.autocorr(7),
        "lag14": series_clean.autocorr(14),
        "lag30": series_clean.autocorr(30),
        "일평균판매량": daily_sales_clean.mean(),
    }



df_all = pd.read_csv(INPUT_PATH, parse_dates=["거래일"])
df_all["SKU"] = df_all["바코드"].astype(str) + "_" + df_all["옵션코드"].astype(str) + "_" + df_all["상품클러스터"].astype(str)

results = []
for center in sorted(df_all["센터"].unique()):
    out_dir = ROOT_OUT / f"center_{center}"
    res = run_eda_for_center(df_all, center, out_dir)
    results.append(res)

compare_df = pd.DataFrame(results).set_index("센터")
compare_lines = ["센터간 핵심 지표 비교 (월말 제외 기준)", "=" * 50, compare_df.to_string()]
(ROOT_OUT / "summary_compare.txt").write_text("\n".join(compare_lines), encoding="utf-8")
print("\n" + "\n".join(compare_lines))
print(f"\n비교 요약 저장 완료: {ROOT_OUT / 'summary_compare.txt'}")
