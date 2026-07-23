"""
Step 5: 센터(A/B)별 주간 총판매수량 분포 확인 및 이상치(Outlier) 탐지.

[센터별 완전 분리 원칙]
모든 통계·시각화는 A/B 센터를 분리해서 수행한다(Step4와 동일 원칙).

[Data Leakage 방지]
이상치 판단 기준(IQR, Z-score)은 학습기간(주시작일 < 2024-01-01) 데이터만으로 산출하고,
2024년 데이터는 그 기준에 비춰 이상치인지 "확인"하는 용도로만 사용한다(기준 산출에는
2024년 데이터를 포함하지 않음).

[구조적 현상 vs 진짜 이상치 구분]
- B센터는 2023-07-01에 반품률 급증·SKU 25% 재편·발주 패턴 변화라는 이미 원인이 파악된
  구조적 전환이 있다(Step4 진단에서 확인됨). 이 시기 전체를 이상치로 뭉뚱그리면 무의미하므로,
  레짐 전(~2023-06)/후(2023-07~) 구간을 나누고 각 구간 내부에서만 이상치를 탐지한다.
- A센터 2024-01-01~06-24 주류 카테고리 26주 결측(Step4 진단에서 확인됨)은 이미 알려진
  사건이다. 센터 총매출(전 카테고리 합산) 레벨에서는 주류 비중이 작아 이상치로 안 잡힐 수
  있는데, 혹시 잡히더라도 새 발견으로 보고하지 않고 각주로만 표시한다.

[이상치 기준 산출 방식 - 코드 내 옵션]
SPLIT_BY_REGIME=True(기본): 레짐 구간별로 IQR/Z 기준을 따로 산출.
  - B센터: 레짐전(21-01~23-06) / 레짐후(23-07~23-12) 두 구간의 학습기간 데이터로 각각 산출.
    2024년 데이터는 "레짐후" 기준과 비교한다(현재도 그 레짐이 이어지고 있다고 보기 때문).
  - A센터: 레짐 전환이 없으므로 학습기간(21-23) 전체를 한 구간으로 취급.
SPLIT_BY_REGIME=False: 학습기간(21-23) 전체를 하나의 기준으로 산출(비교/대안용).
  B센터처럼 레짐 차이가 큰 경우, 레짐후 구간 전체가 통째로 이상치로 잡혀 무의미해질 수 있다.
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
# 이 스크립트는 순수탐색용 결과(boxplot, holiday_covid_impact)와 전처리용 결과
# (outlier_weeks - 이상치 마스킹 후보)를 함께 생성해 출력 폴더가 둘로 나뉜다.
OUT_DIR_EDA = "analysis-results/eda"
OUT_DIR_PREP = "analysis-results/preprocessing"

REGIME_SHIFT_DATE = pd.Timestamp("2023-07-01")
TRAIN_CUTOFF = pd.Timestamp("2024-01-01")
Z_THRESHOLD = 3.0
IQR_MULTIPLIER = 1.5
SPLIT_BY_REGIME = True  # 기본값. False로 두면 학습기간 전체를 단일 기준으로 사용(대안).

CENTERS = ["A", "B"]

# 이미 원인이 파악된 구조적 사건 - 이상치 표에서 "기존발견사항" 각주로만 표시.
KNOWN_EVENTS = {
    "B": [
        (
            pd.Timestamp("2023-07-01"), pd.Timestamp("2023-12-25"),
            "B센터 반품률 레짐 전환(반품 빈도 급증, SKU 25% 재편, 발주패턴 변화) "
            "- Step4에서 원인 확인된 구조적 변화, 개별 이상치 아님",
        ),
        (
            pd.Timestamp("2021-01-01"), pd.Timestamp("2023-06-30"),
            "B센터 레짐전 구간의 주기적 대량발주 스파이크 - Step1~4에서 이미 특징으로 파악된 "
            "정상 패턴(발주 주기 특성), 개별 이상치 아님",
        ),
    ],
    "A": [
        (
            pd.Timestamp("2024-01-01"), pd.Timestamp("2024-06-24"),
            "A센터 주류 카테고리 26주 결측 - Step4에서 확인된 사건. 센터 총매출(전카테고리 합산) "
            "레벨에선 주류 비중이 작아 이상치로 안 잡힐 수 있음",
        )
    ],
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


def load_data() -> pd.DataFrame:
    return pd.read_parquet(SRC_PATH)


def style_axes(ax) -> None:
    ax.grid(True, alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def weekly_totals(df_center: pd.DataFrame) -> pd.DataFrame:
    # 공휴일/covid_영향여부는 이미 날짜 단위(지역 무관) 플래그라 센터 내 어느 행을 봐도 값이
    # 같지만, 명시적으로 max 집계해서 주 단위 대표값을 만든다.
    return (
        df_center.groupby("주시작일")
        .agg(총판매수량=("총판매수량", "sum"), 공휴일=("공휴일", "max"), covid_영향여부=("covid_영향여부", "max"))
        .reset_index()
        .sort_values("주시작일")
    )


def plot_boxplot(weekly: pd.DataFrame, center: str) -> None:
    weekly = weekly.copy()
    weekly["연도분기"] = (
        weekly["주시작일"].dt.year.astype(str) + "-Q" + weekly["주시작일"].dt.quarter.astype(str)
    )
    order = sorted(weekly["연도분기"].unique(), key=lambda s: (int(s.split("-Q")[0]), int(s.split("-Q")[1])))

    fig, ax = plt.subplots(figsize=(14, 7))
    color = "#93c5fd" if center == "A" else "#fdba74"
    sns.boxplot(data=weekly, x="연도분기", y="총판매수량", order=order, ax=ax, color=color)
    ax.set_title(f"[{center}센터] 분기별 주간 총판매수량 분포", fontsize=14, fontweight="bold")
    ax.set_xlabel("")
    ax.set_ylabel("총판매수량 (개)")
    style_axes(ax)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    if center == "B" and "2023-Q3" in order:
        idx = order.index("2023-Q3")
        ax.axvline(idx - 0.5, color="#dc2626", linestyle="--", linewidth=1.5, alpha=0.8)
        ylo, yhi = ax.get_ylim()
        ax.text(idx - 0.5, yhi * 0.97, " 2023-07 레짐 전환", color="#dc2626", fontsize=9, va="top")

    plt.tight_layout()
    out_path = f"{OUT_DIR_EDA}/step5_boxplot_weekly_demand_{center}.png"
    plt.savefig(out_path, dpi=150, facecolor="white")
    plt.close(fig)
    print(f"저장 완료 -> {out_path}")


def compute_thresholds(series: pd.Series) -> dict:
    q1, q3 = series.quantile([0.25, 0.75])
    iqr = q3 - q1
    return {
        "Q1": q1, "Q3": q3, "IQR": iqr,
        "하한_IQR": q1 - IQR_MULTIPLIER * iqr, "상한_IQR": q3 + IQR_MULTIPLIER * iqr,
        "평균": series.mean(), "표준편차": series.std(),
    }


def assign_baseline(weekly: pd.DataFrame, center: str) -> tuple:
    weekly = weekly.copy()
    if center == "B" and SPLIT_BY_REGIME:
        train = weekly[weekly["주시작일"] < TRAIN_CUTOFF]
        pre_train = train[train["주시작일"] < REGIME_SHIFT_DATE]["총판매수량"]
        post_train = train[train["주시작일"] >= REGIME_SHIFT_DATE]["총판매수량"]
        thresholds = {
            "레짐전(21-23.06)": compute_thresholds(pre_train),
            "레짐후(23.07~)": compute_thresholds(post_train),
        }
        weekly["기준그룹"] = weekly["주시작일"].apply(
            lambda d: "레짐전(21-23.06)" if d < REGIME_SHIFT_DATE else "레짐후(23.07~)"
        )
    else:
        train = weekly[weekly["주시작일"] < TRAIN_CUTOFF]["총판매수량"]
        thresholds = {"학습기간전체(21-23)": compute_thresholds(train)}
        weekly["기준그룹"] = "학습기간전체(21-23)"
    return weekly, thresholds


def known_event_note(center: str, week: pd.Timestamp) -> str:
    for start, end, desc in KNOWN_EVENTS.get(center, []):
        if start <= week <= end:
            return desc
    return ""


def detect_outliers(weekly: pd.DataFrame, center: str) -> pd.DataFrame:
    weekly, thresholds = assign_baseline(weekly, center)
    rows = []
    for _, row in weekly.iterrows():
        th = thresholds[row["기준그룹"]]
        std = th["표준편차"]
        z = (row["총판매수량"] - th["평균"]) / std if std else 0.0
        iqr_flag = row["총판매수량"] < th["하한_IQR"] or row["총판매수량"] > th["상한_IQR"]
        z_flag = abs(z) > Z_THRESHOLD
        if not (iqr_flag or z_flag):
            continue
        rows.append({
            "주시작일": row["주시작일"].date(),
            "총판매수량": row["총판매수량"],
            "기준그룹": row["기준그룹"],
            "Z_score": round(z, 2),
            "IQR이상치": iqr_flag,
            "Z이상치": z_flag,
            "판정": "둘다" if (iqr_flag and z_flag) else ("IQR만" if iqr_flag else "Z만"),
            "공휴일": bool(row["공휴일"]),
            "covid_영향여부": bool(row["covid_영향여부"]),
            "기존발견사항": known_event_note(center, row["주시작일"]),
        })
    out = pd.DataFrame(rows).sort_values("주시작일") if rows else pd.DataFrame(
        columns=["주시작일", "총판매수량", "기준그룹", "Z_score", "IQR이상치", "Z이상치", "판정", "공휴일", "covid_영향여부", "기존발견사항"]
    )
    out_path = f"{OUT_DIR_PREP}/step5_outlier_weeks_iqr_zscore_{center}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"저장 완료 -> {out_path}")
    return out


def holiday_covid_impact(weekly: pd.DataFrame, center: str) -> pd.DataFrame:
    s = weekly.set_index("주시작일")["총판매수량"].sort_index()
    rows = []
    for flag_col, label in [("공휴일", "명절(설날/추석)"), ("covid_영향여부", "코로나 영향")]:
        for w in weekly.loc[weekly[flag_col] == 1, "주시작일"]:
            idx = s.index.get_loc(w)
            neighbor_idx = [i for i in range(max(0, idx - 4), min(len(s), idx + 5)) if i != idx]
            surrounding_avg = s.iloc[neighbor_idx].mean()
            value = s.loc[w]
            pct = (value - surrounding_avg) / surrounding_avg * 100 if surrounding_avg else float("nan")
            rows.append({
                "주시작일": w.date(), "유형": label, "총판매수량": value,
                "전후4주평균": round(surrounding_avg, 1), "증감률(%)": round(pct, 2),
            })
    out = pd.DataFrame(rows).sort_values(["유형", "주시작일"])
    out_path = f"{OUT_DIR_EDA}/step5_holiday_covid_impact_{center}.csv"
    out.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"저장 완료 -> {out_path}")
    return out


def print_summary(center: str, outliers: pd.DataFrame, impact: pd.DataFrame) -> None:
    print(f"\n=== [{center}센터] Step5 요약 ===")
    n_iqr = int(outliers["IQR이상치"].sum()) if len(outliers) else 0
    n_z = int(outliers["Z이상치"].sum()) if len(outliers) else 0
    n_both = int((outliers["판정"] == "둘다").sum()) if len(outliers) else 0
    print(f"이상치 주차: IQR기준 {n_iqr}건 / Z-score기준 {n_z}건 / 둘 다 겹침 {n_both}건")

    holiday = impact[impact["유형"] == "명절(설날/추석)"]
    covid = impact[impact["유형"] == "코로나 영향"]
    if len(holiday):
        print(f"명절 주차({len(holiday)}주) 평균 증감률: {holiday['증감률(%)'].mean():+.2f}% "
              f"(범위 {holiday['증감률(%)'].min():+.2f}% ~ {holiday['증감률(%)'].max():+.2f}%)")
    if len(covid):
        print(f"코로나 영향 주차({len(covid)}주, {covid['주시작일'].min()}~{covid['주시작일'].max()}) "
              f"평균 증감률: {covid['증감률(%)'].mean():+.2f}% "
              f"(범위 {covid['증감률(%)'].min():+.2f}% ~ {covid['증감률(%)'].max():+.2f}%)")

    if len(outliers):
        impact_dates = set(impact["주시작일"])
        unexplained = outliers[
            ~outliers["주시작일"].isin(impact_dates) & (outliers["기존발견사항"] == "")
        ]
        print(f"명절/코로나/기존발견사항으로 설명되지 않는 '설명 안 되는' 이상치: {len(unexplained)}건")
        if len(unexplained):
            print(unexplained[["주시작일", "총판매수량", "기준그룹", "판정"]].to_string(index=False))
    else:
        print("이상치 없음")


def main() -> None:
    setup_korean_font()
    print(f"Loading {SRC_PATH} ...")
    df = load_data()
    print(f"Loaded {len(df):,} rows")
    print(f"이상치 기준 산출 방식: SPLIT_BY_REGIME={SPLIT_BY_REGIME}\n")

    for center in CENTERS:
        df_center = df[df["센터"] == center]
        weekly = weekly_totals(df_center)

        plot_boxplot(weekly, center)
        outliers = detect_outliers(weekly, center)
        impact = holiday_covid_impact(weekly, center)
        print_summary(center, outliers, impact)

    print("\n[참고] 결측 구간이 이상치 탐지에 미치는 영향: A센터 주류 카테고리 26주 결측은 센터")
    print("총매출(전카테고리 합산) 시계열에서는 비중이 작아 이상치로 잘 안 잡히지만, 카테고리별로")
    print("쪼개서 보면(예: 주류만 따로) 그 구간이 명확한 이상치로 잡힐 수 있다.")


if __name__ == "__main__":
    main()
