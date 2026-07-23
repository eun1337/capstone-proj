"""
run_eda.py
------------------------------------------------------------------
data/final/cleaned_main_joined.parquet 기준 기초 EDA 스크립트.
    1) 수치형 컬럼 기술통계(describe)
    2) 주요 수치형 컬럼 분포(히스토그램 + 박스플롯)
    3) 타겟 변수(수량/금액) 일별 시계열 플롯 (7일 이동평균 포함)
    4) IQR 기준 이상치 후보 탐지
    5) 주요 범주형 컬럼(시도, KAN_대분류) 분포

결과는 data/ml/eda/ 에 저장 (통계는 csv, 플롯은 png).
"""

import os

import matplotlib.pyplot as plt
import pandas as pd

# Windows 기준 한글 폰트 (Mac이면 'AppleGothic', Linux면 'NanumGothic' 등으로 교체)
plt.rcParams["font.family"] = "Malgun Gothic"
plt.rcParams["axes.unicode_minus"] = False

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IN_FILE = os.path.join(BASE_DIR, "data", "final", "cleaned_main_joined.parquet")
OUT_DIR = os.path.join(BASE_DIR, "data", "ml", "eda")

NUMERIC_COLS = ["수량", "금액", "평균온도", "총강수량", "cpi", "경상지수", "불변지수"]
TARGET_COLS = ["수량", "금액"]
CATEGORICAL_COLS = ["시도", "KAN_대분류"]


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(IN_FILE, engine="pyarrow")
    df["거래일"] = pd.to_datetime(df["거래일"])
    print(f"로드 완료: {df.shape}")
    return df


def describe_stats(df: pd.DataFrame):
    desc = df[NUMERIC_COLS].describe().T
    desc.to_csv(os.path.join(OUT_DIR, "descriptive_stats.csv"), encoding="utf-8-sig")
    print("\n[기술통계]")
    print(desc)


def plot_distributions(df: pd.DataFrame):
    for col in NUMERIC_COLS:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))

        df[col].hist(bins=60, ax=axes[0])
        axes[0].set_title(f"{col} 분포(히스토그램)")

        axes[1].boxplot(df[col].dropna(), vert=False)
        axes[1].set_title(f"{col} 박스플롯")

        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"dist_{col}.png"), dpi=120)
        plt.close(fig)
    print(f"\n[분포 플롯] {len(NUMERIC_COLS)}개 컬럼 저장 완료")


def plot_target_timeseries(df: pd.DataFrame):
    for col in TARGET_COLS:
        daily = df.groupby("거래일")[col].sum().sort_index()
        rolling7 = daily.rolling(7, min_periods=1).mean()

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(daily.index, daily.values, alpha=0.4, label="일별 합계")
        ax.plot(rolling7.index, rolling7.values, linewidth=2, label="7일 이동평균")
        ax.set_title(f"일별 {col} 합계 시계열")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"target_timeseries_{col}.png"), dpi=120)
        plt.close(fig)

        daily.to_csv(os.path.join(OUT_DIR, f"target_daily_{col}.csv"), encoding="utf-8-sig")
    print(f"\n[타겟 시계열] {TARGET_COLS} 저장 완료")


def detect_outliers_iqr(df: pd.DataFrame):
    rows = []
    for col in NUMERIC_COLS:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        mask = (df[col] < lower) | (df[col] > upper)
        rows.append({
            "컬럼": col,
            "하한": lower,
            "상한": upper,
            "이상치건수": int(mask.sum()),
            "이상치비율(%)": round(100 * mask.sum() / len(df), 3),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, "outlier_summary.csv"), index=False, encoding="utf-8-sig")
    print("\n[이상치 후보 (IQR 1.5배 기준)]")
    print(summary)


def summarize_categoricals(df: pd.DataFrame):
    for col in CATEGORICAL_COLS:
        vc = df[col].value_counts()
        vc.to_csv(os.path.join(OUT_DIR, f"category_counts_{col}.csv"), encoding="utf-8-sig")
    print(f"\n[범주형 분포] {CATEGORICAL_COLS} 저장 완료")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    df = load_data()

    describe_stats(df)
    plot_distributions(df)
    plot_target_timeseries(df)
    detect_outliers_iqr(df)
    summarize_categoricals(df)

    print(f"\n전체 결과 저장 위치: {OUT_DIR}")


if __name__ == "__main__":
    main()
