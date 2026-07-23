# 04_spike_day_sku_concentration_analysis.py
# 레짐 체인지로 보이는 현상이 (a) 데이터 기록 방식 문제인지, (b) 특정 SKU의 정기 대량납품
# 패턴인지 구분하기 위해, 스파이크일에 소수 SKU가 물량을 몰아서 만드는지 확인한다.
# 기준 파일: data/final/cleaned_main_joined_cleaned_for_pred.parquet

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
INPUT_PATH = DATA_DIR / "final" / "cleaned_main_joined_cleaned_for_pred.parquet"
OUT_DIR = BASE_DIR / "analysis-results" / "eda" / "overall" / "04_spike_day_sku_concentration_analysis"
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


def analyze_spike_day_sku_concentration(df: pd.DataFrame, center="B",
                                          regime_transition="2023-05-01", top_k=10):
    print("=" * 60)
    print(f"[센터 {center}] 스파이크일 SKU 집중도 분석")
    print("=" * 60)

    transition = pd.Timestamp(regime_transition)
    sub = df[(df[CENTER_COL] == center) & (df[DATE_COL] < transition)]

    daily_total = sub.groupby(DATE_COL)[TARGET_COL].sum().asfreq("D", fill_value=0)
    spike_days = daily_total[daily_total > daily_total.median() * 3].index
    normal_days = daily_total[daily_total <= daily_total.median() * 3].index

    normal_days = daily_total.loc[normal_days][daily_total.loc[normal_days] > 0].index

    if len(spike_days) == 0:
        print(f"\n- 센터 {center}: threshold(median*3) 기준 스파이크일이 없습니다. "
              f"이 센터는 애초에 레짐 체인지 의심 대상이 아닌 것으로 보입니다 (분석 스킵).")
        return None, None, None

    def day_concentration(day):
        day_df = sub[sub[DATE_COL] == day]
        by_sku = day_df.groupby("SKU_KEY")[TARGET_COL].sum().abs().sort_values(ascending=False)
        total = by_sku.sum()
        top_k_share = by_sku.head(top_k).sum() / total if total else np.nan
        return {
            "거래일": day, "참여_SKU수": day_df["SKU_KEY"].nunique(),
            "총_거래량": total, f"상위{top_k}개_SKU_비중": top_k_share,
        }

    spike_stats = pd.DataFrame([day_concentration(d) for d in spike_days])
    normal_stats = pd.DataFrame([day_concentration(d) for d in
                                   np.random.choice(normal_days, size=min(30, len(normal_days)), replace=False)])

    print(f"\n- 스파이크일({len(spike_days)}일) 평균 참여 SKU 수: {spike_stats['참여_SKU수'].mean():.0f}개, "
          f"상위{top_k}개 SKU 비중: {spike_stats[f'상위{top_k}개_SKU_비중'].mean():.1%}")
    print(f"- 평소일(샘플 {len(normal_stats)}일) 평균 참여 SKU 수: {normal_stats['참여_SKU수'].mean():.0f}개, "
          f"상위{top_k}개 SKU 비중: {normal_stats[f'상위{top_k}개_SKU_비중'].mean():.1%}")

    print(f"\n- 스파이크일 상세:")
    print(spike_stats.to_string(index=False))


    top_sku_on_spikes = []
    for d in spike_days:
        day_df = sub[sub[DATE_COL] == d]
        by_sku = day_df.groupby("SKU_KEY")[TARGET_COL].sum().abs().sort_values(ascending=False)
        top_sku_on_spikes.extend(by_sku.head(top_k).index.tolist())
    recurring = pd.Series(top_sku_on_spikes).value_counts()
    print(f"\n- 스파이크일 상위{top_k}에 반복 등장하는 SKU (총 {len(spike_days)}번 중 등장 횟수):")
    print(recurring.head(15))

    spike_stats.to_csv(OUT_DIR / f"center_{center}_spike_day_concentration.csv",
                        index=False, encoding="utf-8-sig")

    print(f"\n※ 해석 기준:")
    print(f"  - 참여 SKU 수가 평소와 비슷하고 상위{top_k}개 비중이 낮다 → 전반적 현상(기록 방식 문제 쪽에 무게)")
    print(f"  - 참여 SKU 수가 적거나 상위{top_k}개 비중이 매우 높다 → 소수 SKU의 정기 대량납품 쪽에 무게")
    print(f"  - 같은 SKU들이 매달 반복 등장한다 → 정기 계약 납품일 가능성 높음")

    return spike_stats, normal_stats, recurring


def main():
    df = load_data()
    for center in ["A", "B"]:
        analyze_spike_day_sku_concentration(df, center=center)
        print()
    print(f"\n모든 분석 결과가 {OUT_DIR} 에 저장되었습니다.")


if __name__ == "__main__":
    main()
