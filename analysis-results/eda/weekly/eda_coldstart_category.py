"""
Step 2 후속: 콜드스타트 상품(21-23년 학습기간에는 없다가 24년에 처음 등장한
상품/지역 키)의 KAN_대분류/중분류 분포를 확인한다.

- 콜드스타트 키 기준: (센터+시도+시군구+바코드+옵션코드+상품클러스터) 조합이
  21-23년 데이터엔 전혀 없다가 24년에 처음 등장한 경우.
- 콜드스타트 키들의 카테고리 분포가 24년 전체 키 분포와 얼마나 다른지 비교해서,
  특정 카테고리에 콜드스타트가 쏠려 있는지 확인한다.
"""

import pandas as pd

SRC_PATH = "data/final/aggregated_weekly_demand.parquet"
OUT_DAEBUNRYU_PATH = "analysis-results/eda/step2_coldstart_kan_daebunryu.csv"
OUT_JUNGBUNRYU_PATH = "analysis-results/eda/step2_coldstart_kan_jungbunryu.csv"

KEY = ["센터", "시도", "시군구", "바코드", "옵션코드", "상품클러스터"]
TRAIN_CUTOFF = "2024-01-01"


def category_distribution(keys_df: pd.DataFrame, all_2024_keys_df: pd.DataFrame, cat_col: str) -> pd.DataFrame:
    coldstart_dist = keys_df[cat_col].value_counts()
    all_2024_dist = all_2024_keys_df[cat_col].value_counts()

    result = pd.DataFrame(
        {
            "콜드스타트_키수": coldstart_dist,
            "24년_전체_키수": all_2024_dist,
        }
    ).fillna(0).astype(int)

    result["콜드스타트_비율(%)"] = (result["콜드스타트_키수"] / result["콜드스타트_키수"].sum() * 100).round(2)
    result["24년_전체_비율(%)"] = (result["24년_전체_키수"] / result["24년_전체_키수"].sum() * 100).round(2)
    result["콜드스타트_발생률(%)"] = (
        result["콜드스타트_키수"] / result["24년_전체_키수"].replace(0, pd.NA) * 100
    ).round(2)

    result = result.sort_values("콜드스타트_키수", ascending=False)
    result.index.name = cat_col
    return result


def main() -> None:
    print(f"Loading {SRC_PATH} ...")
    df = pd.read_parquet(SRC_PATH)

    train = df[df["주시작일"] < TRAIN_CUTOFF]
    test = df[df["주시작일"] >= TRAIN_CUTOFF]

    train_keys = set(map(tuple, train[KEY].drop_duplicates().values))

    test_key_rows = test.drop_duplicates(subset=KEY)
    is_coldstart = test_key_rows.set_index(KEY).index.map(lambda k: k not in train_keys)
    coldstart_keys_df = test_key_rows[is_coldstart.values]

    print(f"24년 고유 키 수: {len(test_key_rows):,}")
    print(f"콜드스타트 키 수: {len(coldstart_keys_df):,} ({len(coldstart_keys_df) / len(test_key_rows):.2%})\n")

    dae_dist = category_distribution(coldstart_keys_df, test_key_rows, "KAN_대분류")
    jung_dist = category_distribution(coldstart_keys_df, test_key_rows, "KAN_중분류")

    print("=== 콜드스타트 키 KAN_대분류 분포 (vs 24년 전체 키 분포) ===")
    print(dae_dist.to_string())
    print()
    print("=== 콜드스타트 키 KAN_중분류 분포 Top 15 (vs 24년 전체 키 분포) ===")
    print(jung_dist.head(15).to_string())

    dae_dist.to_csv(OUT_DAEBUNRYU_PATH, encoding="utf-8-sig")
    jung_dist.to_csv(OUT_JUNGBUNRYU_PATH, encoding="utf-8-sig")
    print(f"\n저장 완료 -> {OUT_DAEBUNRYU_PATH}")
    print(f"저장 완료 -> {OUT_JUNGBUNRYU_PATH}")


if __name__ == "__main__":
    main()
