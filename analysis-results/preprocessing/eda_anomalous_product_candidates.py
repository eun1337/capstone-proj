"""
Step 3 후속: 정상적인 판매/반품 관계로 보기 어려운 상품(회수·조정용 더미 코드,
또는 부호가 뒤집혀 기록되는 것으로 의심되는 상품)의 후보 리스트를 뽑는다.

처음엔 상품명에 "XXX"/"!!"가 포함된 상품(`말통]공XXX`, `!!야채행사...`)을 의심했으나,
실제로 확인해보니 XXX/!!는 전체 상품의 약 10%(2,589개)에 쓰이는 흔한 표기 방식이고
(예: `XXX카스473캔(가성비)XXX`, `!!흙대파1묶음(5단)국내산` 등 전부 정상 상품) 음수 비율도
낮아 신호로 쓸 수 없었다. 대신 아래 기준이 훨씬 신뢰도 높은 신호로 확인됨:

플래그 기준: 거래건수 10건 이상 & 수량<0 비율 90% 이상
  - 정상 상품은 반품보다 판매가 훨씬 많은 게 정상. 이 비율이 비정상적으로 높으면
    "반품"이 아니라 애초에 회수/차감 전용으로 쓰이거나, 해당 SKU의 수량 부호가
    구조적으로 반대로 기록되고 있을 가능성이 있음.
  - 실제로 이 기준으로 걸러지는 상품의 84%(175개 중 147개)가 KAN_중분류
    "즉석/편의식품"·"과자류"에 집중되어 있고, 전부 EA(낱개) 옵션코드다. 라면/과자/껌
    처럼 흔히 팔리는 상품이 대부분 음수로 찍히는 건 실제 반품 행동으로 보기 어려워,
    EA 옵션코드 특정 카테고리에서 수량 부호가 구조적으로 반대로 들어가는 데이터
    파이프라인 이슈일 가능성을 협업자와 확인해야 한다.
"""

import pandas as pd

SRC_PATH = "data/final/cleaned_main_joined_cleaned_for_pred.parquet"
OUT_PATH = "analysis-results/preprocessing/step3_anomalous_product_candidates.csv"

MIN_TXN_FOR_FLAG = 10
NEG_RATIO_THRESHOLD = 0.9

KEY = ["바코드", "옵션코드", "상품클러스터"]


def main() -> None:
    print(f"Loading {SRC_PATH} ...")
    df = pd.read_parquet(SRC_PATH)
    print(f"Loaded {len(df):,} rows\n")

    grouped = (
        df.groupby(KEY)
        .agg(
            상품명=("상품명", "first"),
            KAN_대분류=("KAN_대분류", "first"),
            KAN_중분류=("KAN_중분류", "first"),
            거래건수=("수량", "count"),
            수량합계=("수량", "sum"),
            음수건수=("수량", lambda s: (s < 0).sum()),
        )
        .reset_index()
    )
    grouped["수량음수비율(%)"] = (grouped["음수건수"] / grouped["거래건수"] * 100).round(2)

    flag = (grouped["거래건수"] >= MIN_TXN_FOR_FLAG) & (
        grouped["수량음수비율(%)"] >= NEG_RATIO_THRESHOLD * 100
    )
    candidates = grouped[flag].sort_values("거래건수", ascending=False).copy()

    print(f"전체 상품키 수: {len(grouped):,}")
    print(f"후보 상품키 수 (거래건수>={MIN_TXN_FOR_FLAG} & 음수비율>={NEG_RATIO_THRESHOLD:.0%}): {len(candidates):,}\n")

    print("KAN_중분류 분포 (후보 상품 기준, 상위 10개):")
    print(candidates["KAN_중분류"].value_counts().head(10).to_string())
    print()

    print("옵션코드 분포 (후보 상품 기준):")
    print(candidates["옵션코드"].value_counts().to_string())
    print()

    print("거래건수 상위 20개 후보:")
    print(candidates.head(20).to_string(index=False))

    candidates.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
