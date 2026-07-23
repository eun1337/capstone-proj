"""
Step 2: 주 단위 집계 데이터(aggregated_weekly_demand.parquet)에 대한 Zero-Ratio(희소성) 측정.

1) 현재 존재하는 행 기준 Zero-Ratio
   - 총판매수량 == 0 인 행 비율
   - 반품수량 == 0 인 행 비율

2) 시계열 Grid Expansion 관점의 희소성
   - (상품/지역 키) x (전체 주차) 이론상 모든 조합을 만들었을 때,
     실제 거래가 있어 존재하는 행이 전체 그리드의 몇 %인지, 나머지(Zero-Demand Week)가
     몇 %인지 산출.

21-24년 전체와, 실제 학습에 쓸 21-23년만 따로 계산해서 같이 비교한다. 24년 정보가 섞인
전체 기준 희소성 수치를 그대로 전처리/모델링 기준으로 확정하면 leakage가 되므로, 21-23년
쪽을 기준으로 삼고 21-24년 쪽은 "24년 들어 희소성이 달라졌는지" 참고용으로 비교한다.

주의: 집계 파일 자체가 "거래가 1건이라도 있었던 조합"만 남아있는 구조라, 실제로 거래가
없는 주(수요=0)는 애초에 행 자체가 존재하지 않는다. 이 규모를 추정하기 위한 것으로,
실제로 0을 채워 넣는 건 이후 단계에서 모델링 방식에 따라 결정한다.
"""

import pandas as pd

SRC_PATH = "data/final/aggregated_weekly_demand.parquet"
OUT_PATH = "analysis-results/eda/step2_sparsity_summary.csv"

REGION_PRODUCT_KEY = ["센터", "시도", "시군구", "바코드", "옵션코드", "상품클러스터"]

# 21-23년 학습 기간과 24년 예측 대상 기간의 경계.
# 주시작일=2020-12-28 주는 실제 거래일이 2021-01-02(토)라 21-23년 쪽에 포함된다.
TRAIN_CUTOFF = "2024-01-01"


def compute_sparsity(df: pd.DataFrame) -> dict:
    n_rows = len(df)

    zero_sales = (df["총판매수량"] == 0).sum()
    zero_returns = (df["반품수량"] == 0).sum()

    n_weeks = df["주시작일"].nunique()
    n_keys = df.drop_duplicates(subset=REGION_PRODUCT_KEY).shape[0]
    theoretical_rows = n_keys * n_weeks
    missing_rows = theoretical_rows - n_rows
    missing_ratio = missing_rows / theoretical_rows if theoretical_rows else float("nan")

    return {
        "실제 존재 행 수": n_rows,
        "총판매수량==0 행 건수": zero_sales,
        "총판매수량==0 비율(%)": round(zero_sales / n_rows * 100, 4),
        "반품수량==0 행 건수": zero_returns,
        "반품수량==0 비율(%)": round(zero_returns / n_rows * 100, 4),
        "전체 주차 수": n_weeks,
        "고유 상품/지역 키 조합 수": n_keys,
        "이론상 전체 가능 행 수": theoretical_rows,
        "Zero-Demand Week 건수": missing_rows,
        "Zero-Demand Week 비율(%)": round(missing_ratio * 100, 4),
    }


def print_summary(title: str, stats: dict) -> None:
    print(f"=== {title} ===")
    print(f"실제 존재 행 수                              : {stats['실제 존재 행 수']:,}")
    print(f"총판매수량 == 0 인 행                        : {stats['총판매수량==0 행 건수']:,} ({stats['총판매수량==0 비율(%)']}%)")
    print(f"반품수량   == 0 인 행                        : {stats['반품수량==0 행 건수']:,} ({stats['반품수량==0 비율(%)']}%)")
    print(f"전체 주차 수 (주시작일 고유값)              : {stats['전체 주차 수']:,}")
    print(f"고유 상품/지역 키 조합 수                    : {stats['고유 상품/지역 키 조합 수']:,}")
    print(f"이론상 전체 가능 행 수 (키 수 x 주차 수)      : {stats['이론상 전체 가능 행 수']:,}")
    print(f"Zero-Demand Week 건수 (이론상 - 실제)        : {stats['Zero-Demand Week 건수']:,}")
    print(f"Zero-Demand Week 비율                        : {stats['Zero-Demand Week 비율(%)']}%")
    print()


def main() -> None:
    print(f"Loading {SRC_PATH} ...")
    df = pd.read_parquet(SRC_PATH)
    train_df = df[df["주시작일"] < TRAIN_CUTOFF]
    print(f"Loaded {len(df):,} rows (21-24 전체), 학습기간(21-23) {len(train_df):,}행\n")

    full_stats = compute_sparsity(df)
    train_stats = compute_sparsity(train_df)

    print_summary("[1+2] 전체(21-24) 기준 희소성", full_stats)
    print_summary("[1+2] 학습기간(21-23) 기준 희소성", train_stats)

    summary = pd.DataFrame(
        [{"기간": "전체(21-24)", **full_stats}, {"기간": "학습기간(21-23)", **train_stats}]
    )
    summary.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")
    print(f"저장 완료 -> {OUT_PATH}")


if __name__ == "__main__":
    main()
