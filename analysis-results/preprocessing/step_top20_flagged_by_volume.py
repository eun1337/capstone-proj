"""
최종 처리대상 316개(제외안+부호수정안 합집합, step_return_treatment_comparison.py 참고)
상품을 학습기간(21-23) 총거래규모(총판매수량+총반품수량) 기준으로 정렬해 상위 20개를 본다.

목적: 이 316개가 원래 저볼륨 상품인지, 아니면 고볼륨인데 반품만 왜곡된 건지 판단.
"전체 SKU 중 판매량 순위(백분위)"는 같은 센터 내 전체 상품키 대비 총판매수량 기준
백분위 순위(rank(pct=True)*100, 높을수록 잘 팔리는 상품)로 계산한다.
"""

import pandas as pd

from step_return_treatment_comparison import (
    DAILY_SRC, PRODUCT_KEY_COLS, TRAIN_CUTOFF,
    load_daily, identify_daily_candidates, compare_with_old,
)

OUT_CSV = "analysis-results/preprocessing/step_top20_flagged_by_volume.csv"


def main() -> None:
    print(f"Loading {DAILY_SRC} ...")
    df = load_daily()
    df_train = df[df["거래일"] < TRAIN_CUTOFF]

    new_cand, _ = identify_daily_candidates(df)
    union_keys = compare_with_old(df, new_cand)  # {(센터,바코드,옵션코드,상품클러스터), ...}

    # 센터별 전체 SKU의 학습기간 총판매수량(백분위 순위 산출용)
    all_sku = (
        df_train.groupby(["센터"] + PRODUCT_KEY_COLS)
        .agg(총판매수량=("수량", lambda s: s[s > 0].sum()),
             총반품수량=("수량", lambda s: (-s[s < 0]).sum()),
             상품명=("상품명", "first"))
        .reset_index()
    )
    all_sku["판매량_백분위"] = all_sku.groupby("센터")["총판매수량"].rank(pct=True) * 100
    all_sku["총거래규모"] = all_sku["총판매수량"] + all_sku["총반품수량"]

    key_index = all_sku.set_index(["센터"] + PRODUCT_KEY_COLS).index
    flagged = all_sku[key_index.isin(union_keys)].copy()

    print(f"\n최종 처리대상 316개 중 학습기간 데이터에 존재하는 상품키: {len(flagged)}개")
    top20 = flagged.sort_values("총거래규모", ascending=False).head(20)

    cols = ["상품명", "센터", "바코드", "옵션코드", "상품클러스터", "총판매수량", "총반품수량", "총거래규모", "판매량_백분위"]
    print("\n=== 총거래규모(학습기간, 판매+반품) 상위 20개 ===")
    print(top20[cols].to_string(index=False))

    print("\n분포 참고 (316개 전체):")
    print(flagged["판매량_백분위"].describe())
    print(f"\n판매량 백분위 상위 50% 이상(중간 이상)인 상품 수: {(flagged['판매량_백분위'] >= 50).sum()} / {len(flagged)}")
    print(f"판매량 백분위 상위 10% 이상인 상품 수: {(flagged['판매량_백분위'] >= 90).sum()} / {len(flagged)}")

    flagged.sort_values("총거래규모", ascending=False).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료 -> {OUT_CSV}")


if __name__ == "__main__":
    main()
