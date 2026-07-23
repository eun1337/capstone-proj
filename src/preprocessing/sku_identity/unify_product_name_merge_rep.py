# unify_product_name_merge_rep.py
# data/csv/sku_key_diagnosis/center_A 의 판정결과 xlsx를 읽어,
# (바코드, 옵션코드_정규화)가 동일한 그룹 내에서 처리유형이 'MERGE'인 행들에 한해
# 권장대표상품명을 아래 우선순위로 재계산하여 덮어쓴다.
#   1순위: 마지막거래일이 가장 최근인 행의 원본상품명
#   2순위(동률 시): 거래건수가 가장 큰 행의 원본상품명
# 처리유형이 MERGE가 아닌 행(SPLIT_PRODUCT 등)과 다른 모든 컬럼은 절대 수정하지 않는다.

from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data" / "csv" / "sku_key_diagnosis" / "center_A"
INPUT_PATH = DATA_DIR / "상품명_충돌_판정결과_최종_상세판단_색상_A.xlsx"
OUTPUT_PATH = DATA_DIR / "상품명_충돌_판정결과_최종_상세판단_색상_A_대표명적용.xlsx"

SHEET_NAME = 0

COL_BARCODE = "바코드"
COL_OPTION = "옵션코드_정규화"
COL_TYPE = "처리유형"
COL_ORIG_NAME = "원본상품명"
COL_TXN_COUNT = "거래건수"
COL_LAST_DATE = "마지막거래일"
COL_REP_NAME = "권장대표상품명"

MERGE_VALUE = "MERGE"


def main():
    df = pd.read_excel(INPUT_PATH, sheet_name=SHEET_NAME, dtype=str)

    df["_txn_count_num"] = pd.to_numeric(df[COL_TXN_COUNT], errors="coerce")
    df["_last_date_dt"] = pd.to_datetime(df[COL_LAST_DATE], errors="coerce")

    merge_mask = df[COL_TYPE] == MERGE_VALUE

    tmp = df.loc[merge_mask, [COL_BARCODE, COL_OPTION, COL_ORIG_NAME, "_txn_count_num", "_last_date_dt"]].copy()

    tmp_sorted = tmp.sort_values(
        by=[COL_BARCODE, COL_OPTION, "_last_date_dt", "_txn_count_num"],
        ascending=[True, True, False, False]
    )

    rep = (
        tmp_sorted
        .groupby([COL_BARCODE, COL_OPTION], as_index=False)
        .first()[[COL_BARCODE, COL_OPTION, COL_ORIG_NAME]]
        .rename(columns={COL_ORIG_NAME: "_대표상품명"})
    )

    df = df.merge(rep, on=[COL_BARCODE, COL_OPTION], how="left")

    df.loc[merge_mask, COL_REP_NAME] = df.loc[merge_mask, "_대표상품명"]

    df = df.drop(columns=["_대표상품명", "_txn_count_num", "_last_date_dt"])

    df.to_excel(OUTPUT_PATH, index=False)

    print(f"완료: MERGE 행 {merge_mask.sum()}건 중 대표상품명 재계산 -> {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
