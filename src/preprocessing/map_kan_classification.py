"""
map_kan_classification.py
─────────────────────────────────────────────────────────────────────
위치: src/preprocessing/map_kan_classification.py

기능:
  - data/parquet/ 폴더의 기존 parquet 파일을 입력으로 사용
  - 매핑 테이블(바코드+상품명 → KAN 분류 코드) 2단계 매핑:
      1차: 바코드 원본 그대로 매핑
      2차: 1차 실패 행 중 바코드 앞자리 0 제거 후 재매핑
  - 기존 대분류/중분류/소분류 컬럼 제거
  - KAN_CODE(6자리), KAN_대분류, KAN_중분류, KAN_소분류 컬럼 추가
  - 원본 parquet은 유지하고 _mapped.parquet으로 별도 저장

실행 방법:
  python src/preprocessing/map_kan_classification.py
"""

import pandas as pd
from pathlib import Path


BASE_DIR     = Path(__file__).resolve().parents[2]
DATA_DIR     = BASE_DIR / "data"
PARQUET_DIR  = DATA_DIR / "parquet"
MAPPING_FILE = DATA_DIR / "ab_final_product_master.xlsx"


TARGET_FILES = [
    ("a_center_purchase_2021_2024.parquet",  "a_center_purchase_2021_2024_mapped.parquet"),
    ("a_center_sales_2021_2023.parquet",     "a_center_sales_2021_2023_mapped.parquet"),
    ("a_center_sales_2024.parquet",          "a_center_sales_2024_mapped.parquet"),
    ("b_center_purchase_2021_2024.parquet",  "b_center_purchase_2021_2024_mapped.parquet"),
    ("b_center_sales_2021_2023.parquet",     "b_center_sales_2021_2023_mapped.parquet"),
    ("b_center_sales_2024.parquet",          "b_center_sales_2024_mapped.parquet"),
]


MAP_BARCODE  = "바코드"
MAP_NAME     = "상품명"
MAP_KAN_CODE = "KAN_CODE" 
MAP_KAN_L    = "KAN_대분류"
MAP_KAN_M    = "KAN_중분류"
MAP_KAN_S    = "KAN_소분류"

OUT_KAN_CODE = "KAN_CODE"
OUT_KAN_L    = "KAN_대분류"
OUT_KAN_M    = "KAN_중분류"
OUT_KAN_S    = "KAN_소분류"
KAN_COLS     = [OUT_KAN_CODE, OUT_KAN_L, OUT_KAN_M, OUT_KAN_S]

SRC_BARCODE  = "바코드"
SRC_NAME     = "상품명"
OLD_COLS     = ["대분류", "중분류", "소분류"]


def pad_kan_code(series: pd.Series) -> pd.Series:

    def _pad(val):
        if pd.isna(val):
            return val
        return str(val).strip().zfill(6)
    return series.apply(_pad)


def load_mapping(path: Path) -> pd.DataFrame:
    """매핑 테이블 로드 및 전처리"""
    df = pd.read_excel(path, dtype={MAP_BARCODE: str, MAP_KAN_CODE: str})
    df[MAP_BARCODE]  = df[MAP_BARCODE].str.strip()
    df[MAP_NAME]     = df[MAP_NAME].str.strip()
    df[MAP_KAN_CODE] = pad_kan_code(df[MAP_KAN_CODE])
    df = df.drop_duplicates(subset=[MAP_BARCODE, MAP_NAME], keep="first")
    df = df[[MAP_BARCODE, MAP_NAME, MAP_KAN_CODE, MAP_KAN_L, MAP_KAN_M, MAP_KAN_S]].rename(columns={
        MAP_KAN_CODE: OUT_KAN_CODE,
        MAP_KAN_L:    OUT_KAN_L,
        MAP_KAN_M:    OUT_KAN_M,
        MAP_KAN_S:    OUT_KAN_S,
    })
    return df


def merge_kan(df: pd.DataFrame, df_kan: pd.DataFrame, barcode_col: str) -> pd.DataFrame:

    return df[[barcode_col, SRC_NAME]].merge(
        df_kan,
        left_on=[barcode_col, SRC_NAME],
        right_on=[MAP_BARCODE, MAP_NAME],
        how="left"
    )[KAN_COLS]


def process_file(src_path: Path, out_path: Path, df_kan: pd.DataFrame):

    print(f"\n[처리 중] {src_path.name} → {out_path.name}")

    if not src_path.exists():
        print(f"  ⚠️  파일 없음, 건너뜀: {src_path}")
        return


    df = pd.read_parquet(src_path)
    print(f"  ✔ 로드 완료: {len(df):,}행")


    drop_cols = [c for c in OLD_COLS + KAN_COLS if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
        print(f"  ✔ 제거된 컬럼: {drop_cols}")


    df[SRC_BARCODE] = df[SRC_BARCODE].astype(str).str.strip()
    df[SRC_NAME]    = df[SRC_NAME].astype(str).str.strip()


    result1 = merge_kan(df, df_kan, SRC_BARCODE)
    df[KAN_COLS] = result1[KAN_COLS].values

    mapped1   = df[OUT_KAN_CODE].notna().sum()
    fail_mask = df[OUT_KAN_CODE].isna()
    print(f"  ✔ 1차 매핑 성공: {mapped1:,}행 ({mapped1/len(df)*100:.1f}%)")


    if fail_mask.any():

        df_fail = df[fail_mask].copy()
        leading_zero_mask = df_fail[SRC_BARCODE].str.startswith("0")
        df_retry = df_fail[leading_zero_mask].copy()

        if not df_retry.empty:
            df_retry["_barcode_strip"] = df_retry[SRC_BARCODE].str.lstrip("0")

            result2 = merge_kan(df_retry, df_kan, "_barcode_strip")
            df_retry[KAN_COLS] = result2[KAN_COLS].values

            df.loc[df_retry.index, KAN_COLS] = df_retry[KAN_COLS].values
            mapped2 = df_retry[OUT_KAN_CODE].notna().sum()
            print(f"  ✔ 2차 매핑 성공 (앞자리 0 제거): {mapped2:,}행")

    total    = len(df)
    mapped   = df[OUT_KAN_CODE].notna().sum()
    unmapped = total - mapped
    print(f"  ✔ 최종 매핑 성공: {mapped:,}행 ({mapped/total*100:.1f}%)")
    if unmapped:
        print(f"  ⚠️  최종 매핑 실패: {unmapped:,}행 ({unmapped/total*100:.1f}%)")
        sample = (
            df.loc[df[OUT_KAN_CODE].isna(), [SRC_BARCODE, SRC_NAME]]
            .drop_duplicates()
            .head(5)
        )
        print("     실패 샘플 (바코드 | 상품명):")
        for _, row in sample.iterrows():
            print(f"       {row[SRC_BARCODE]} | {row[SRC_NAME]}")


    df.to_parquet(out_path, index=False)
    print(f"  ✔ 저장 완료: {out_path.name}")


def main():
    print("=" * 60)
    print("KAN 분류 매핑 스크립트 시작")
    print("=" * 60)

    print(f"\n[매핑 테이블 로드] {MAPPING_FILE.name}")
    df_kan = load_mapping(MAPPING_FILE)
    print(f"  ✔ 매핑 항목 수: {len(df_kan):,}건")

    sample_codes = df_kan[OUT_KAN_CODE].dropna().head(5).tolist()
    print(f"  ✔ KAN_CODE 샘플 (6자리 확인): {sample_codes}")

    for src_name, out_name in TARGET_FILES:
        process_file(PARQUET_DIR / src_name, PARQUET_DIR / out_name, df_kan)

    print("\n" + "=" * 60)
    print("모든 파일 처리 완료")
    print("=" * 60)


if __name__ == "__main__":
    main()