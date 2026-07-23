"""
create_barcode_option_multi_name_detail.py
------------------------------------------------------------------
정규화가 끝난(data/csv/option_normalized/) 센터별(A/B) 입고·매출 데이터를
통합한 뒤, 동일한 '바코드 + 정규화 옵션코드'에 서로 다른 상품명이 2개 이상
연결된 조합을 추출하여 상품명별로 상세 집계합니다.

옵션코드 정규화는 normalize_option_code.py 에서 선행 수행되며,
이 스크립트는 그 결과(옵션코드_정규화 컬럼 포함)를 그대로 읽어서 사용합니다.
이 스크립트에서는 옵션코드 정규화를 다시 수행하지 않습니다.

등장배치(원본 파일 단위)는 상품명 충돌 원인 판정의 핵심 근거입니다.
파일마다 상품 마스터가 달라 같은 기간에도 표기가 갈리므로, 배치가 서로소이면
마스터 불일치, 한 배치 안에서 표기가 갈리면 실제 변경으로 해석합니다.
데이터유형(purchase/sales)은 등장배치에서 파생되므로 별도로 두지 않습니다.
판정 자체는 이 스크립트에서 하지 않고 후속 스크립트에서 수행합니다.

입력 : data/csv/option_normalized/{a,b}_center_{purchase,sales}_*_mapped.csv
출력 : data/csv/sku_key_diagnosis/center_{A,B}/barcode_option_multi_name_detail.csv
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
NORMALIZED_DIR = BASE_DIR / "data" / "csv" / "option_normalized"
OUT_DIR = BASE_DIR / "data" / "csv" / "sku_key_diagnosis"

INPUT_FILES = {
    "A": [
        ("a_center_purchase_2021_2024_mapped.csv", "purchase_2021_2024"),
        ("a_center_sales_2021_2023_mapped.csv", "sales_2021_2023"),
        ("a_center_sales_2024_mapped.csv", "sales_2024"),
    ],
    "B": [
        ("b_center_purchase_2021_2024_mapped.csv", "purchase_2021_2024"),
        ("b_center_sales_2021_2023_mapped.csv", "sales_2021_2023"),
        ("b_center_sales_2024_mapped.csv", "sales_2024"),
    ],
}

REQUIRED_COLUMNS = ["거래일", "바코드", "상품명", "옵션코드_정규화", "규격", "입수", "수량"]

GROUP_KEYS = ["바코드", "옵션코드_정규화"]
DETAIL_KEYS = GROUP_KEYS + ["규격", "입수_정규화", "상품명"]

OUTPUT_COLUMNS = [
    "바코드",
    "옵션코드_정규화",
    "규격",
    "입수_정규화",
    "상품명수",
    "상품명",
    "거래건수",
    "총수량",
    "첫거래일",
    "마지막거래일",
    "등장배치수",
    "등장배치",
]


def read_csv_safely(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"정규화된 입력 파일이 없습니다: {path}\n"
            f"normalize_option_code.py 를 먼저 실행했는지 확인하세요."
        )

    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str, low_memory=False)
        except UnicodeDecodeError:
            continue

    raise RuntimeError(f"파일 인코딩을 판별하지 못했습니다: {path}")


def validate_columns(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"[{filename}] 필수 컬럼 누락: {missing} / 현재: {df.columns.tolist()}")

    return df[REQUIRED_COLUMNS]


def clean_text(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.replace("[\\s\u3000]+", " ", regex=True)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA, "<NA>": pd.NA})
    )


def normalize_pack_size(value: object) -> str:
    if pd.isna(value) or not str(value).strip():
        return "(미상)"

    text = str(value).strip()
    try:
        number = float(text)
    except ValueError:
        return text

    return str(int(number)) if number.is_integer() else str(number)


def to_number(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype("string").str.replace(",", "", regex=False), errors="coerce"
    )


def join_unique(values: pd.Series) -> str:
    return ",".join(sorted(set(values)))


def load_center_data(center: str) -> pd.DataFrame:
    frames = []
    for filename, batch in INPUT_FILES[center]:
        df = validate_columns(read_csv_safely(NORMALIZED_DIR / filename), filename)
        df["등장배치"] = batch
        frames.append(df)
        print(f"로드: {filename:<45} {len(df):>12,}행 | {batch}")

    data = pd.concat(frames, ignore_index=True)

    for column in ("바코드", "상품명", "규격", "옵션코드_정규화"):
        data[column] = clean_text(data[column]).fillna("(미상)")

    data["입수_정규화"] = clean_text(data["입수"]).map(normalize_pack_size)
    data["거래일"] = pd.to_datetime(data["거래일"], errors="coerce")
    data["수량_숫자"] = to_number(data["수량"])

    return data


def build_multi_name_detail(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    name_counts = data.groupby(GROUP_KEYS, dropna=False)["상품명"].nunique()
    multi_name_keys = name_counts[name_counts >= 2].rename("상품명수").reset_index()

    selected = data.merge(multi_name_keys, on=GROUP_KEYS, how="inner")

    detail = (
        selected.groupby(DETAIL_KEYS, dropna=False)
        .agg(
            거래건수=("상품명", "size"),
            총수량=("수량_숫자", "sum"),
            첫거래일=("거래일", "min"),
            마지막거래일=("거래일", "max"),
            등장배치수=("등장배치", "nunique"),
            등장배치=("등장배치", join_unique),
        )
        .reset_index()
        .merge(multi_name_keys, on=GROUP_KEYS, how="left")
        .sort_values(
            ["바코드", "옵션코드_정규화", "거래건수", "상품명"],
            ascending=[True, True, False, True],
        )
        .reset_index(drop=True)[OUTPUT_COLUMNS]
    )

    return multi_name_keys, detail


def main() -> None:
    for center in INPUT_FILES:
        print("\n" + "-" * 80)
        print(f"[센터 {center}]")

        data = load_center_data(center)
        multi_name_keys, detail = build_multi_name_detail(data)

        output_dir = OUT_DIR / f"center_{center}"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "barcode_option_multi_name_detail.csv"
        detail.to_csv(output_path, index=False, encoding="utf-8-sig")

        print(f"통합 원본 행 수              : {len(data):,}")
        print(f"다중 상품명 바코드+옵션 조합 : {len(multi_name_keys):,}")
        print(f"상세 결과 행 수              : {len(detail):,}")
        print(f"저장 완료                    : {output_path}")


if __name__ == "__main__":
    main()
