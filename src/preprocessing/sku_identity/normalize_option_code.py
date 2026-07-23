"""
normalize_option_code.py
------------------------------------------------------------------
센터별(A/B) 입고·매출 mapped csv 원본에서 옵션코드 컬럼을 정규화하여
data/csv/option_normalized/ 에 동일한 파일명으로 저장합니다.

원본 컬럼은 하나도 삭제/변경하지 않고, 정규화 결과를 '옵션코드_정규화' 컬럼으로
추가만 합니다. 단 '일자/판매일'->'거래일', '판매수량'->'수량' 컬럼명 통일은
후속 단계(barcode_option_multi_name_detail)에서 그대로 재사용할 수 있도록
이 단계에서 함께 적용합니다.

옵션코드 정규화 규칙
  1) 공백 제거 후 대문자화
  2) BOX/박스 -> BX, C/S, C／S -> CS, E/A -> EA (기본 매핑)
  3) 앞에 숫자 '1'이 붙어 매핑에서 누락되던 값(예: '1C/S', '1EA', '1BOX')은
     '1' + 알려진 코드 전체와 정확히 일치할 때만 '1'을 제거한 뒤 매핑
  4) 뒤에 숫자 접미사가 붙은 값(예: 'BX2', 'CS1', 'C/S1')은 접미사를 제거하고
     기본 코드로 병합. 실제 데이터에서 접미사 코드가 기본 코드와 같은 바코드에
     섞여 등장하는 비율이 20~50%로 확인되어(표기 노이즈로 판단), 접미사를 별도
     코드로 유지하지 않고 병합함. 원본 표기는 '옵션코드' 컬럼에 그대로 남아있어
     추후 이 판단을 재검토할 수 있음.

입력 : data/csv/{a,b}_center_{purchase,sales}_*_mapped.csv
출력 : data/csv/option_normalized/{a,b}_center_{purchase,sales}_*_mapped.csv
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
CSV_DIR = BASE_DIR / "data" / "csv"
OUT_DIR = CSV_DIR / "option_normalized"

INPUT_FILES = {
    "A": [
        "a_center_purchase_2021_2024_mapped.csv",
        "a_center_sales_2021_2023_mapped.csv",
        "a_center_sales_2024_mapped.csv",
    ],
    "B": [
        "b_center_purchase_2021_2024_mapped.csv",
        "b_center_sales_2021_2023_mapped.csv",
        "b_center_sales_2024_mapped.csv",
    ],
}

RENAME_MAP = {
    "일자": "거래일",
    "판매일": "거래일",
    "판매수량": "수량",
}

REQUIRED_COLUMNS = ["거래일", "바코드", "상품명", "옵션코드", "규격", "입수", "수량"]

OPTION_MAP = {"BOX": "BX", "박스": "BX", "C/S": "CS", "C／S": "CS", "E/A": "EA"}

# 앞에 '1'이 붙어 원래 코드가 가려진 경우만 벗겨낸다.
LEADING_ONE_PATTERN = re.compile(r"^1(BOX|박스|C/S|C／S|E/A|BX|CS|EA)$")

# 뒤에 숫자 접미사가 붙은 경우(BX2, CS1, C/S1 등) 접미사를 제거하고 기본 코드로 병합한다.
SUFFIX_PATTERN = re.compile(r"^(BOX|박스|C/S|C／S|E/A|BX|CS|EA)\d+$")


def read_csv_safely(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"입력 파일이 없습니다: {path}")

    for encoding in ("utf-8-sig", "utf-8", "cp949"):
        try:
            return pd.read_csv(path, encoding=encoding, dtype=str, low_memory=False)
        except UnicodeDecodeError:
            continue

    raise RuntimeError(f"파일 인코딩을 판별하지 못했습니다: {path}")


def standardize_columns(df: pd.DataFrame, filename: str) -> pd.DataFrame:
    df = df.rename(columns=RENAME_MAP)

    missing = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing:
        raise ValueError(f"[{filename}] 필수 컬럼 누락: {missing} / 현재: {df.columns.tolist()}")

    return df


def normalize_option_code(value: object) -> str:
    if pd.isna(value):
        return "(미상)"

    option = re.sub(r"\s+", "", str(value).strip().upper())

    leading_match = LEADING_ONE_PATTERN.match(option)
    if leading_match:
        option = leading_match.group(1)

    suffix_match = SUFFIX_PATTERN.match(option)
    if suffix_match:
        option = suffix_match.group(1)

    return OPTION_MAP.get(option, option or "(미상)")


def process_file(center: str, filename: str) -> None:
    path = CSV_DIR / filename
    df = standardize_columns(read_csv_safely(path), filename)

    before_unmapped = sorted(
        set(df["옵션코드"].dropna().astype(str).str.strip())
        - set(OPTION_MAP.keys())
    )

    df["옵션코드_정규화"] = df["옵션코드"].map(normalize_option_code)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUT_DIR / filename
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    def was_adjusted(raw_value: str) -> bool:
        cleaned = re.sub(r"\s+", "", raw_value.upper())
        return bool(LEADING_ONE_PATTERN.match(cleaned) or SUFFIX_PATTERN.match(cleaned))

    fixed = sorted(v for v in before_unmapped if was_adjusted(v))

    print(f"[{center}] {filename:<45} {len(df):>10,}행 -> {output_path.name}")
    if fixed:
        print(f"    '1+코드' 또는 '코드+숫자접미사' 형태로 보정된 원본값: {fixed}")


def main() -> None:
    for center, files in INPUT_FILES.items():
        print("\n" + "-" * 80)
        print(f"[센터 {center}]")
        for filename in files:
            process_file(center, filename)

    print("\n" + "-" * 80)
    print(f"완료: {OUT_DIR}")


if __name__ == "__main__":
    main()
