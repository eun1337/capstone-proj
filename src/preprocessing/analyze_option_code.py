"""
analyze_option_code.py

설명: data/final 폴더의 데이터에서 '옵션코드' 컬럼의 고유값과 빈도수를 확인하고,
    정규화(OPTION_MAP) 초안을 제안합니다. 실제 변환은 하지 않는 분석 전용 스크립트입니다.

"""

import glob
import os
import re
import unicodedata

import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FINAL_DIR = os.path.join(BASE_DIR, "data", "final")

COL = "옵션코드"

# 같은 데이터가 여러 포맷으로 존재할 때 우선순위 (앞에 있을수록 우선)
EXT_PRIORITY = [".parquet", ".csv", ".xlsx", ".xls"]


def pick_files(final_dir: str) -> list[str]:
    """파일명(확장자 제외)이 같으면 EXT_PRIORITY 순으로 한 파일만 선택."""
    by_stem: dict[str, str] = {}

    for ext in EXT_PRIORITY:
        for path in glob.glob(os.path.join(final_dir, f"*{ext}")):
            stem = os.path.splitext(os.path.basename(path))[0]
            by_stem.setdefault(stem, path)

    return sorted(by_stem.values())


def read_option_col(path: str) -> pd.Series:
    """파일 확장자에 맞게 옵션코드 컬럼만 읽어옴."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".parquet":
        df = pd.read_parquet(path, columns=[COL])
    elif ext == ".csv":
        df = pd.read_csv(path, usecols=[COL], dtype=str)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(path, usecols=[COL], dtype=str)
    else:
        raise ValueError(f"지원하지 않는 확장자: {ext}")

    return df[COL]


def main():
    files = pick_files(FINAL_DIR)
    if not files:
        print(f"[!] {FINAL_DIR} 에서 읽을 파일을 찾지 못했습니다.")
        return

    print("[ 옵션코드 분석 대상 파일 ]")
    for f in files:
        print(f"  - {os.path.relpath(f, BASE_DIR)}")

    series_list = []
    for f in files:
        s = read_option_col(f)
        print(f"\n{os.path.basename(f)}: {len(s):,}행 읽음 (결측 {s.isna().sum():,}건)")
        series_list.append(s)

    all_values = pd.concat(series_list, ignore_index=True)

    counts = all_values.value_counts(dropna=False).sort_values(ascending=False)

    print("\n" + "=" * 50)
    print(f"[ '{COL}' 고유값 및 빈도수 (총 {counts.shape[0]}개, 내림차순) ]")
    print("=" * 50)
    for value, cnt in counts.items():
        display_value = "<NaN>" if pd.isna(value) else repr(value)
        print(f"  {display_value:<15} {cnt:,}")

    print_option_map_draft(counts.index.tolist())


def normalize_guess(value) -> str | None:
    """공백/전각문자/대소문자 차이를 없앤 뒤 표준 코드를 추정."""
    if pd.isna(value):
        return None

    v = unicodedata.normalize("NFKC", str(value))  # 전각 문자(／ 등) -> 반각 통일
    v = re.sub(r"\s+", "", v).upper()
    v = v.strip("/").replace("/", "")

    if v in ("BOX", "BX", "박스") or re.fullmatch(r"BX\d+", v):
        return "BX"
    if v in ("CS", "C S") or re.fullmatch(r"C?S\d*", v) and "S" in v:
        return "CS"
    if v in ("EA", "E A"):
        return "EA"
    return None


def print_option_map_draft(unique_values: list) -> None:
    print("\n" + "=" * 50)
    print("[ OPTION_MAP 딕셔너리 초안 ]")
    print("(자동 추정 결과이며, 확정 전 반드시 검토/수정하세요)")
    print("=" * 50)

    mapped, unmapped = {}, []
    for value in unique_values:
        if pd.isna(value):
            continue
        guess = normalize_guess(value)
        if guess:
            mapped[value] = guess
        else:
            unmapped.append(value)

    print("\nOPTION_MAP = {")
    for original, target in mapped.items():
        print(f"    {original!r}: {target!r},")
    print("}")

    if unmapped:
        print("\n# [!] 아래 값들은 자동 규칙으로 판단하지 못했습니다. 수동으로 확인 후 매핑을 추가하세요:")
        for value in unmapped:
            print(f"#   {value!r}")


if __name__ == "__main__":
    main()
