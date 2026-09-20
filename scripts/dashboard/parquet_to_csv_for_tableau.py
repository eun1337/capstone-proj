"""data/dashboard/*.parquet -> data/dashboard/csv/*.csv 일괄 변환 (Tableau Cloud 업로드용).

Tableau Desktop/Prep이 한글 컬럼명/값을 깨지지 않게 읽도록 utf-8-sig(BOM)로 저장한다.
"""

from pathlib import Path

import pandas as pd

DASHBOARD_DIR = Path(__file__).resolve().parents[2] / "data" / "dashboard"
OUT_DIR = DASHBOARD_DIR / "csv"


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    parquet_files = sorted(DASHBOARD_DIR.glob("*.parquet"))
    if not parquet_files:
        print(f"parquet 파일을 찾을 수 없습니다: {DASHBOARD_DIR}")
        return

    for src in parquet_files:
        dst = OUT_DIR / (src.stem + ".csv")
        df = pd.read_parquet(src)
        df.to_csv(dst, index=False, encoding="utf-8-sig")
        print(f"{src.name:55s} {len(df):>10,} rows -> {dst.relative_to(DASHBOARD_DIR.parents[1])} "
              f"({dst.stat().st_size / 1e6:,.1f} MB)")


if __name__ == "__main__":
    main()
