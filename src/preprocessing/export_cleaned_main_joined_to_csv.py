"""
export_cleaned_main_joined_to_csv.py
------------------------------------------------------------------
export_cleaned_main_joined.py로 만든 data/final/cleaned_main_joined.parquet을 CSV로 변환한다
- 대용량이라 parquet_to_csv.py와 동일하게 청크 단위로 읽어서 이어쓰기
- 한글 깨짐 방지를 위해 UTF-8-SIG(BOM) 인코딩 사용
"""

import os

import pyarrow.parquet as pq

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FINAL_DIR = os.path.join(BASE_DIR, "data", "final")
IN_FILE = os.path.join(FINAL_DIR, "cleaned_main_joined.parquet")
OUT_FILE = os.path.join(FINAL_DIR, "cleaned_main_joined.csv")

CHUNK_SIZE = 200_000


def main():
    if not os.path.exists(IN_FILE):
        raise FileNotFoundError(
            f"{IN_FILE} 이 없습니다. 먼저 export_cleaned_main_joined.py를 실행해주세요."
        )

    pf = pq.ParquetFile(IN_FILE)
    total_rows = pf.metadata.num_rows
    print(f"총 {total_rows:,}행 변환 시작 -> {OUT_FILE}")

    first_chunk = True
    written_rows = 0

    for batch in pf.iter_batches(batch_size=CHUNK_SIZE):
        df = batch.to_pandas()

        df.to_csv(
            OUT_FILE,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            encoding="utf-8-sig",
            lineterminator="\n",
        )

        written_rows += len(df)
        first_chunk = False
        print(f"  - {written_rows:,}/{total_rows:,}행 저장됨")

    size_mb = os.path.getsize(OUT_FILE) / (1024 * 1024)
    print(f"\n변환 완료: {OUT_FILE} ({size_mb:.1f} MB, {written_rows:,}행)")


if __name__ == "__main__":
    main()
