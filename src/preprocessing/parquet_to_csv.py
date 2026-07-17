"""
- Tableau Public / MySQL LOAD DATA INFILE 모두에서 바로 쓸 수 있도록 CSV로 저장
- 한글 깨짐 방지를 위해 UTF-8-SIG(BOM) 인코딩 사용 (엑셀/Tableau 호환)
- 대용량 파일은 청크 단위로 읽어서 메모리 절약 + 한 번에 이어쓰기

실행 위치: src/preprocessing 에서 실행한다고 가정 (필요시 PARQUET_DIR, OUTPUT_DIR 경로만 수정)
"""

import os
import glob
import pandas as pd
import pyarrow.parquet as pq


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARQUET_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "parquet")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "..", "..", "data", "csv")
CHUNK_SIZE = 200_000                  


def convert_parquet_to_csv(parquet_path: str, output_dir: str, chunk_size: int = CHUNK_SIZE):
    file_name = os.path.splitext(os.path.basename(parquet_path))[0]
    csv_path = os.path.join(output_dir, f"{file_name}.csv")

    pf = pq.ParquetFile(parquet_path)
    total_rows = pf.metadata.num_rows
    print(f"[{file_name}] 총 {total_rows:,}행 변환 시작 -> {csv_path}")

    first_chunk = True
    written_rows = 0

    for batch in pf.iter_batches(batch_size=chunk_size):
        df = batch.to_pandas()

        df.to_csv(
            csv_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            encoding="utf-8-sig",   
            lineterminator="\n",
        )

        written_rows += len(df)
        first_chunk = False
        print(f"  - {written_rows:,}/{total_rows:,}행 저장됨")

    print(f"[{file_name}] 변환 완료. 최종 행 수: {written_rows:,}\n")
    return csv_path, written_rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"parquet 폴더 경로: {os.path.abspath(PARQUET_DIR)}")
    parquet_files = sorted(glob.glob(os.path.join(PARQUET_DIR, "*_mapped*.parquet")))

    if not parquet_files:
        print(f"'{PARQUET_DIR}' 경로에 '_mapped'가 포함된 parquet 파일이 없습니다. 경로를 확인해주세요.")
        return

    print(f"총 {len(parquet_files)}개 parquet 파일 발견\n")

    summary = []
    for path in parquet_files:
        csv_path, rows = convert_parquet_to_csv(path, OUTPUT_DIR)
        summary.append((os.path.basename(csv_path), rows))

    print("=" * 50)
    print("변환 요약")
    print("=" * 50)
    for name, rows in summary:
        print(f"{name}: {rows:,}행")


if __name__ == "__main__":
    main()