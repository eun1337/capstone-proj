"""
export_cleaned_main_joined.py
------------------------------------------------------------------
MySQL의 cleaned_main_joined_v1(피처엔지니어링 직전 최종 정제 테이블,
매출 기준 356만행)을 로컬 parquet 파일로 내보낸다. 같은 DB에 접속할 수
없는 협업자와 파일로 데이터를 공유하기 위한 용도.

- 대용량(356만행)이라 pandas.read_sql을 청크 단위로 읽어 메모리 부담을 줄임
- csv 대신 parquet으로 저장(용량 훨씬 작고 dtype 보존, 이 프로젝트의 기존
  convert_to_parquet.py와 동일한 포맷 컨벤션)
- 결과 파일: data/final/cleaned_main_joined_v1.parquet
  (data/ 전체가 .gitignore 대상이라 git에는 안 올라감 -> 별도로 파일
  전달해야 함, 예: 구글드라이브/USB 등)
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FINAL_DIR = os.path.join(BASE_DIR, "data", "final")
OUT_FILE = os.path.join(FINAL_DIR, "cleaned_main_joined_v1.parquet")

CHUNK_SIZE = 200_000


def get_engine():
    load_dotenv(os.path.join(BASE_DIR, ".env"))

    host = os.getenv("MYSQL_HOST")
    port = os.getenv("MYSQL_PORT", "3306")
    user = os.getenv("MYSQL_USER")
    password = os.getenv("MYSQL_PASSWORD")
    database = os.getenv("MYSQL_DATABASE")

    missing = [k for k, v in {
        "MYSQL_HOST": host,
        "MYSQL_USER": user,
        "MYSQL_PASSWORD": password,
        "MYSQL_DATABASE": database,
    }.items() if not v]

    if missing:
        raise ValueError(f".env에 다음 값이 비어 있습니다: {', '.join(missing)}")

    url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}?charset=utf8mb4"
    return create_engine(url)


def main():
    os.makedirs(FINAL_DIR, exist_ok=True)
    engine = get_engine()

    with engine.connect() as conn:
        total_rows = conn.execute(
            "SELECT COUNT(*) FROM cleaned_main_joined_v1"
        ).scalar()
    print(f"cleaned_main_joined_v1 총 {total_rows:,}행 내보내기 시작 (청크 {CHUNK_SIZE:,}행씩)")

    chunks = []
    read = 0
    for chunk in pd.read_sql(
        "SELECT * FROM cleaned_main_joined_v1",
        con=engine,
        chunksize=CHUNK_SIZE,
    ):
        chunks.append(chunk)
        read += len(chunk)
        print(f"  읽음: {read:,} / {total_rows:,}")

    df = pd.concat(chunks, ignore_index=True)
    print(f"\n최종 shape: {df.shape}")

    df.to_parquet(OUT_FILE, index=False, engine="pyarrow")
    size_mb = os.path.getsize(OUT_FILE) / (1024 * 1024)
    print(f"저장 완료: {OUT_FILE} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
