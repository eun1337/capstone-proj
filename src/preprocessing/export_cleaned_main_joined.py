"""
export_cleaned_main_joined.py
------------------------------------------------------------------
- 대용량(356만행)이라 pandas.read_sql을 청크 단위로 읽어 메모리 부담을 줄임
- csv 대신 parquet으로 저장(용량 훨씬 작고 dtype 보존, 이 프로젝트의 기존 convert_to_parquet.py와 동일한 포맷 컨벤션)
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FINAL_DIR = os.path.join(BASE_DIR, "data", "final")
OUT_FILE = os.path.join(FINAL_DIR, "cleaned_main_joined.parquet")

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
            text("SELECT COUNT(*) FROM cleaned_main_joined")
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

    # 청크마다 pandas가 독립적으로 dtype을 추론해서, concat 후에도 컬럼
    # 하나에 타입이 섞여 남을 수 있어 명시적으로 고정
    df["공휴일"] = df["공휴일"].astype(int)

    print(f"\n최종 shape: {df.shape}")

    df.to_parquet(OUT_FILE, index=False, engine="pyarrow")
    size_mb = os.path.getsize(OUT_FILE) / (1024 * 1024)
    print(f"저장 완료: {OUT_FILE} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
