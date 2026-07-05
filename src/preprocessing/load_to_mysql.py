"""
load_to_mysql.py

설명: data/csv/ 폴더의 CSV 6개를 MySQL에 각각 별도 테이블로 적재합니다.
      - .env 파일에서 MySQL 접속 정보를 읽어옵니다.
      - if_exists="fail" 설정으로 기존 테이블과 이름이 겹치면 에러를 내고 중단합니다.
      - 적재 후 테이블 row count와 원본 CSV row 수를 비교해 검증합니다.

실행 전 체크리스트:
  1. .env 파일에 MySQL 접속 정보를 모두 입력했는지 확인
  2. check_files.py를 먼저 실행해서 데이터 상태 확인
  3. MySQL DB에 이미 같은 이름의 테이블이 없는지 확인
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CSV_DIR = os.path.join(BASE_DIR, "data", "csv")

# 파일명 → 테이블명 매핑
CSV_FILES = {
    "a_center_purchase_2021_2024_mapped.csv":  "a_purchase",
    "a_center_sales_2021_2023_mapped.csv":     "a_sales_2021_2023",
    "a_center_sales_2024_mapped.csv":          "a_sales_2024",
    "b_center_purchase_2021_2024_mapped.csv":  "b_purchase",
    "b_center_sales_2021_2023_mapped.csv":     "b_sales_2021_2023",
    "b_center_sales_2024_mapped.csv":          "b_sales_2024",
}


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


def load_csv_to_table(engine, file_name: str, table_name: str) -> tuple[int, int]:
    path = os.path.join(CSV_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    print(f"\n[{table_name}] 적재 시작 ← {file_name}")
    df = pd.read_csv(path, encoding="utf-8-sig", low_memory=False)
    csv_rows = len(df)
    print(f"  CSV rows: {csv_rows:,}")

    # if_exists="fail" → 테이블이 이미 존재하면 ValueError 발생 (안전장치)
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="fail",
        index=False,
        chunksize=5000,
        method="multi",
    )

    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM `{table_name}`"))
        db_rows = result.scalar()

    print(f"  DB rows : {db_rows:,}")
    return csv_rows, db_rows


def verify(table_name: str, csv_rows: int, db_rows: int):
    status = "OK" if csv_rows == db_rows else "MISMATCH"
    print(f"  검증    : [{status}] CSV {csv_rows:,} vs DB {db_rows:,}")
    if status == "MISMATCH":
        raise RuntimeError(
            f"[{table_name}] row 수 불일치 — CSV: {csv_rows:,}, DB: {db_rows:,}"
        )


def main():
    print("MySQL 접속 중...")
    engine = get_engine()
    print("접속 성공.\n")

    results = {}
    for file_name, table_name in CSV_FILES.items():
        csv_rows, db_rows = load_csv_to_table(engine, file_name, table_name)
        verify(table_name, csv_rows, db_rows)
        results[table_name] = {"csv": csv_rows, "db": db_rows}

    print(f"\n{'=' * 55}")
    print(f"{'테이블명':<30} {'CSV rows':>10} {'DB rows':>10} {'상태':>5}")
    print("-" * 55)
    for tbl, cnt in results.items():
        status = "OK" if cnt["csv"] == cnt["db"] else "MISMATCH"
        print(f"{tbl:<30} {cnt['csv']:>10,} {cnt['db']:>10,} {status:>5}")
    print("=" * 55)
    print("전체 적재 완료.")


if __name__ == "__main__":
    main()
