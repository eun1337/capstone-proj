"""
load_external_to_mysql.py

data/external/ 하위 정제된 외부 데이터 7개를 MySQL에 각각 별도 테이블로 적재

    - dtype을 컬럼별로 명시 지정합니다.
    - if_exists="replace"를 씁니다. 메인 데이터는 한 번 적재된 뒤 고정이므로 "fail"로 중복 적재를 막는 게 맞지만, 외부 데이터는 clean_*.py를 돌릴 때마다(코로나 데이터처럼) 값이 바뀔 수 있어 매번 새로 갱신되어야 하므로 replace가 실용적
"""

import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXTERNAL_DIR = os.path.join(BASE_DIR, "data", "external")

# 파일 경로(EXTERNAL_DIR 기준 상대경로) -> (테이블명, dtype, 적재 후 실행할 ALTER문 목록)
CSV_FILES = {
    "weather/weather_by_region_clean.csv": (
        "weather_daily",
        {"시도": str, "시군구": str},
        [
            "ALTER TABLE weather_daily MODIFY 시도 VARCHAR(20) NOT NULL",
            "ALTER TABLE weather_daily MODIFY 시군구 VARCHAR(20) NULL",
            "ALTER TABLE weather_daily ADD KEY idx_region_date (시도, 시군구, 년, 월, 일)",
            # sql/join/01_master_join.sql의 시군구 OR 매칭(정확매칭/광역시 broadcast/
            # 시(구) 세분화 fallback)은 시군구가 인덱스 2번째 컬럼이면 못 타서
            # 메인 데이터(수백만 행) x weather_daily 풀스캔이 발생함 — 날짜/시도처럼
            # 항상 등호로 매칭되는 컬럼을 앞에 둔 인덱스를 별도로 추가해 후보를
            # 먼저 소수로 좁힌 뒤 OR 조건을 걸러내도록 함.
            "ALTER TABLE weather_daily ADD INDEX idx_date_first (시도, 년, 월, 일, 시군구)",
        ],
    ),
    "regional/postal_code_region_mapping_clean.csv": (
        "postal_code_region",
        {"우편번호": str, "시도": str, "시군구": str},
        [
            "ALTER TABLE postal_code_region MODIFY 우편번호 CHAR(5) NOT NULL",
            "ALTER TABLE postal_code_region ADD PRIMARY KEY (우편번호)",
        ],
    ),
    "economic/cpi_monthly_national.csv": (
        "cpi_monthly_national",
        {},
        ["ALTER TABLE cpi_monthly_national ADD KEY idx_month (년, 월)"],
    ),
    "economic/retail_sales_index_current_clean.csv": (
        "retail_sales_current",
        {"시도": str},
        [
            "ALTER TABLE retail_sales_current MODIFY 시도 VARCHAR(20) NOT NULL",
            "ALTER TABLE retail_sales_current ADD KEY idx_region_month (시도, 년, 월)",
        ],
    ),
    "economic/retail_sales_index_constant_clean.csv": (
        "retail_sales_constant",
        {"시도": str},
        [
            "ALTER TABLE retail_sales_constant MODIFY 시도 VARCHAR(20) NOT NULL",
            "ALTER TABLE retail_sales_constant ADD KEY idx_region_month (시도, 년, 월)",
        ],
    ),
    "holiday/public_holidays_2021_2024_clean.csv": (
        "public_holidays",
        {"공휴일": str},
        [
            "ALTER TABLE public_holidays MODIFY 공휴일 VARCHAR(10) NOT NULL",
            "ALTER TABLE public_holidays ADD KEY idx_date (년, 월, 일)",
        ],
    ),
    "covid/covid_impact_daily_clean.csv": (
        "covid_impact_daily",
        {},
        ["ALTER TABLE covid_impact_daily ADD UNIQUE KEY uk_date (년, 월, 일)"],
    ),
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


def load_csv_to_table(engine, rel_path: str, table_name: str, dtype: dict) -> tuple[int, int]:
    path = os.path.join(EXTERNAL_DIR, rel_path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {path}")

    print(f"\n[{table_name}] 적재 시작 ← data/external/{rel_path}")
    df = pd.read_csv(path, encoding="utf-8-sig", dtype=dtype, low_memory=False)
    csv_rows = len(df)
    print(f"  CSV rows: {csv_rows:,}")

    # if_exists="replace" -> 외부 데이터는 재정제될 수 있어 매번 새로 갱신
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="replace",
        index=False,
        chunksize=5000,
        method="multi",
    )

    with engine.connect() as conn:
        result = conn.execute(text(f"SELECT COUNT(*) FROM `{table_name}`"))
        db_rows = result.scalar()

    print(f"  DB rows : {db_rows:,}")
    return csv_rows, db_rows


def apply_alters(engine, table_name: str, alter_statements: list[str]):
    if not alter_statements:
        return
    with engine.begin() as conn:
        for stmt in alter_statements:
            conn.execute(text(stmt))
    print(f"  PK/인덱스 적용 완료: {len(alter_statements)}건")


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
    for rel_path, (table_name, dtype, alters) in CSV_FILES.items():
        csv_rows, db_rows = load_csv_to_table(engine, rel_path, table_name, dtype)
        verify(table_name, csv_rows, db_rows)
        apply_alters(engine, table_name, alters)
        results[table_name] = {"csv": csv_rows, "db": db_rows}

    print(f"\n{'=' * 10}")
    print(f"{'테이블명':<25} {'CSV rows':>10} {'DB rows':>10} {'상태':>6}")
    print("-" * 10)
    for tbl, cnt in results.items():
        status = "OK" if cnt["csv"] == cnt["db"] else "MISMATCH"
        print(f"{tbl:<25} {cnt['csv']:>10,} {cnt['db']:>10,} {status:>6}")
    print("=" * 10)
    print("전체 적재 완료.")


if __name__ == "__main__":
    main()
