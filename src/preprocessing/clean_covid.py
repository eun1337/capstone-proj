# clean_covid.py
# 코로나19(COVID-19) 영향 여부를 일자별로 나타내는 데이터를 전처리하는 스크립트.
# date 컬럼을 연/월/일로 분리하고, covid_impact 컬럼명을 영향여부로 변경한 뒤
# covid_impact_daily_clean.csv로 저장한다. (값 자체는 원본 그대로 사용)

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
INPUT_PATH = DATA_DIR / "external" / "covid" / "covid_impact_daily.csv"
OUTPUT_PATH = DATA_DIR / "external" / "covid" / "covid_impact_daily_clean.csv"


def clean_covid_data(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    df["date"] = pd.to_datetime(df["date"])

    df["년"] = df["date"].dt.year
    df["월"] = df["date"].dt.month
    df["일"] = df["date"].dt.day

    df = df.rename(columns={"covid_impact": "영향여부"})

    result = df[["년", "월", "일", "영향여부"]]

    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    return result


if __name__ == "__main__":
    clean_covid_data(INPUT_PATH, OUTPUT_PATH)
    print(f"저장 완료: {OUTPUT_PATH}")