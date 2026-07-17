"""
check_aws_fallback.py

설명: asos_by_region_updated.csv에서 AWS 덮어쓰기가 적용되지 않은 행
    (AWS 원본 결측으로 인해 기존 인접 지역값이 남아있는 행)을 출력합니다.
"""

import os
import pandas as pd

THIS_FILE   = os.path.abspath(__file__)
PROJ_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(THIS_FILE)))
WEATHER_DIR = os.path.join(PROJ_ROOT, "data", "external", "weather")

AWS_PATH    = os.path.join(WEATHER_DIR, "aws_by_region.xls")
UPDATED_PATH = os.path.join(WEATHER_DIR, "asos_by_region_updated.csv")

STN_MAP = {
    "삼척": "강원도 삼척시",
    "김포": "경기도 김포시",
    "성남": "경기도 성남시 분당구",
    "평택": "경기도 평택시",
    "고성": "경상남도 고성군",
    "사천": "경상남도 사천시",
    "창녕": "경상남도 창녕군",
    "하동": "경상남도 하동군",
    "함안": "경상남도 함안군",
    "경산": "경상북도 경산시",
    "익산": "전라북도 익산시",
    "논산": "충청남도 논산시",
    "음성": "충청북도 음성군",
}


def main():
    # AWS에서 결측 날짜 수집: {region: {date: {col: True/False}}}
    aws = pd.read_csv(AWS_PATH, sep="\t", encoding="euc-kr")
    aws.columns = ["stn_id", "stn_name", "date", "avg_ta", "sum_rn"]
    aws["date"]   = pd.to_datetime(aws["date"])
    aws["avg_ta"] = pd.to_numeric(aws["avg_ta"], errors="coerce")
    aws["sum_rn"] = pd.to_numeric(aws["sum_rn"], errors="coerce")

    # 결측 (date, region) 쌍 수집
    nan_index: dict[str, set] = {}   # region → set of NaN dates
    for stn, region in STN_MAP.items():
        sub = aws[aws["stn_name"] == stn]
        nan_dates = set(sub[sub["avg_ta"].isna() | sub["sum_rn"].isna()]["date"])
        if nan_dates:
            nan_index[region] = nan_dates

    # 업데이트 파일에서 해당 행 추출
    asos = pd.read_csv(UPDATED_PATH, encoding="utf-8-sig")
    asos["date"] = pd.to_datetime(asos["date"])

    fallback_rows = []
    for region, nan_dates in nan_index.items():
        mask = (asos["region"] == region) & (asos["date"].isin(nan_dates))
        fallback_rows.append(asos[mask])

    result = pd.concat(fallback_rows).sort_values(["region", "date"]).reset_index(drop=True)

    # 출력
    pd.set_option("display.max_rows", None)
    pd.set_option("display.width", 120)
    pd.set_option("display.float_format", "{:.1f}".format)

    print(f"결측 잔존 행 수: {len(result)}행  /  지역 수: {result['region'].nunique()}개\n")

    for region, group in result.groupby("region", sort=False):
        print(f"{'─' * 70}")
        print(f"  {region}  ({len(group)}행)")
        print(f"{'─' * 70}")
        print(group[["date", "stn_name", "avg_ta", "sum_rn"]].to_string(index=False))
        print()


if __name__ == "__main__":
    main()
