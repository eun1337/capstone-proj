"""
clean_economic_data.py
------------------------------------------------------------------
소비자물가지수.xlsx, 소비자심리지수.xlsx 정제 스크립트.
- 두 파일 모두 "데이터" 시트에 전국 단위 값 1행만 존재 (시도별 구분 없음)
- wide format(연월이 컬럼)을 melt로 long format 변환
- 최종 컬럼: 년,월,소비자물가지수 / 년,월,소비자심리지수
"""

from pathlib import Path

import pandas as pd

ECONOMIC_DIR = Path(__file__).resolve().parents[2] / "data" / "external" / "economic"


def wide_to_national_long(df, value_name):
    df_long = pd.melt(df, var_name="연월", value_name=value_name)
    df_long["연월"] = pd.to_datetime(df_long["연월"], format="%Y.%m")
    df_long["년"] = df_long["연월"].dt.year
    df_long["월"] = df_long["연월"].dt.month
    return df_long[["년", "월", value_name]].sort_values(["년", "월"]).reset_index(drop=True)


cpi_raw = pd.read_excel(ECONOMIC_DIR / "소비자물가지수.xlsx", sheet_name="데이터")
cpi_raw = cpi_raw[cpi_raw["시도별"] == "전국"].drop(columns=["시도별"])
cpi = wide_to_national_long(cpi_raw, "소비자물가지수")
cpi.to_csv(ECONOMIC_DIR / "consumer_price_index_national.csv", index=False, encoding="utf-8-sig")

csi_raw = pd.read_excel(ECONOMIC_DIR / "소비자심리지수.xlsx", sheet_name="데이터")
csi_raw = csi_raw[csi_raw["CSI분류코드별"] == "전체"].drop(columns=["CSI코드별", "CSI분류코드별"])
csi = wide_to_national_long(csi_raw, "소비자심리지수")
csi.to_csv(ECONOMIC_DIR / "consumer_sentiment_index_national.csv", index=False, encoding="utf-8-sig")
