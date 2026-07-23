"""
01_center_sku_csv_conversion.py

data/final/aggregated_daily_demand.parquet -> 센터+SKU(바코드+옵션코드+상품클러스터)
레벨로 시도/시군구를 합산한 일별 CSV로 변환.

- 지역(시도/시군구)은 합산하여 group-by에서 제외 (센터+SKU 레벨 분석 목적)
- 수량/금액/건수 계열은 sum, 공휴일/covid/기온/강수량/cpi 등 날짜 단위 값은 first
  (같은 날짜는 원래 동일 값이므로 first로 손실 없음)

출력: data/csv/daily_center_sku.csv (다음 단계인 02_full_daily_eda_by_center.py의 입력이 됨)
"""
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"

SRC = DATA_DIR / "final" / "aggregated_daily_demand.parquet"
DST_DIR = DATA_DIR / "csv"
DST_DIR.mkdir(parents=True, exist_ok=True)
DST = DST_DIR / "daily_center_sku.csv"

df = pd.read_parquet(SRC)

group_cols = ["거래일", "센터", "바코드", "옵션코드", "상품클러스터"]

sum_cols = [
    "총판매수량", "총판매금액",
    "반품수량", "반품금액",
    "순수량", "순금액",
    "총거래건수", "판매건수", "반품건수",
]

first_cols = [
    "상품명", "규격", "입수", "KAN_CODE", "KAN_대분류", "KAN_중분류", "KAN_소분류",
    "공휴일", "covid_영향여부", "평균온도", "총강수량", "cpi", "경상지수", "불변지수",
]

sum_cols = [c for c in sum_cols if c in df.columns]
first_cols = [c for c in first_cols if c in df.columns]

agg_dict = {c: "sum" for c in sum_cols}
agg_dict.update({c: "first" for c in first_cols})

out = df.groupby(group_cols, as_index=False).agg(agg_dict)

out.to_csv(DST, index=False, encoding="utf-8-sig")
print(f"원본 행 수: {len(df):,}")
print(f"변환 후 행 수 (센터+SKU 레벨): {len(out):,}")
print(f"저장 완료: {DST}")
