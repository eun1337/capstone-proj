"""
clean_weather_by_region.py
------------------------------------------------------------------
weather_by_region.csv (overwrite_asos_with_aws.py로 생성된, ASOS 원본 중
인접 지점 대체 지역을 AWS 실관측값으로 보정한 최종 날씨 데이터)
정제 스크립트.
- stn_id를 문자열로 캐스팅 (관측소 코드는 산술 연산 대상이 아니므로 str 고정,
  다른 파일과 dtype 불일치 방지)
- date를 datetime으로 변환 후 년/월/일 컬럼으로 분리
- 'region' 컬럼("강원도 삼척시"처럼 시도+시군구가 하나의 문자열로 합쳐진 형태)을
  첫 번째 공백 기준으로 시도/시군구로 분리 (시도는 항상 공백 없는 단일 단어이고,
  시군구는 "성남시 분당구"처럼 내부에 공백이 있을 수 있으므로
  split(' ', maxsplit=1) 사용. 공백이 없는 값(예: "서울특별시"만 있는 행)은
  시군구를 NaN으로 둠 - 관측소가 광역시 전체를 대표하는 경우로 보임)
- 시도명은 2024년 이후 공식 행정구역명으로 표준화
  (강원도->강원특별자치도, 전라북도->전북특별자치도, 제주도->제주특별자치도)
- stn_id, stn_name 컬럼 제거 (최종 데이터에서 불필요)
- avg_ta -> 평균온도, sum_rn -> 총강수량 으로 컬럼명 변경
- 최종 컬럼: 시도,시군구,년,월,일,평균온도,총강수량
- 결과를 같은 폴더에 weather_by_region_clean.csv 로 저장

[커버리지 관련 주의사항]
REGION_TO_STN(collect_weather_asos.py)은 현재 매장이 존재하는 15개 시도만 커버하며, 
인천광역시·세종특별자치시 관측지점은 매핑되어 있지 않음 
(regional/postal_code_region_mapping.xlsx의 커버리지와 정확히 일치 - clean_postal_code_region_mapping.py 주석 참고). 
다른 물류 데이터로 파이프라인을 확장해 인천/세종 지역 매장을 다뤄야 한다면 REGION_TO_STN에 해당 지역 인접 관측지점을 추가하고 재수집해야 함.
"""

from pathlib import Path

import pandas as pd

BASE_DIR    = Path(__file__).resolve().parents[2]
DATA_DIR    = BASE_DIR / "data"
WEATHER_DIR = DATA_DIR / "external" / "weather"
SRC_FILE    = WEATHER_DIR / "weather_by_region.csv"
OUT_FILE    = WEATHER_DIR / "weather_by_region_clean.csv"


SIDO_STANDARD = {
    "강원도": "강원특별자치도",
    "전라북도": "전북특별자치도",
    "제주도": "제주특별자치도",
    "제주": "제주특별자치도",
}

df = pd.read_csv(SRC_FILE, dtype={"stn_id": str}, parse_dates=["date"], encoding="cp949")
print(f"원본 shape: {df.shape}")
print("원본 컬럼:", df.columns.tolist())


split = df["region"].str.split(" ", n=1, expand=True)
df["시도"] = split[0]
df["시군구"] = split[1] if split.shape[1] > 1 else None


before_unique = set(df["시도"].unique())
df["시도"] = df["시도"].replace(SIDO_STANDARD)
after_unique = set(df["시도"].unique())
if before_unique - after_unique:
    print(f"\n표준화로 바뀐 시도명: {before_unique - after_unique}")


df["년"] = df["date"].dt.year
df["월"] = df["date"].dt.month
df["일"] = df["date"].dt.day


df = df.drop(columns=["region", "stn_id", "stn_name", "date"])
df = df.rename(columns={"avg_ta": "평균온도", "sum_rn": "총강수량"})
df = df[["시도", "시군구", "년", "월", "일", "평균온도", "총강수량"]]

print(f"\n최종 shape: {df.shape}")
print(df.head())
print()
print("시도 고유값:", sorted(df["시도"].unique()))
print("시군구 결측(공백 없던 행) 수:", df["시군구"].isna().sum(), "건")
print("결측인 행의 시도 예시:", sorted(df[df["시군구"].isna()]["시도"].unique()))
print("중복행:", df.duplicated().sum(), "건")

df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
print(f"저장 완료: {OUT_FILE}")