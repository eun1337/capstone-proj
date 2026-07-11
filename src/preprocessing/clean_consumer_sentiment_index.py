"""
clean_consumer_sentiment_index.py
------------------------------------------------------------------
consumer_sentiment_index.xlsx 정제 스크립트.
- CSI코드별 컬럼은 전부 "소비자심리지수" 하나뿐이라 버림
- wide format(날짜가 컬럼)을 melt로 long format 변환
- 담당부서별이 "대구경북","광주전남","대전세종충남"처럼 2~3개 시도가
  하나로 묶인 권역 단위라서, 개별 시도로 값을 복제(broadcast)함
  (권역 값을 소속 시도들에 동일하게 적용 - 최선의 근사치이며 실제로는
  시도별 세부 값이 다를 수 있음을 유의)
- 단, "강원"과 "강릉"은 둘 다 강원특별자치도 소속이지만 실제로는 서로 다른
  관할 시군구를 담당하는 별도 부서이므로(강원=춘천 본부권, 강릉=강릉 본부권),
  시도 레벨이 아니라 시군구 레벨까지 구분해서 broadcast함:
    - 강원(춘천권): 춘천시, 원주시 / 홍천군, 횡성군, 철원군, 화천군, 양구군, 인제군
    - 강릉(강릉권): 강릉시, 동해시, 태백시, 속초시, 삼척시 /
                    영월군, 평창군, 정선군, 고성군, 양양군
  나머지 권역(부산, 경기 등)은 시군구 세부 구분 없이 시도 단위로만 broadcast하며
  시군구는 결측(NaN)으로 둠.
- 최종 컬럼: 시도,시군구,년,월,소비자심리지수
  (연월 -> 년/월 분리, 시도별->시도, CSI->소비자심리지수)
- 결과를 같은 폴더에 consumer_sentiment_index_clean.csv 로 저장
"""

from pathlib import Path

import pandas as pd

BASE_DIR     = Path(__file__).resolve().parents[2]           
DATA_DIR     = BASE_DIR / "data"
ECONOMIC_DIR = DATA_DIR / "external" / "economic"
SRC_FILE     = ECONOMIC_DIR / "consumer_sentiment_index.xlsx"
OUT_FILE     = ECONOMIC_DIR / "consumer_sentiment_index_clean.csv"

# 강원(춘천권) 관할 시군구: 2개시 + 6개군
GANGWON_CHUNCHEON = [
    "춘천시", "원주시",
    "홍천군", "횡성군", "철원군", "화천군", "양구군", "인제군",
]

# 강릉권 관할 시군구: 5개시 + 5개군
GANGWON_GANGNEUNG = [
    "강릉시", "동해시", "태백시", "속초시", "삼척시",
    "영월군", "평창군", "정선군", "고성군", "양양군",
]

REGION_TO_LOCATIONS = {
    "부산": [("부산광역시", None)],
    "대구경북": [("대구광역시", None), ("경상북도", None)],
    "인천": [("인천광역시", None)],
    "광주전남": [("광주광역시", None), ("전라남도", None)],
    "대전세종충남": [("대전광역시", None), ("세종특별자치시", None), ("충청남도", None)],
    "울산": [("울산광역시", None)],
    "경기": [("경기도", None)],
    "강원": [("강원특별자치도", gu) for gu in GANGWON_CHUNCHEON],
    "충북": [("충청북도", None)],
    "전북": [("전북특별자치도", None)],
    "경남": [("경상남도", None)],
    "제주": [("제주특별자치도", None)],
    "강릉": [("강원특별자치도", gu) for gu in GANGWON_GANGNEUNG],
}

df = pd.read_excel(SRC_FILE)
print(f"원본 shape: {df.shape}")
print("담당부서별 고유값:", df["담당부서별"].unique().tolist())

df = df.drop(columns=["CSI코드별"])

df_long = pd.melt(df, id_vars=["담당부서별"], var_name="연월", value_name="CSI")
df_long["연월"] = pd.to_datetime(df_long["연월"], format="%Y.%m")


unmapped = set(df_long["담당부서별"].unique()) - set(REGION_TO_LOCATIONS.keys())
if unmapped:
    print(f"\n[경고] REGION_TO_LOCATIONS에 없는 담당부서별 값 발견: {unmapped}")
    print("-> 위 딕셔너리에 추가해주세요. 일단 이 값들은 매핑 안 된 채로 둡니다.")


rows = []
for _, row in df_long.iterrows():
    locations = REGION_TO_LOCATIONS.get(row["담당부서별"], [(row["담당부서별"], None)])
    for sido, sigungu in locations:
        rows.append({"시도별": sido, "시군구": sigungu, "연월": row["연월"], "CSI": row["CSI"]})

df_broadcast = pd.DataFrame(rows)


n_before = len(df_broadcast)
df_broadcast = df_broadcast.groupby(
    ["시도별", "시군구", "연월"], as_index=False, dropna=False
)["CSI"].mean()
n_after = len(df_broadcast)
if n_before != n_after:
    print(f"\n[안내] 중복된 시도+시군구+연월 {n_before - n_after}건을 평균으로 병합함")


df_broadcast = df_broadcast.rename(columns={"시도별": "시도", "CSI": "소비자심리지수"})
df_broadcast["년"] = df_broadcast["연월"].dt.year
df_broadcast["월"] = df_broadcast["연월"].dt.month
df_broadcast = df_broadcast.drop(columns=["연월"])
df_broadcast = df_broadcast[["시도", "시군구", "년", "월", "소비자심리지수"]]

print(f"\n최종 shape: {df_broadcast.shape}")
print(df_broadcast.head(10))
print()
print("시도 고유값:", sorted(df_broadcast["시도"].unique()))
print(
    "강원특별자치도 세부 시군구:",
    sorted(df_broadcast.loc[df_broadcast["시도"] == "강원특별자치도", "시군구"].dropna().unique()),
)
print(
    "중복행(시도+시군구+년+월 기준):",
    df_broadcast.duplicated(subset=["시도", "시군구", "년", "월"]).sum(),
    "건",
)

df_broadcast.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
print(f"저장 완료: {OUT_FILE}")