"""
기상청 ASOS 일자료 수집 스크립트 (외부 데이터)
- 수집 항목: avgTa (평균기온), sumRn (일강수량)
- 기간: 2021-01-01 ~ 2024-12-31
- 저장 위치: CAPSTONE-PROJ/data/external/weather/
- 실행 위치: CAPSTONE-PROJ/src/preprocessing/
- 지점 선택 원칙: 해당 시군구 직접 지점 우선, 없으면 같은 도 내 인접 지점

[매핑 요약]
직접 지점 (26개 지역):
    이천, 거제, 김해시, 남해, 밀양, 산청, 양산시, 의령군, 진주, 창원,
    통영, 합천, 경주시, 구미, 문경, 영덕, 영천, 울진, 포항,
    광주(동구), 대구, 대전(동구), 부산, 서울, 울산, 순천, 제주, 천안, 제천

인접 지점 - 같은 도 내 (16개 지역):
    강원도 삼척시    → 동해(106)  : 삼척 전용 없음, 강원도 내 최근접
    경기도 김포시    → 파주(99)   : 경기 북서부 동일 기후대 (인천은 경기도 아님)
    경기도 성남시    → 수원(119)  : 경기 남부 대표 지점
    경기도 평택시    → 수원(119)  : 경기 서남부 동일 기후대
    경상남도 고성군  → 통영(162)  : 경남 내 접경 최근접
    경상남도 사천시  → 진주(192)  : 경남 내 접경
    경상남도 창녕군  → 밀양(288)  : 경남 내 낙동강 중류 동일 기후대
    경상남도 하동군  → 진주(192)  : 경남 내 최근접
    경상남도 함안군  → 창원(155)  : 경남 내 낙동강 하류 동일 기후대
    경상북도 경산시  → 대구(143)  : 경북 내 대구 도시권 (대구는 경북 접경 광역시)
    ※ 경북 내 지점 중 경산과 가장 가까운 대구 인접 지점 없어 대구 사용
    전라북도 익산시  → 전주(146)  : 전북 내 호남평야 동일 기후대
    충청남도 논산시  → 부여(236)  : 충남 내 최근접 (대전은 충남 아님)
    충청북도 음성군  → 충주(127)  : 충북 내 최근접
"""

import os
import time
import requests
import pandas as pd


THIS_FILE  = os.path.abspath(__file__)
SRC_DIR    = os.path.dirname(THIS_FILE)          # src/preprocessing/
PROJ_ROOT  = os.path.dirname(os.path.dirname(SRC_DIR))  # CAPSTONE-PROJ/
SAVE_DIR   = os.path.join(PROJ_ROOT, "data", "external", "weather")
os.makedirs(SAVE_DIR, exist_ok=True)


WEATHER_API_KEY = [l.split("=",1)[1].strip().strip('"').strip("'") for l in open(os.path.join(os.path.dirname(__file__),"api_keys.env"), encoding="utf-8") if l.strip().startswith("WEATHER_API_KEY")][0]
BASE_URL = "http://apis.data.go.kr/1360000/AsosDalyInfoService/getWthrDataList"
YEARS    = list(range(2021, 2025))   # 2021 ~ 2024


REGION_TO_STN = {
    # ── 강원도 ──────────────────────────────────────────────
    "강원도 삼척시":            ("동해",    106),  # 인접 → 동해

    # ── 경기도 ──────────────────────────────────────────────
    "경기도 김포시":            ("파주",     99),  # 인접 → 파주
    "경기도 성남시 분당구":     ("수원",    119),  # 인접 → 수원
    "경기도 이천시":            ("이천",    203),  # 직접
    "경기도 평택시":            ("수원",    119),  # 인접 → 수원

    # ── 경상남도 ────────────────────────────────────────────
    "경상남도 거제시":          ("거제",    294),  # 직접
    "경상남도 고성군":          ("통영",    162),  # 인접 → 통영
    "경상남도 김해시":          ("김해시",  253),  # 직접
    "경상남도 남해군":          ("남해",    295),  # 직접
    "경상남도 밀양시":          ("밀양",    288),  # 직접
    "경상남도 사천시":          ("진주",    192),  # 인접 → 진주
    "경상남도 산청군":          ("산청",    289),  # 직접
    "경상남도 양산시":          ("양산시",  257),  # 직접
    "경상남도 의령군":          ("의령군",  263),  # 직접
    "경상남도 진주시":          ("진주",    192),  # 직접
    "경상남도 창녕군":          ("밀양",    288),  # 인접 → 밀양
    "경상남도 창원시":          ("창원",    155),  # 직접
    "경상남도 통영시":          ("통영",    162),  # 직접
    "경상남도 하동군":          ("진주",    192),  # 인접 → 진주
    "경상남도 함안군":          ("창원",    155),  # 인접 → 창원
    "경상남도 합천군":          ("합천",    285),  # 직접

    # ── 경상북도 ────────────────────────────────────────────
    "경상북도 경산시":          ("대구",    143),  # 인접 → 대구
    "경상북도 경주시":          ("경주시",  283),  # 직접
    "경상북도 구미시":          ("구미",    279),  # 직접
    "경상북도 문경시":          ("문경",    273),  # 직접
    "경상북도 영덕군":          ("영덕",    277),  # 직접
    "경상북도 영천시":          ("영천",    281),  # 직접
    "경상북도 울진군":          ("울진",    130),  # 직접
    "경상북도 포항시":          ("포항",    138),  # 직접

    # ── 광역시 / 특별시 / 특별자치도 ────────────────────────
    "광주광역시 동구":          ("광주",    156),  # 직접
    "대구광역시":               ("대구",    143),  # 직접
    "대전광역시 동구":          ("대전",    133),  # 직접
    "부산광역시":               ("부산",    159),  # 직접
    "서울특별시":               ("서울",    108),  # 직접
    "울산광역시":               ("울산",    152),  # 직접
    "제주특별자치도 제주시":    ("제주",    184),  # 직접

    # ── 전라도 ──────────────────────────────────────────────
    "전라남도 순천시":          ("순천",    174),  # 직접
    "전라북도 익산시":          ("전주",    146),  # 인접 → 전주

    # ── 충청도 ──────────────────────────────────────────────
    "충청남도 논산시":          ("부여",    236),  # 인접 → 부여
    "충청남도 천안시":          ("천안",    232),  # 직접
    "충청북도 음성군":          ("충주",    127),  # 인접 → 충주
    "충청북도 제천시":          ("제천",    221),  # 직접
}


UNIQUE_STNS: dict[int, str] = {}
for _region, (_stn_name, _stn_id) in REGION_TO_STN.items():
    if _stn_id not in UNIQUE_STNS:
        UNIQUE_STNS[_stn_id] = _stn_name


def fetch_asos_year(stn_id: int, year: int) -> list[dict]:
    """특정 지점의 1개년 일자료 조회 (최대 366행)"""
    params = {
        "serviceKey": WEATHER_API_KEY,
        "numOfRows":  400,
        "pageNo":     1,
        "dataType":   "JSON",
        "dataCd":     "ASOS",
        "dateCd":     "DAY",
        "startDt":    f"{year}0101",
        "endDt":      f"{year}1231",
        "stnIds":     stn_id,
    }
    try:
        resp = requests.get(BASE_URL, params=params, timeout=10)
        resp.raise_for_status()
        items = (
            resp.json()
                .get("response", {})
                .get("body", {})
                .get("items", {})
                .get("item", [])
        )
        return items if isinstance(items, list) else [items]
    except Exception as e:
        print(f"  [ERROR] stn={stn_id}, year={year}: {e}")
        return []


def collect_all() -> pd.DataFrame:
    records = []
    total = len(UNIQUE_STNS) * len(YEARS)
    done  = 0

    for stn_id, stn_name in UNIQUE_STNS.items():
        for year in YEARS:
            done += 1
            print(f"[{done:3d}/{total}] {stn_name}({stn_id}) {year}년 수집 중...")
            for item in fetch_asos_year(stn_id, year):
                records.append({
                    "stn_id":   item.get("stnId"),
                    "stn_name": item.get("stnNm"),
                    "date":     item.get("tm"),      # YYYY-MM-DD
                    "avg_ta":   item.get("avgTa"),   # 평균기온(°C)
                    "sum_rn":   item.get("sumRn"),   # 일강수량(mm)
                })
            time.sleep(0.3)   # rate limit 대응

    df = pd.DataFrame(records)
    df["date"]   = pd.to_datetime(df["date"])
    df["avg_ta"] = pd.to_numeric(df["avg_ta"], errors="coerce")
    df["sum_rn"] = pd.to_numeric(df["sum_rn"], errors="coerce")
    return df


def add_region_column(df: pd.DataFrame) -> pd.DataFrame:
    stn_to_regions: dict[int, list[str]] = {}
    for region, (_, stn_id) in REGION_TO_STN.items():
        stn_to_regions.setdefault(stn_id, []).append(region)

    rows = []
    for row in df.itertuples(index=False):
        for region in stn_to_regions.get(int(row.stn_id), [str(row.stn_id)]):
            rows.append({
                "region":   region,
                "stn_id":   row.stn_id,
                "stn_name": row.stn_name,
                "date":     row.date,
                "avg_ta":   row.avg_ta,
                "sum_rn":   row.sum_rn,
            })
    return pd.DataFrame(rows)



if __name__ == "__main__":
    print("=" * 60)
    print("  ASOS 일자료 수집 시작 (2021~2024)")
    print(f"  대상 지점 : {len(UNIQUE_STNS)}개")
    print(f"  저장 경로 : {SAVE_DIR}")
    print("=" * 60)

    df_raw = collect_all()
    print(f"\n수집 완료 : {len(df_raw):,}건")

    df = add_region_column(df_raw)
    print(f"지역 확장  : {len(df):,}건 ({df['region'].nunique()}개 지역)\n")

    # 결측치 처리
    df["sum_rn"] = df["sum_rn"].fillna(0)  # 강수 없는 날 → 0
    df["avg_ta"] = df.groupby("region")["avg_ta"].transform(
        lambda x: x.interpolate(method="linear")  # 기온 누락 → 앞뒤 평균 보간
    )
    print("결측치 처리 완료")
    print(df[["avg_ta", "sum_rn"]].isnull().sum().to_string())


    path_raw    = os.path.join(SAVE_DIR, "asos_raw.parquet")
    path_region = os.path.join(SAVE_DIR, "asos_by_region.parquet")
    path_csv    = os.path.join(SAVE_DIR, "asos_by_region.csv")

    df_raw.to_parquet(path_raw,    index=False)
    df.to_parquet(path_region,     index=False)
    df.to_csv(path_csv, index=False, encoding="utf-8-sig")

    print("저장 완료:")
    print(f"  {path_raw}")
    print(f"  {path_region}")
    print(f"  {path_csv}")

    print("\n--- 미리보기 (5행) ---")
    print(df[["region", "stn_name", "date", "avg_ta", "sum_rn"]].head().to_string(index=False))

    print("\n--- 결측치 현황 ---")
    print(df[["avg_ta", "sum_rn"]].isnull().sum().to_string())