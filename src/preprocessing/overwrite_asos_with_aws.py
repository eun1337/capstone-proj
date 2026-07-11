"""
overwrite_asos_with_aws.py

설명: ASOS 기상 파일에서 인접 지역 데이터로 임시 대체된 13개 지역을
      AWS 실관측 데이터(avg_ta, sum_rn)로 덮어씁니다.
      특정 날짜의 AWS 값이 NaN이면 기존 ASOS 값을 유지합니다(fallback).

출력: data/external/weather/asos_by_region_updated.csv (utf-8-sig)

비고: aws_by_region.xls는 실제 탭 구분 텍스트(EUC-KR)이므로
      pd.read_csv(sep='\\t', encoding='euc-kr')로 읽습니다.
"""

import os
import pandas as pd

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
THIS_FILE   = os.path.abspath(__file__)
PROJ_ROOT   = os.path.dirname(os.path.dirname(os.path.dirname(THIS_FILE)))
WEATHER_DIR = os.path.join(PROJ_ROOT, "data", "external", "weather")

ASOS_PATH   = os.path.join(WEATHER_DIR, "asos_by_region.csv")
AWS_PATH    = os.path.join(WEATHER_DIR, "aws_by_region.xls")
OUTPUT_PATH = os.path.join(WEATHER_DIR, "asos_by_region_updated.csv")

# ── AWS 지점명 → ASOS region 매핑 ────────────────────────────────────────────
# AWS 파일: 지점명(한글 약칭) 기준 / ASOS 파일: 행정구역 전체명 기준
AWS_STN_TO_ASOS_REGION: dict[str, str] = {
    "삼척": "강원도 삼척시",
    "김포": "경기도 김포시",
    "성남": "경기도 성남시 분당구",   # ASOS 파일 내 실제 region명
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


def load_asos(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_aws(path: str) -> pd.DataFrame:
    # 실제 파일 형식: 탭 구분 텍스트(EUC-KR) — 확장자만 .xls
    df = pd.read_csv(path, sep="\t", encoding="euc-kr")

    print(f"  AWS 원본 컬럼: {list(df.columns)}")

    # 컬럼명 표준화 (한글 → 내부 표준명)
    df = df.rename(columns={
        "지점":        "stn_id",
        "지점명":      "stn_name",
        "일시":        "date",
        "평균기온(°C)": "avg_ta",
        "일강수량(mm)": "sum_rn",
    })

    df["date"]   = pd.to_datetime(df["date"])
    df["avg_ta"] = pd.to_numeric(df["avg_ta"], errors="coerce")
    df["sum_rn"] = pd.to_numeric(df["sum_rn"], errors="coerce")

    # 지점명 기준으로 ASOS region 컬럼 추가
    df["region"] = df["stn_name"].map(AWS_STN_TO_ASOS_REGION)

    return df


def overwrite_region(
    asos: pd.DataFrame,
    aws:  pd.DataFrame,
    asos_region: str,
) -> tuple[int, int]:
    """
    ASOS에서 asos_region 행의 avg_ta/sum_rn을
    AWS의 해당 지역 데이터로 날짜별로 덮어씁니다.

    AWS 값이 NaN인 날짜는 기존 ASOS 값을 유지합니다(fallback).

    Returns:
        (asos_region_rows, updated_rows)
    """
    # AWS에서 해당 region 데이터를 date 인덱스로 준비
    aws_sub = (
        aws[aws["region"] == asos_region][["date", "avg_ta", "sum_rn"]]
        .drop_duplicates(subset="date")
        .set_index("date")
    )

    if aws_sub.empty:
        return 0, 0

    # ASOS에서 해당 region의 행 위치와 날짜
    asos_mask  = asos["region"] == asos_region
    asos_dates = asos.loc[asos_mask, "date"]  # index=ASOS row idx, value=date

    updated_count = 0
    for col in ["avg_ta", "sum_rn"]:
        # 날짜를 키로 AWS 값 조회 → 매칭 없으면 NaN
        aws_values = asos_dates.map(aws_sub[col].to_dict())

        # NaN이 아닌 행만 업데이트 (NaN이면 기존 ASOS 값 유지 = fallback)
        valid = aws_values.dropna()
        if len(valid) > 0:
            asos.loc[valid.index, col] = valid.values

        if col == "avg_ta":
            updated_count = len(valid)

    # ── 옵션: stn_id / stn_name을 AWS 관측소 정보로 업데이트하려면 아래 주석 해제
    # aws_stn_row = aws[aws["region"] == asos_region].iloc[0]
    # asos.loc[asos_mask, "stn_id"]   = aws_stn_row["stn_id"]
    # asos.loc[asos_mask, "stn_name"] = aws_stn_row["stn_name"]

    return int(asos_mask.sum()), updated_count


def main() -> None:
    print("=" * 60)
    print("  ASOS → AWS 실관측 데이터 덮어쓰기")
    print("=" * 60)

    # ── 파일 로드 ────────────────────────────────────────────────────────────
    print(f"\n[1] ASOS 로드: {ASOS_PATH}")
    asos = load_asos(ASOS_PATH)
    print(f"  행 수: {len(asos):,}  /  지역 수: {asos['region'].nunique()}")

    print(f"\n[2] AWS  로드: {AWS_PATH}")
    aws = load_aws(AWS_PATH)
    print(f"  행 수: {len(aws):,}  /  지역 수: {aws['region'].nunique()}")

    # ── AWS에서 매핑 안 된 지점 경고 ─────────────────────────────────────────
    unmapped = aws[aws["region"].isna()]["stn_name"].unique()
    if len(unmapped) > 0:
        print(f"\n  [WARNING] AWS_STN_TO_ASOS_REGION에 없는 지점: {list(unmapped)}")

    # ── 지역별 덮어쓰기 ──────────────────────────────────────────────────────
    target_regions = list(AWS_STN_TO_ASOS_REGION.values())
    print(f"\n[3] 지역별 업데이트 ({len(target_regions)}개 지역)")
    print("-" * 60)

    total_updated = 0
    for asos_region in target_regions:
        region_rows, updated = overwrite_region(asos, aws, asos_region)

        if region_rows == 0:
            print(f"  [SKIP] '{asos_region}' — ASOS에 해당 지역 없음")
        elif updated == 0:
            print(f"  [WARN] '{asos_region}' — 매칭된 날짜 없음")
        else:
            fallback = region_rows - updated
            fallback_str = f", fallback {fallback}일" if fallback > 0 else ""
            print(f"  [OK]   '{asos_region}' ({updated}일 업데이트{fallback_str})")
            total_updated += updated

    # ── 저장 ─────────────────────────────────────────────────────────────────
    print("-" * 60)
    print(f"\n[4] 저장: {OUTPUT_PATH}")
    asos.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

    print(f"\n완료  총 업데이트: {total_updated:,}행")
    print("=" * 60)


if __name__ == "__main__":
    main()
