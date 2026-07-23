"""
build_product_master_filter.py

설명: 판정 파일(product_conflict_resolved_A/B.xlsx)과 센터별 매출 원천(규격 포함)을
     이용해 상품 식별 키를 [바코드 + 옵션코드 + 상품클러스터]로 통일하는 상품 마스터
     필터 테이블을 만들고, final/cleaned_main_joined.parquet에 적용합니다.

[처리 흐름]
Step 0. 규격 불일치 선행 검증
    - 판정 파일에 등록되지 않은 [바코드+옵션코드] 조합 중, 센터별 매출 원천에서
      '규격' 텍스트가 2개 이상으로 갈리는 케이스를 찾아 리포트로 출력.
    - 기본적으로 여기서 멈춘다. (input()으로 'go'/'proceed' 입력 대기)
      비대화형으로 실행되면(stdin 없음) 안내 메시지를 출력하고 종료하며,
      --proceed 옵션을 주면 Step 1/2까지 이어서 실행한다.
Step 1. 센터별 상품 마스터 필터 테이블 생성 (product_master_filter_A/B.parquet)
    - MERGE: [바코드+옵션코드] 단위로 대표상품명/규격 하나로 통일, 상품클러스터=1
    - SPLIT_PRODUCT: 판정 파일의 원본상품명까지 키로 사용해 상품클러스터를 그대로 부여
    - 일반 상품(판정 파일 미등록, 규격 충돌 없음): 원천 데이터의 상품명/규격을 그대로
      사용, 상품클러스터=1
    - 규격 오표기(SPEC_CORRECTIONS, 사용자가 직접 검토해 확정한 19건): 정답 규격으로
      통일한 뒤 위 '일반 상품'과 동일하게 처리(클러스터=1)
    - 규격이 진짜로 갈리는 조합(Step 0에서 발견, 보정 대상 아닌 나머지): 규격별로
      상품클러스터 1,2...를 자동 부여(AUTO_SPLIT_BY_SPEC). 상품명이 규격 간에 동일해
      구분이 안 되므로, [바코드+옵션코드+거래일+수량+금액]으로 원본 규격을 역추적해
      매칭한다(검증 결과 이 조합으로 규격이 100% 유일하게 결정됨).
Step 2. final/cleaned_main_joined.parquet에 좌측조인(Left Join) 적용
    - 우선순위: AUTO_SPLIT_BY_SPEC(바코드+옵션코드+거래일+수량+금액) >
      SPLIT_PRODUCT(바코드+옵션코드+원본상품명) > MERGE/GENERAL(바코드+옵션코드)
    - 상품명/규격/입수를 마스터의 대표값으로 갱신하고 상품클러스터 컬럼 추가
      (입수는 판정 파일의 '입수_정규화', 매출 원천의 '입수' 컬럼에서 가져옴 -
       cleaned_main_joined.parquet에는 원래 입수 컬럼이 없었음)
    - 결과는 원본을 덮어쓰지 않고 cleaned_main_joined_cleaned_for_pred.parquet로 저장

[사용법]
    python build_product_master_filter.py            # Step 0(규격 검증)만 실행
    python build_product_master_filter.py --proceed  # Step 0 검증 후 Step 1/2까지 실행
"""

import argparse
import os

import pandas as pd

from normalize_option_code import OPTION_MAP

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
PARQUET_DIR = os.path.join(DATA_DIR, "parquet")
FINAL_DIR = os.path.join(DATA_DIR, "final")

FINAL_SALES_FILE = os.path.join(FINAL_DIR, "cleaned_main_joined.parquet")
OUT_FINAL_FILE = os.path.join(FINAL_DIR, "cleaned_main_joined_cleaned_for_pred.parquet")

# cleaned_main_joined.parquet은 '매출' 행만 있어서(유형 컬럼 확인 결과), 마스터도
# 매출 원천만 사용한다. 매입 파일은 대상에서 제외.
CENTERS = {
    "A": {
        "judgment_file": os.path.join(FINAL_DIR, "product_conflict_resolved_A.xlsx"),
        "judgment_sheet": "Sheet1",
        "source_files": [
            os.path.join(PARQUET_DIR, "a_center_sales_2021_2023_mapped.parquet"),
            os.path.join(PARQUET_DIR, "a_center_sales_2024_mapped.parquet"),
        ],
        "master_filter_out": os.path.join(FINAL_DIR, "product_master_filter_A.parquet"),
    },
    "B": {
        "judgment_file": os.path.join(FINAL_DIR, "product_conflict_resolved_B.xlsx"),
        "judgment_sheet": "시트1",
        "source_files": [
            os.path.join(PARQUET_DIR, "b_center_sales_2021_2023_mapped.parquet"),
            os.path.join(PARQUET_DIR, "b_center_sales_2024_mapped.parquet"),
        ],
        "master_filter_out": os.path.join(FINAL_DIR, "product_master_filter_B.parquet"),
    },
}

RAW_COLS = ["바코드", "옵션코드", "규격", "입수", "상품명", "판매일", "판매수량", "공급가액"]
JUDGMENT_COLS = ["바코드", "옵션코드", "원본상품명", "권장대표상품명", "규격", "입수", "처리유형", "상품클러스터"]

# 사용자가 직접 검토해서 확정한 규격 오표기 보정. 판정 파일에 없는 [바코드+옵션코드]
# 조합인데 원천 데이터에 규격이 2가지로 갈리던 것 중, 실제로는 같은 상품이고 표기만
# 달랐던 19건 -> 정답 규격 하나로 통일한다(보정 후에는 GENERAL로 자연스럽게 처리됨).
SPEC_CORRECTIONS = {
    ("18801223100503", "BX"): "30입",
    ("18806011618246", "BX"): "50입",
    ("18809422044437", "BX"): "24입",
    ("18809422044444", "BX"): "24입",
    ("8801223100254", "EA"): "30입",
    ("8806011618256", "CS"): "50입",
    ("8809422041309", "CS"): "1CS",
    ("8809422043891", "CS"): "1CS",
    ("8809422044430", "EA"): "24입",
    ("8809422044447", "EA"): "24입",
    ("18801037040026", "CS"): "36",
    ("18801037040040", "BX"): "36",
    ("18801037042860", "CS"): "36",
    ("18801037042877", "BX"): "36",
    ("8801007002989", "EA"): "48",
    ("8801007054186", "EA"): "30",
    ("8801037039993", "EA"): "36",
    ("8801037040029", "EA"): "36",
    ("8801037042863", "EA"): "36",
    # 바코드 앞자리 0 버그 수정 후 새로 발견된 규격 충돌. 기간이 완전히 안 겹치고
    # (2022-03~10 규격 15 / 2022-12~2023-09 규격 1) 수량 대비 금액이 일관되게
    # 비례해 규격 차이가 아니라 유통 표기 변경으로 판단, 사용자 확정으로 병합.
    ("80765486", "EA"): "1",
}

# 위와 같은 이유로 상품명 표기도 함께 바뀐 경우 대표명을 강제 통일한다.
NAME_CORRECTIONS = {
    ("80765486", "EA"): "페레로누텔라370g",
}


def _strip(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip()


def _normalize_barcode(s: pd.Series) -> pd.Series:
    """센터별 매출 원천 parquet은 바코드 앞자리 0이 살아있는데, cleaned_main_joined와
    판정 파일은 이미 앞자리 0이 소실된 채로 저장되어 있어(둘 다 숫자 0 충돌 없음을
    확인함) 비교 전 항상 앞자리 0을 떼어 통일한다."""
    stripped = _strip(s).str.lstrip("0")
    return stripped.where(stripped != "", "0")


def load_center_raw(center: str) -> pd.DataFrame:
    frames = [pd.read_parquet(p, columns=RAW_COLS) for p in CENTERS[center]["source_files"]]
    raw = pd.concat(frames, ignore_index=True)

    for col in ["옵션코드", "규격", "입수", "상품명"]:
        raw[col] = _strip(raw[col])
    raw["바코드"] = _normalize_barcode(raw["바코드"])

    unmapped = set(raw["옵션코드"].unique()) - set(OPTION_MAP.keys())
    if unmapped:
        print(f"  [!] {center}센터 원천에 OPTION_MAP 미등록 옵션코드 발견(원본 유지): {sorted(unmapped)}")
    raw["옵션코드"] = raw["옵션코드"].map(lambda v: OPTION_MAP.get(v, v))

    raw["규격"] = raw["규격"].replace({"nan": pd.NA, "": pd.NA})
    raw["입수"] = pd.to_numeric(raw["입수"].replace({"nan": pd.NA, "": pd.NA}), errors="coerce")

    for (bc, opt), correct_spec in SPEC_CORRECTIONS.items():
        mask = (raw["바코드"] == bc) & (raw["옵션코드"] == opt)
        raw.loc[mask, "규격"] = correct_spec

    for (bc, opt), correct_name in NAME_CORRECTIONS.items():
        mask = (raw["바코드"] == bc) & (raw["옵션코드"] == opt)
        raw.loc[mask, "상품명"] = correct_name

    raw["_key"] = raw["바코드"] + "||" + raw["옵션코드"]
    return raw


def load_judgment(center: str) -> pd.DataFrame:
    cfg = CENTERS[center]
    df = pd.read_excel(cfg["judgment_file"], sheet_name=cfg["judgment_sheet"], dtype=str)
    df = df.rename(columns={"옵션코드_정규화": "옵션코드", "입수_정규화": "입수"})

    for col in JUDGMENT_COLS:
        df[col] = _strip(df[col])
    df["바코드"] = _normalize_barcode(df["바코드"])

    df["입수"] = pd.to_numeric(df["입수"], errors="coerce")
    df["처리유형"] = df["처리유형"].str.upper()
    df["_key"] = df["바코드"] + "||" + df["옵션코드"]
    return df


def find_spec_conflicts(center: str, raw: pd.DataFrame, judgment: pd.DataFrame) -> pd.DataFrame:
    """판정 파일에 없는 [바코드+옵션코드] 중 규격이 2개 이상으로 갈리는 조합 리포트."""
    judged_keys = set(judgment["_key"])
    unjudged = raw[~raw["_key"].isin(judged_keys) & raw["규격"].notna()]

    spec_nunique = unjudged.groupby(["바코드", "옵션코드"])["규격"].nunique()
    conflict_pairs = spec_nunique[spec_nunique > 1].index

    if len(conflict_pairs) == 0:
        return pd.DataFrame(
            columns=["센터", "바코드", "옵션코드", "규격_종류수", "총건수", "규격_목록", "상품명_예시"]
        )

    conflict_idx = pd.MultiIndex.from_tuples(conflict_pairs, names=["바코드", "옵션코드"])
    subset = unjudged.set_index(["바코드", "옵션코드"]).loc[conflict_idx].reset_index()

    def spec_summary(s: pd.Series) -> str:
        vc = s.value_counts()
        return ", ".join(f"{k}({v}건)" for k, v in vc.items())

    report = (
        subset.groupby(["바코드", "옵션코드"])
        .agg(
            규격_종류수=("규격", "nunique"),
            총건수=("규격", "size"),
            규격_목록=("규격", spec_summary),
            상품명_예시=("상품명", "first"),
        )
        .reset_index()
    )
    report.insert(0, "센터", center)
    return report.sort_values("총건수", ascending=False)


def build_master_filter(center: str, raw: pd.DataFrame, judgment: pd.DataFrame, conflict_report: pd.DataFrame) -> pd.DataFrame:
    conflict_keys = set(conflict_report["바코드"] + "||" + conflict_report["옵션코드"]) if not conflict_report.empty else set()

    rows = []

    # 1) MERGE -> [바코드+옵션코드] 단위로 대표상품명/규격 하나, 상품클러스터=1
    merge_df = judgment[judgment["처리유형"] == "MERGE"]
    for (bc, opt), g in merge_df.groupby(["바코드", "옵션코드"]):
        rep_names = g["권장대표상품명"].unique()
        rep_specs = g["규격"].unique()
        rep_counts = g["입수"].unique()
        if len(rep_names) > 1 or len(rep_specs) > 1:
            print(f"  [!] {center} MERGE 그룹 내 대표상품명/규격이 여러 개: {bc}/{opt} -> {list(rep_names)}, {list(rep_specs)}")
        rows.append(
            {
                "바코드": bc,
                "옵션코드": opt,
                "원본상품명_키": None,
                "거래일_키": pd.NaT,
                "수량_키": None,
                "금액_키": None,
                "대표상품명": rep_names[0],
                "규격": rep_specs[0],
                "입수": rep_counts[0],
                "상품클러스터": 1,
                "처리유형": "MERGE",
            }
        )

    # 2) SPLIT_PRODUCT -> 원본상품명까지 키로 사용, 판정 파일의 상품클러스터 그대로
    #    REVIEW도 사용자 검토 결과 서로 다른 상품으로 확정되어 동일하게 처리한다.
    #    단, REVIEW는 파일 상 상품클러스터가 그룹 내에서 전부 1로 동일해 신뢰할 수
    #    없으므로 그룹 내 등장 순서로 1,2...를 다시 부여한다.
    split_df = judgment[judgment["처리유형"].isin(["SPLIT_PRODUCT", "REVIEW"])].copy()
    review_mask = split_df["처리유형"] == "REVIEW"
    if review_mask.any():
        split_df.loc[review_mask, "상품클러스터"] = (
            split_df[review_mask].groupby(["바코드", "옵션코드"]).cumcount() + 1
        ).astype(str)
        # REVIEW는 권장대표상품명이 그룹 내에서 전부 동일해(=아직 대표명이 정해지지
        # 않은 상태) 클러스터 구분이 안 되므로, 원본상품명을 그대로 대표상품명으로 쓴다.
        split_df.loc[review_mask, "권장대표상품명"] = split_df.loc[review_mask, "원본상품명"]

    for _, r in split_df.iterrows():
        rows.append(
            {
                "바코드": r["바코드"],
                "옵션코드": r["옵션코드"],
                "원본상품명_키": r["원본상품명"],
                "거래일_키": pd.NaT,
                "수량_키": None,
                "금액_키": None,
                "대표상품명": r["권장대표상품명"],
                "규격": r["규격"],
                "입수": r["입수"],
                "상품클러스터": int(r["상품클러스터"]),
                "처리유형": "SPLIT_PRODUCT",
            }
        )

    # 3) 일반 상품(판정 파일 미등록, 규격 충돌 없음) -> 원천의 상품명/규격 그대로, 클러스터=1
    judged_keys = set(judgment["_key"])
    general_raw = raw[~raw["_key"].isin(judged_keys) & ~raw["_key"].isin(conflict_keys)]
    general_agg = general_raw.groupby(["바코드", "옵션코드"]).agg(
        대표상품명=("상품명", lambda s: s.value_counts().idxmax()),
        규격=("규격", lambda s: s.dropna().iloc[0] if s.notna().any() else pd.NA),
        입수=("입수", lambda s: s.dropna().iloc[0] if s.notna().any() else pd.NA),
    )
    for (bc, opt), r in general_agg.iterrows():
        rows.append(
            {
                "바코드": bc,
                "옵션코드": opt,
                "원본상품명_키": None,
                "거래일_키": pd.NaT,
                "수량_키": None,
                "금액_키": None,
                "대표상품명": r["대표상품명"],
                "규격": r["규격"],
                "입수": r["입수"],
                "상품클러스터": 1,
                "처리유형": "GENERAL",
            }
        )

    # 4) 일반 상품인데 규격이 진짜로 갈리는 조합(사용자 검토 결과, 보정 대상 아님)
    #    -> 규격별로 상품클러스터 1,2...를 자동 부여. 상품명이 동일해서 구분이 안 되므로
    #    [바코드+옵션코드+거래일+수량+금액]으로 원본 규격을 역추적해 매칭한다.
    conflict_raw = raw[raw["_key"].isin(conflict_keys) & raw["규격"].notna()]
    for (bc, opt), g in conflict_raw.groupby(["바코드", "옵션코드"]):
        spec_order = g["규격"].value_counts().index.tolist()  # 빈도 내림차순 -> 클러스터 1,2...
        spec_to_cluster = {spec: i + 1 for i, spec in enumerate(spec_order)}
        print(f"  [자동분리] {center} {bc}/{opt}: {spec_to_cluster}")

        keyed = g.drop_duplicates(subset=["판매일", "판매수량", "공급가액"])
        for _, r in keyed.iterrows():
            rows.append(
                {
                    "바코드": bc,
                    "옵션코드": opt,
                    "원본상품명_키": None,
                    "거래일_키": r["판매일"],
                    "수량_키": r["판매수량"],
                    "금액_키": r["공급가액"],
                    "대표상품명": r["상품명"],
                    "규격": r["규격"],
                    "입수": r["입수"],
                    "상품클러스터": spec_to_cluster[r["규격"]],
                    "처리유형": "AUTO_SPLIT_BY_SPEC",
                }
            )

    master = pd.DataFrame(rows)
    # dict 리스트로 만들면 NaT/None이 섞인 컬럼이 object dtype이 되어 최종 데이터와
    # 병합(merge) 시 dtype 충돌이 나므로 명시적으로 캐스팅한다.
    master["거래일_키"] = pd.to_datetime(master["거래일_키"])
    master["수량_키"] = pd.to_numeric(master["수량_키"])
    master["금액_키"] = pd.to_numeric(master["금액_키"])
    master["입수"] = pd.to_numeric(master["입수"])

    print(f"  {center}센터 마스터 필터: {len(master):,}건 "
          f"(MERGE {sum(master['처리유형']=='MERGE'):,}, "
          f"SPLIT_PRODUCT {sum(master['처리유형']=='SPLIT_PRODUCT'):,}, "
          f"GENERAL {sum(master['처리유형']=='GENERAL'):,}, "
          f"AUTO_SPLIT_BY_SPEC {sum(master['처리유형']=='AUTO_SPLIT_BY_SPEC'):,})")
    return master


def apply_master_filter(sales_df: pd.DataFrame, master: pd.DataFrame) -> pd.DataFrame:
    sales_df = sales_df.copy()

    auto_map = master[master["처리유형"] == "AUTO_SPLIT_BY_SPEC"][
        ["바코드", "옵션코드", "거래일_키", "수량_키", "금액_키", "대표상품명", "규격", "입수", "상품클러스터"]
    ].rename(
        columns={
            "거래일_키": "_거래일_dt", "수량_키": "수량", "금액_키": "금액",
            "대표상품명": "_대표상품명_auto", "규격": "_규격_auto", "입수": "_입수_auto", "상품클러스터": "_클러스터_auto",
        }
    )

    split_map = master[master["처리유형"] == "SPLIT_PRODUCT"][
        ["바코드", "옵션코드", "원본상품명_키", "대표상품명", "규격", "입수", "상품클러스터"]
    ].rename(columns={
        "원본상품명_키": "상품명", "대표상품명": "_대표상품명_split", "규격": "_규격_split",
        "입수": "_입수_split", "상품클러스터": "_클러스터_split",
    })

    general_map = master[master["처리유형"].isin(["MERGE", "GENERAL"])][
        ["바코드", "옵션코드", "대표상품명", "규격", "입수", "상품클러스터"]
    ].rename(columns={
        "대표상품명": "_대표상품명_gen", "규격": "_규격_gen", "입수": "_입수_gen", "상품클러스터": "_클러스터_gen",
    })

    sales_df = sales_df.merge(auto_map, on=["바코드", "옵션코드", "_거래일_dt", "수량", "금액"], how="left")
    sales_df = sales_df.merge(split_map, on=["바코드", "옵션코드", "상품명"], how="left")
    sales_df = sales_df.merge(general_map, on=["바코드", "옵션코드"], how="left")

    # 우선순위: 자동분리(AUTO_SPLIT_BY_SPEC) > SPLIT_PRODUCT > MERGE/GENERAL > 원본 유지
    새상품명 = sales_df["_대표상품명_auto"].combine_first(sales_df["_대표상품명_split"]).combine_first(sales_df["_대표상품명_gen"])
    새규격 = sales_df["_규격_auto"].combine_first(sales_df["_규격_split"]).combine_first(sales_df["_규격_gen"])
    새입수 = sales_df["_입수_auto"].combine_first(sales_df["_입수_split"]).combine_first(sales_df["_입수_gen"])
    새클러스터 = sales_df["_클러스터_auto"].combine_first(sales_df["_클러스터_split"]).combine_first(sales_df["_클러스터_gen"]).fillna(1).astype(int)

    sales_df["규격"] = 새규격
    sales_df["입수"] = 새입수
    sales_df["상품명"] = 새상품명.combine_first(sales_df["상품명"])
    sales_df["상품클러스터"] = 새클러스터

    sales_df = sales_df.drop(
        columns=[
            "_거래일_dt",
            "_대표상품명_auto", "_규격_auto", "_입수_auto", "_클러스터_auto",
            "_대표상품명_split", "_규격_split", "_입수_split", "_클러스터_split",
            "_대표상품명_gen", "_규격_gen", "_입수_gen", "_클러스터_gen",
        ]
    )
    return sales_df


def run_precheck() -> dict:
    reports = {}
    raws, judgments = {}, {}
    for center in CENTERS:
        print(f"\n[ {center}센터 원천 데이터 로딩 ]")
        raws[center] = load_center_raw(center)
        judgments[center] = load_judgment(center)
        reports[center] = find_spec_conflicts(center, raws[center], judgments[center])

    full_report = pd.concat(reports.values(), ignore_index=True)

    print("\n" + "=" * 70)
    print(f"[ 규격 불일치 검증 리포트 ] 판정 파일 미등록 상품 중 규격이 갈리는 조합: 총 {len(full_report)}건")
    print("=" * 70)
    if full_report.empty:
        print("발견된 규격 불일치 없음.")
    else:
        with pd.option_context("display.max_rows", None, "display.max_colwidth", 60, "display.width", 200):
            print(full_report.to_string(index=False))

    return {"raws": raws, "judgments": judgments, "report": full_report}


def run_build_and_join(ctx: dict):
    masters = {}
    for center in CENTERS:
        print(f"\n[ {center}센터 상품 마스터 필터 생성 ]")
        master = build_master_filter(center, ctx["raws"][center], ctx["judgments"][center], ctx["report"][ctx["report"]["센터"] == center])
        out_path = CENTERS[center]["master_filter_out"]
        master.to_parquet(out_path, index=False, engine="pyarrow")
        print(f"  저장 완료: {out_path}")
        masters[center] = master

    print(f"\n[ 매출 데이터 로딩 ] {FINAL_SALES_FILE}")
    sales = pd.read_parquet(FINAL_SALES_FILE)
    sales["바코드"] = _normalize_barcode(sales["바코드"])
    sales["옵션코드"] = _strip(sales["옵션코드"])
    sales["상품명"] = _strip(sales["상품명"])
    # 거래일이 datetime.date object dtype이라 마스터의 datetime64 거래일_키와 병합이
    # 안 되므로 datetime64로 맞춘다. 원본 컬럼 자체는 그대로 두고 병합용으로만 변환.
    sales["_거래일_dt"] = pd.to_datetime(sales["거래일"])

    parts = []
    for center in CENTERS:
        part = sales[sales["센터"] == center]
        print(f"  {center}센터 {len(part):,}행에 마스터 필터 적용 중...")
        parts.append(apply_master_filter(part, masters[center]))

    result = pd.concat(parts, ignore_index=True)

    print("\n  [ 상품클러스터 분포 ]")
    print(result["상품클러스터"].value_counts().sort_index().to_string())

    auto_split_keys = set(
        zip(ctx["report"]["바코드"], ctx["report"]["옵션코드"])
    ) - set(SPEC_CORRECTIONS.keys())
    if auto_split_keys:
        mask = result.apply(lambda r: (r["바코드"], r["옵션코드"]) in auto_split_keys, axis=1)
        print(f"\n  [ 자동분리 대상 {len(auto_split_keys)}개 조합의 클러스터 배정 결과 ]")
        print(result.loc[mask].groupby(["바코드", "옵션코드", "상품클러스터"]).size().to_string())
        unmatched = mask & (result["규격"].isna())
        print(f"  -> 매칭 실패(규격 역추적 안 됨) 행: {unmatched.sum():,}건")

    print("\n  [ 수량/입수 데이터 품질 점검 ]")
    print(f"  수량 결측: {result['수량'].isna().sum():,}건, 0: {(result['수량']==0).sum():,}건, 음수: {(result['수량']<0).sum():,}건")
    print(f"  입수 결측: {result['입수'].isna().sum():,}건, 0: {(result['입수']==0).sum():,}건, 음수: {(result['입수']<0).sum():,}건")

    # 수량=0 행은 실제 판매/반품이 아닌 이상 케이스로 판단해 제거한다(음수/반품 행은
    # 처리 방침이 아직 미정이라 그대로 둔다).
    before = len(result)
    result = result[result["수량"] != 0].reset_index(drop=True)
    print(f"\n  수량=0 행 제거: {before - len(result):,}건 삭제 ({before:,} -> {len(result):,}행)")

    result.to_parquet(OUT_FINAL_FILE, index=False, engine="pyarrow")
    size_mb = os.path.getsize(OUT_FINAL_FILE) / (1024 * 1024)
    print(f"\n저장 완료: {OUT_FINAL_FILE} ({size_mb:.1f} MB, {len(result):,}행)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--proceed", action="store_true", help="Step 0 검증 후 바로 Step 1/2까지 실행")
    args = parser.parse_args()

    ctx = run_precheck()

    if not args.proceed:
        print("\n" + "-" * 70)
        try:
            while True:
                ans = input("리포트를 확인하셨습니까? 'go' 또는 'proceed' 입력 시 Step 1/2를 진행합니다 (그 외 입력 시 종료): ").strip().lower()
                if ans in ("go", "proceed"):
                    break
                print("종료합니다. 다시 실행하려면 스크립트를 재실행하거나 --proceed 옵션을 사용하세요.")
                return
        except EOFError:
            print("\n[!] 비대화형 환경이라 입력을 받을 수 없습니다.")
            print("    리포트를 검토한 뒤 아래 명령으로 이어서 실행하세요:")
            print("    python build_product_master_filter.py --proceed")
            return

    run_build_and_join(ctx)


if __name__ == "__main__":
    main()
