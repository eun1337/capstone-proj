"""
evaluate_consistency.py

A/B 두 센터의 1~3차 실험 결과를 비교하여
모델별 답변 일관성(Consistency)을 채점합니다.

[일관성 기준]
- 대/중/소분류 명칭 텍스트가 1차=2차=3차 모두 완전히 일치 → O
- 하나라도 다르면 → X

[출력 파일]  data/master/model_consistency_evaluation.xlsx
  시트 1: 종합요약   — 센터×구분(전체/결측/정상)별 모델 일관성 비율(%)
  시트 2: A센터_상세 — 행별 1~3차 예측 및 일관성 O/X 채점지
  시트 3: B센터_상세 — 위와 동일
"""

import pathlib
import sys
import pandas as pd
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter

sys.stdout.reconfigure(encoding="utf-8")

# ── 경로 ─────────────────────────────────────────────────────────────────────
_HERE = pathlib.Path(__file__).resolve().parent
_DATA = (_HERE / "../../data/master").resolve()
OUTPUT_FILE = str(_DATA / "model_consistency_evaluation.xlsx")

# ── 입력 파일 ─────────────────────────────────────────────────────────────────
CENTER_FILES = {
    "A": [
        "a_llm_testset_375_final.xlsx",
        "a_llm_testset_375_final_2.xlsx",
        "a_llm_testset_375_final_3.xlsx",
    ],
    "B": [
        "b_llm_testset_374_final.xlsx",
        "b_llm_testset_374_final_2.xlsx",
        "b_llm_testset_374_final_3.xlsx",
    ],
}

MODELS      = ["GPT", "Gemini", "Claude", "Clova"]
LEVELS      = ["대분류", "중분류", "소분류"]
MERGE_KEY   = "sample_id"
TEST_CASE   = "테스트케이스"
STATIC_COLS = [MERGE_KEY, TEST_CASE, "바코드", "상품명",
               "정답_대분류", "정답_중분류", "정답_소분류"]


# ── 데이터 로딩 & 병합 ────────────────────────────────────────────────────────
def load_and_merge(center: str) -> pd.DataFrame:
    """3개 회차 파일을 sample_id 기준으로 가로 병합. 컬럼에 _r1/_r2/_r3 접미사."""
    fnames = CENTER_FILES[center]
    static_df = None
    round_frames = []

    for round_n, fname in enumerate(fnames, 1):
        df = pd.read_excel(str(_DATA / fname), dtype=str)

        if round_n == 1:
            static_df = df[[c for c in STATIC_COLS if c in df.columns]].copy()
            for col in static_df.columns:
                static_df[col] = static_df[col].fillna("").str.strip()

        # 모델 예측 컬럼만 추출하고 회차 접미사 부착
        cols = {MERGE_KEY: df[MERGE_KEY].str.strip()}
        for model in MODELS:
            for level in LEVELS:
                src = f"{model}_{level}"
                if src in df.columns:
                    cols[f"{model}_{level}_r{round_n}"] = df[src].fillna("").str.strip()
        round_frames.append(pd.DataFrame(cols))

    # 순차 병합
    merged = round_frames[0]
    for rdf in round_frames[1:]:
        merged = merged.merge(rdf, on=MERGE_KEY, how="inner")

    return static_df.merge(merged, on=MERGE_KEY, how="inner")


# ── 일관성 판정 ───────────────────────────────────────────────────────────────
def _pred_str(row, model: str, round_n: int) -> str:
    """한 회차의 대>중>소 예측을 하나의 문자열로 반환."""
    return " > ".join(
        str(row.get(f"{model}_{level}_r{round_n}", "")).strip()
        for level in LEVELS
    )


def add_consistency_columns(df: pd.DataFrame) -> pd.DataFrame:
    """각 모델별로 _일관성(O/X), _1차~_3차(예측 문자열) 컬럼 추가."""
    for model in MODELS:
        # 회차별 예측 텍스트 컬럼
        for r in [1, 2, 3]:
            df[f"{model}_{r}차"] = df.apply(
                lambda row, m=model, rn=r: _pred_str(row, m, rn), axis=1
            )

        # 일관성: 대/중/소 모두 3회차 일치 여부
        def check(row, m=model):
            for level in LEVELS:
                vals = {str(row.get(f"{m}_{level}_r{i}", "")).strip() for i in [1, 2, 3]}
                if len(vals) > 1:
                    return "X"
            return "O"

        df[f"{model}_일관성"] = df.apply(check, axis=1)

    return df


# ── 요약 집계 ─────────────────────────────────────────────────────────────────
def make_summary(center_dfs: dict) -> pd.DataFrame:
    rows = []
    for center in ["A", "B"]:
        df = center_dfs[center]
        subsets = [
            ("전체", pd.Series(True, index=df.index)),
            ("결측", df[TEST_CASE].str.strip() == "결측"),
            ("정상", df[TEST_CASE].str.strip() == "정상"),
        ]
        for label, mask in subsets:
            sub = df[mask]
            n   = len(sub)
            row = {"센터": center, "구분": label, "행수": n}
            for model in MODELS:
                n_ok = (sub[f"{model}_일관성"] == "O").sum()
                row[f"{model} 일관성(%)"] = round(n_ok / n * 100, 1) if n > 0 else float("nan")
                row[f"{model} 일관 수"]   = int(n_ok)
            rows.append(row)
    return pd.DataFrame(rows)


# ── 상세 채점지 DataFrame 생성 ────────────────────────────────────────────────
def make_detail_df(df: pd.DataFrame) -> pd.DataFrame:
    """종합 열 순서: 기본정보 → [모델별: 1차·2차·3차예측, 일관성O/X]"""
    base_cols = [c for c in STATIC_COLS if c in df.columns]
    model_cols = []
    for model in MODELS:
        for r in [1, 2, 3]:
            model_cols.append(f"{model}_{r}차")
        model_cols.append(f"{model}_일관성")

    return df[base_cols + model_cols].copy()


# ── 콘솔 출력 ─────────────────────────────────────────────────────────────────
def print_summary(summary_df: pd.DataFrame) -> None:
    SEP = "=" * 72
    print(f"\n{SEP}")
    print("  모델별 답변 일관성 평가 결과")
    print(SEP)

    for center in ["A", "B"]:
        sub = summary_df[summary_df["센터"] == center]
        print(f"\n■ {center}센터")
        header = f"  {'구분':6s}  {'행수':>5s}  " + "  ".join(f"{m:>9s}" for m in MODELS)
        print(header)
        print("  " + "-" * (len(header) - 2))
        for _, row in sub.iterrows():
            vals = "  ".join(f"{row[f'{m} 일관성(%)']:>8.1f}%" for m in MODELS)
            print(f"  {row['구분']:6s}  {int(row['행수']):>5d}  {vals}")

    print(f"\n{SEP}")
    print("  종합 (A+B 합산)")
    print(SEP)

    # A+B 합산 (행수 더하고 정확도 재계산)
    for label in ["전체", "결측", "정상"]:
        sub = summary_df[summary_df["구분"] == label]
        total_n = int(sub["행수"].sum())
        vals = []
        for m in MODELS:
            total_ok = sub[f"{m} 일관 수"].sum()
            pct = round(total_ok / total_n * 100, 1) if total_n > 0 else 0
            vals.append(f"{pct:>8.1f}%")
        print(f"  {label:6s}  {total_n:>5d}  {'  '.join(vals)}")
    print()


# ── Excel 스타일링 ────────────────────────────────────────────────────────────
C = {
    "blue":         "2F5597",
    "orange":       "C55A11",
    "green_dark":   "375623",
    "red_dark":     "9C0006",
    "green_light":  "E2EFDA",
    "yellow_light": "FFF2CC",
    "red_light":    "FCE4D6",
    "gray":         "F2F2F2",
    "white":        "FFFFFF",
    "model_header": {
        "GPT":    "BDD7EE",  # 하늘
        "Gemini": "E2EFDA",  # 연초록
        "Claude": "FCE4D6",  # 연살구
        "Clova":  "FFF2CC",  # 연노랑
    },
}


def _fill(hex_color: str) -> PatternFill:
    return PatternFill("solid", fgColor=hex_color)


def _font(bold=False, color="000000", size=10) -> Font:
    return Font(bold=bold, color=color, size=size)


def _border() -> Border:
    s = Side(style="thin", color="CCCCCC")
    return Border(left=s, right=s, top=s, bottom=s)


def _center_align(wrap=False) -> Alignment:
    return Alignment(horizontal="center", vertical="center", wrap_text=wrap)


def _left_align() -> Alignment:
    return Alignment(horizontal="left", vertical="center")


def _auto_width(ws, min_w=8, max_w=30):
    for col_cells in ws.columns:
        ltr = get_column_letter(col_cells[0].column)
        w = max(
            len(str(col_cells[0].value or "")),
            max((len(str(c.value or "")) for c in col_cells[1:]), default=0),
        )
        ws.column_dimensions[ltr].width = min(max(w * 1.2 + 2, min_w), max_w)


def style_summary_sheet(ws):
    """종합요약 시트 스타일링."""
    # 헤더 행
    for cell in ws[1]:
        cell.fill      = _fill(C["blue"])
        cell.font      = _font(bold=True, color="FFFFFF", size=10)
        cell.alignment = _center_align(wrap=True)
        cell.border    = _border()
    ws.row_dimensions[1].height = 32

    header_map = {cell.value: cell.column for cell in ws[1]}

    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row:
            cell.border    = _border()
            cell.alignment = _center_align()
            hdr = ws.cell(1, cell.column).value or ""

            # 센터·구분 열: 굵게
            if hdr in ("센터", "구분"):
                cell.font = _font(bold=True, size=10)
                cell.fill = _fill(C["gray"])
            elif "%" in str(hdr):
                cell.font = _font(size=10)
                try:
                    val = float(cell.value)
                    if val >= 80:
                        cell.fill = _fill(C["green_light"])
                        cell.font = _font(bold=True, color=C["green_dark"])
                    elif val >= 60:
                        cell.fill = _fill(C["yellow_light"])
                    else:
                        cell.fill = _fill(C["red_light"])
                        cell.font = _font(bold=True, color=C["red_dark"])
                except (TypeError, ValueError):
                    pass
            else:
                cell.font = _font(size=10)

    _auto_width(ws, min_w=8, max_w=18)
    ws.freeze_panes = "C2"


def style_detail_sheet(ws):
    """상세 채점지 시트 스타일링."""
    header_vals = [cell.value for cell in ws[1]]
    header_map  = {v: i + 1 for i, v in enumerate(header_vals)}

    # 모델별 헤더 색상 결정 함수
    def model_of(col_name: str) -> str | None:
        for m in MODELS:
            if col_name and col_name.startswith(m):
                return m
        return None

    # ── 헤더 행 ─────────────────────────────────────────────────────────────
    base_hdrs = set(STATIC_COLS)
    for cell in ws[1]:
        cname = cell.value or ""
        m = model_of(cname)
        if cname in base_hdrs:
            cell.fill = _fill(C["blue"])
            cell.font = _font(bold=True, color="FFFFFF", size=9)
        elif m:
            cell.fill = _fill(C["model_header"][m])
            cell.font = _font(bold=True, color="000000", size=9)
        else:
            cell.fill = _fill(C["gray"])
            cell.font = _font(bold=True, size=9)
        cell.alignment = _center_align(wrap=True)
        cell.border    = _border()
    ws.row_dimensions[1].height = 34

    # ── 데이터 행 ────────────────────────────────────────────────────────────
    consistency_cols = {header_map[f"{m}_일관성"] for m in MODELS if f"{m}_일관성" in header_map}
    name_col         = header_map.get("상품명")

    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row:
            cell.border = _border()
            cname = ws.cell(1, cell.column).value or ""
            m = model_of(cname)

            # 상품명: 좌측 정렬
            if cell.column == name_col:
                cell.alignment = _left_align()
                cell.font      = _font(size=9)
                cell.fill      = _fill(C["gray"])
                continue

            # 기본 정보 열 (정답 등)
            if cname in base_hdrs:
                cell.alignment = _center_align()
                cell.font      = _font(size=9)
                cell.fill      = _fill(C["gray"])
                continue

            # 일관성 O/X 셀
            if cell.column in consistency_cols:
                cell.alignment = _center_align()
                if cell.value == "O":
                    cell.fill = _fill(C["green_light"])
                    cell.font = _font(bold=True, color=C["green_dark"], size=10)
                elif cell.value == "X":
                    cell.fill = _fill(C["red_light"])
                    cell.font = _font(bold=True, color=C["red_dark"], size=10)
                else:
                    cell.font = _font(size=9)
                continue

            # 예측 텍스트 셀 (1·2·3차)
            if m:
                cell.alignment = _center_align(wrap=True)
                cell.font      = _font(size=8)
            else:
                cell.alignment = _center_align()
                cell.font      = _font(size=9)

    _auto_width(ws, min_w=8, max_w=28)
    ws.freeze_panes = "E2"   # 기본정보 4열 이후부터 스크롤


# ── Excel 저장 ────────────────────────────────────────────────────────────────
def save_excel(summary_df: pd.DataFrame, detail_dfs: dict) -> None:
    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        summary_df.to_excel(writer, sheet_name="종합요약", index=False)
        detail_dfs["A"].to_excel(writer, sheet_name="A센터_상세", index=False)
        detail_dfs["B"].to_excel(writer, sheet_name="B센터_상세", index=False)

    wb = openpyxl.load_workbook(OUTPUT_FILE)

    style_summary_sheet(wb["종합요약"])
    style_detail_sheet(wb["A센터_상세"])
    style_detail_sheet(wb["B센터_상세"])

    wb.save(OUTPUT_FILE)
    print(f"저장 완료 → '{OUTPUT_FILE}'")


# ── 메인 ─────────────────────────────────────────────────────────────────────
def main():
    center_dfs = {}

    for center in ["A", "B"]:
        print(f"\n{center}센터 데이터 로딩 & 병합 중...")
        merged = load_and_merge(center)
        print(f"  → {len(merged)}행 병합 완료")

        print(f"  일관성 채점 중...")
        merged = add_consistency_columns(merged)

        # 간단 미리보기
        for model in MODELS:
            n_ok  = (merged[f"{model}_일관성"] == "O").sum()
            total = len(merged)
            print(f"    {model:8s}: {n_ok}/{total} ({round(n_ok/total*100,1)}%) 일관")

        center_dfs[center] = merged

    summary_df  = make_summary(center_dfs)
    detail_dfs  = {c: make_detail_df(df) for c, df in center_dfs.items()}

    print_summary(summary_df)

    print("엑셀 저장 중...")
    save_excel(summary_df, detail_dfs)


if __name__ == "__main__":
    main()
