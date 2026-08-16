#!/usr/bin/env python3
"""
compare_llm_results.py
═══════════════════════════════════════════════════════════════════════════
설명 : LLM 분류 정답 비교 및 오답 마킹 스크립트

기능
    - GPT / Gemini / Claude / Clova 답변 셀을 정답 컬럼과 비교
    - 에러/오류가 포함된 셀은 비교에서 제외(패스) — 회색으로 표시
    - 오답 셀 배경색을 ★ #FF89BF ★ 로 마킹 (openpyxl PatternFill)
    - 두 가지 컬럼 형식 자동 감지
        · 3분류 형식 : GPT_대분류 / GPT_중분류 / GPT_소분류 + 정답_대/중/소분류
        · 단일 형식  : gpt / 제미나이 / 클로드 / 클로바 + 정답 카테고리 분류
    - 모델별 정확도 요약 시트(요약) 포함
    - 결과를 LLM_오답_마킹_리포트.xlsx 로 저장

위치
src/preprocessing/compare_llm_results.py

사용법
    # 기본 — 스크립트 기준으로 data/master/ 내 파일 자동 탐색
    python compare_llm_results.py

    # 입력 파일 직접 지정 (.xlsx 또는 .csv 모두 지원)
    python compare_llm_results.py -i data/master/a_llm_testset_375_final.xlsx

    # 출력 경로 지정
    python compare_llm_results.py -i testset.csv -o 결과리포트.xlsx

    # A·B 센터 모두 처리 (기본 동작)
    python compare_llm_results.py --all
═══════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

import argparse
import pathlib
import re
import sys
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
import openpyxl
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

# ════════════════════════════════════════════════════════════════════════
# ★ 설정 섹션 — 컬럼명이 바뀌면 여기만 수정하세요 ★
# ════════════════════════════════════════════════════════════════════════

_HERE = pathlib.Path(__file__).resolve().parent          # src/preprocessing/
_DATA = (_HERE / "../../data/master").resolve()          # data/master/

# ── 기본 입출력 경로 ─────────────────────────────────────────────────
DEFAULT_INPUTS: list[pathlib.Path] = [
    _DATA / "a_llm_testset_375_final.xlsx",
    _DATA / "b_llm_testset_374_final.xlsx",
]
DEFAULT_OUTPUT: pathlib.Path = _DATA / "LLM_오답_마킹_리포트.xlsx"

# ── 3분류 형식: 정답 컬럼 후보 (대분류, 중분류, 소분류 순서) ─────────
ANSWER_3LEVEL_CANDIDATES: list[tuple[str, str, str]] = [
    ("정답_대분류",  "정답_중분류",  "정답_소분류"),
    ("정답 대분류",  "정답 중분류",  "정답 소분류"),
    ("ans_cls1",     "ans_cls2",     "ans_cls3"),
    ("answer_cls1",  "answer_cls2",  "answer_cls3"),
]

# ── 단일 형식: 정답 컬럼 후보 ───────────────────────────────────────
ANSWER_SINGLE_CANDIDATES: list[str] = [
    "정답 카테고리 분류",
    "정답카테고리분류",
    "정답_카테고리_분류",
    "정답분류",
    "정답",
    "answer",
    "label",
    "category",
]

# ── LLM 모델 컬럼 후보 ──────────────────────────────────────────────
# 형식: (표시명, [대분류 후보], [중분류 후보], [소분류 후보])
LLM_3LEVEL: list[tuple[str, list[str], list[str], list[str]]] = [
    ("GPT",
    ["GPT_대분류",    "gpt_대분류",    "GPT대분류"],
    ["GPT_중분류",    "gpt_중분류",    "GPT중분류"],
    ["GPT_소분류",    "gpt_소분류",    "GPT소분류"]),
    ("Gemini",
    ["Gemini_대분류", "gemini_대분류", "제미나이_대분류"],
    ["Gemini_중분류", "gemini_중분류", "제미나이_중분류"],
    ["Gemini_소분류", "gemini_소분류", "제미나이_소분류"]),
    ("Claude",
    ["Claude_대분류", "claude_대분류", "클로드_대분류"],
    ["Claude_중분류", "claude_중분류", "클로드_중분류"],
    ["Claude_소분류", "claude_소분류", "클로드_소분류"]),
    ("Clova",
    ["Clova_대분류",  "clova_대분류",  "클로바_대분류"],
    ["Clova_중분류",  "clova_중분류",  "클로바_중분류"],
    ["Clova_소분류",  "clova_소분류",  "클로바_소분류"]),
]

# 형식: (표시명, [단일 컬럼 후보들])
LLM_SINGLE: list[tuple[str, list[str]]] = [
    ("GPT",    ["gpt", "GPT", "chatgpt", "ChatGPT", "gpt4", "gpt-4"]),
    ("Gemini", ["gemini", "제미나이", "Gemini", "GEMINI"]),
    ("Claude", ["claude", "클로드", "Claude", "CLAUDE"]),
    ("Clova",  ["clova", "클로바", "Clova", "CLOVA", "clova_result", "clova-x"]),
]

# ── 에러 패스 설정 ───────────────────────────────────────────────────
# 셀 값에 아래 키워드 중 하나라도 포함되면 비교 제외 (대소문자 무시)
ERROR_KEYWORDS: list[str] = [
    "코드 오류", "코드오류",
    "error", "exception", "traceback",
    "api error", "api 오류", "rate limit",
    "timeout", "timed out",
    "parsing error", "json decode",
    "invalid response", "connection error",
    "오류 발생", "오류가 발생", "실패",
]
# strip() 후 완전 일치 시 에러로 처리
ERROR_EXACT: set[str] = {"nan", "none", "null", "n/a", "-", "", "error", "na"}

# ── 색상 코드 (openpyxl: '#' 없이 6자리 HEX) ─────────────────────────
C_WRONG   = "FF89BF"   # ★ 오답 마킹 (#ff89bf)
C_SKIP    = "D9D9D9"   # 에러 패스 셀 (연회색)
C_HEADER  = "2F5496"   # 일반 헤더 (남색)
C_ANS_HD  = "1F7391"   # 정답 컬럼 헤더 (청록)
C_LLM_HD  = "7030A0"   # LLM 컬럼 헤더 (보라)
C_SUMMARY_GOOD = "C6EFCE"  # 요약: 80% 이상 (연초록)
C_SUMMARY_MID  = "FFEB9C"  # 요약: 60–79% (연노랑)
C_SUMMARY_BAD  = "FFC7CE"  # 요약: 60% 미만 (연빨강)

# ════════════════════════════════════════════════════════════════════════
# 데이터 클래스
# ════════════════════════════════════════════════════════════════════════

@dataclass
class ColumnMap:
    """감지된 컬럼 매핑 정보."""
    mode: str                                          # "3level" | "single"
    answer_cols: list[str]                             # 정답 컬럼명 리스트
    llm_cols: list[tuple[str, list[str]]]              # [(표시명, [컬럼명])]
    product_col: Optional[str] = None                  # 상품명 컬럼

@dataclass
class ModelStats:
    """모델별 채점 통계."""
    name: str
    total: int = 0
    correct: int = 0
    wrong: int = 0
    skipped: int = 0

    @property
    def evaluated(self) -> int:
        return self.total - self.skipped

    @property
    def accuracy(self) -> float:
        return round(self.correct / self.evaluated * 100, 2) if self.evaluated else 0.0

# ════════════════════════════════════════════════════════════════════════
# 유틸리티 함수
# ════════════════════════════════════════════════════════════════════════

def _normalize(v) -> str:
    """셀 값을 비교용 문자열로 정규화."""
    if v is None:
        return ""
    s = str(v).strip()
    # 괄호/공백/특수문자 정규화
    s = re.sub(r"\s+", " ", s)
    return s


def is_error_cell(value) -> bool:
    """에러/오류 셀 여부 판단 — True 이면 비교에서 제외."""
    raw = _normalize(value)
    low = raw.lower()

    # 완전 일치 (빈 셀, none, nan 등)
    if low in ERROR_EXACT:
        return True

    # 부분 포함 (대소문자 무시)
    for kw in ERROR_KEYWORDS:
        if kw.lower() in low:
            return True

    return False


def find_col(candidates: list[str], columns: list[str]) -> Optional[str]:
    """후보 이름 목록에서 실제 컬럼명과 일치하는 첫 번째를 반환."""
    col_set = set(columns)
    for c in candidates:
        if c in col_set:
            return c
    return None


def detect_product_col(columns: list[str]) -> Optional[str]:
    """상품명 컬럼 자동 탐색."""
    for candidate in ["상품명", "product_name", "품명", "item_name"]:
        if candidate in columns:
            return candidate
    return None


def detect_column_map(df: pd.DataFrame) -> ColumnMap:
    """
    DataFrame 컬럼을 분석해 ColumnMap 을 반환.
    3분류 형식 우선, 없으면 단일 형식으로 폴백.
    """
    cols = df.columns.tolist()

    # ── 3분류 형식 탐색 ───────────────────────────────────────────────
    ans3 = None
    for cand in ANSWER_3LEVEL_CANDIDATES:
        if all(c in cols for c in cand):
            ans3 = list(cand)
            break

    llm3_found: list[tuple[str, list[str]]] = []
    for name, c1s, c2s, c3s in LLM_3LEVEL:
        c1 = find_col(c1s, cols)
        c2 = find_col(c2s, cols)
        c3 = find_col(c3s, cols)
        if c1 and c2 and c3:
            llm3_found.append((name, [c1, c2, c3]))

    if ans3 and llm3_found:
        return ColumnMap(
            mode="3level",
            answer_cols=ans3,
            llm_cols=llm3_found,
            product_col=detect_product_col(cols),
        )

    # ── 단일 형식 폴백 ────────────────────────────────────────────────
    ans1 = find_col(ANSWER_SINGLE_CANDIDATES, cols)

    llm1_found: list[tuple[str, list[str]]] = []
    for name, cands in LLM_SINGLE:
        c = find_col(cands, cols)
        if c:
            llm1_found.append((name, [c]))

    if ans1 and llm1_found:
        return ColumnMap(
            mode="single",
            answer_cols=[ans1],
            llm_cols=llm1_found,
            product_col=detect_product_col(cols),
        )

    # ── 감지 실패 ─────────────────────────────────────────────────────
    raise ValueError(
        "정답 컬럼 또는 LLM 컬럼을 찾지 못했습니다.\n"
        "설정 섹션의 ANSWER_*_CANDIDATES / LLM_* 후보 목록을 확인하세요.\n"
        f"감지된 전체 컬럼: {cols}"
    )

# ════════════════════════════════════════════════════════════════════════
# 핵심 비교 로직
# ════════════════════════════════════════════════════════════════════════

@dataclass
class CellFlag:
    """행·열 인덱스(0-base) + 플래그 종류."""
    row: int
    col_names: list[str]   # 마킹할 컬럼명들
    kind: str              # "wrong" | "skip"


def compare_dataframe(df: pd.DataFrame, cmap: ColumnMap) -> tuple[list[CellFlag], list[ModelStats]]:
    """
    DataFrame을 순회하며 오답/에러 셀을 CellFlag 리스트로 반환.
    동시에 ModelStats(정확도 요약)를 계산한다.
    """
    flags: list[CellFlag] = []
    stats_map: dict[str, ModelStats] = {name: ModelStats(name=name) for name, _ in cmap.llm_cols}

    for row_idx, row in df.iterrows():
        for model_name, llm_col_list in cmap.llm_cols:
            st = stats_map[model_name]
            st.total += 1

            # ── 에러 패스 판단 ────────────────────────────────────────
            has_error = any(is_error_cell(row.get(c)) for c in llm_col_list)
            if has_error:
                st.skipped += 1
                flags.append(CellFlag(row=int(row_idx), col_names=llm_col_list, kind="skip"))
                continue

            # ── 정오답 비교 ───────────────────────────────────────────
            if cmap.mode == "3level":
                # 대/중/소 각 레벨별로 개별 비교 → 틀린 레벨 셀만 마킹
                wrong_cols = []
                all_correct = True
                for llm_c, ans_c in zip(llm_col_list, cmap.answer_cols):
                    llm_val = _normalize(row.get(llm_c))
                    ans_val = _normalize(row.get(ans_c))
                    if llm_val != ans_val:
                        wrong_cols.append(llm_c)
                        all_correct = False

                if wrong_cols:
                    st.wrong += 1
                    flags.append(CellFlag(row=int(row_idx), col_names=wrong_cols, kind="wrong"))
                else:
                    st.correct += 1

            else:  # single
                llm_c  = llm_col_list[0]
                ans_c  = cmap.answer_cols[0]
                llm_val = _normalize(row.get(llm_c))
                ans_val = _normalize(row.get(ans_c))

                if llm_val != ans_val:
                    st.wrong += 1
                    flags.append(CellFlag(row=int(row_idx), col_names=[llm_c], kind="wrong"))
                else:
                    st.correct += 1

    return flags, list(stats_map.values())

# ════════════════════════════════════════════════════════════════════════
# openpyxl 스타일링
# ════════════════════════════════════════════════════════════════════════

# PatternFill 인스턴스 (모듈 수준 캐시)
_FILL_WRONG  = PatternFill("solid", fgColor=C_WRONG)
_FILL_SKIP   = PatternFill("solid", fgColor=C_SKIP)
_FILL_H_DEF  = PatternFill("solid", fgColor=C_HEADER)
_FILL_H_ANS  = PatternFill("solid", fgColor=C_ANS_HD)
_FILL_H_LLM  = PatternFill("solid", fgColor=C_LLM_HD)

_THIN = Side(style="thin", color="D0D0D0")
_BORDER = Border(left=_THIN, right=_THIN, top=_THIN, bottom=_THIN)
_ALIGN_C = Alignment(horizontal="center", vertical="center", wrap_text=False)
_ALIGN_L = Alignment(horizontal="left",   vertical="center", wrap_text=False)
_FONT_WH = Font(bold=True, color="FFFFFF", size=10)
_FONT_NM = Font(size=10)
_FONT_BL = Font(bold=True, size=10)


def _col_letter(ws, col_name: str) -> Optional[str]:
    """헤더에서 컬럼명을 찾아 컬럼 문자(A, B, …)를 반환."""
    for cell in ws[1]:
        if cell.value == col_name:
            return get_column_letter(cell.column)
    return None


def style_main_sheet(
    ws,
    df: pd.DataFrame,
    flags: list[CellFlag],
    cmap: ColumnMap,
) -> None:
    """채점결과 시트 스타일 적용."""

    # 컬럼명 → 엑셀 열 번호 매핑 (헤더 행 기준)
    col_num: dict[str, int] = {}
    for cell in ws[1]:
        if cell.value is not None:
            col_num[str(cell.value)] = cell.column

    # 어떤 역할을 하는 컬럼인지 분류
    answer_col_set = set(cmap.answer_cols)
    llm_col_set: set[str] = set()
    for _, cols in cmap.llm_cols:
        llm_col_set.update(cols)

    # ── 헤더 행 스타일 ────────────────────────────────────────────────
    for cell in ws[1]:
        cname = str(cell.value or "")
        if cname in answer_col_set:
            cell.fill = _FILL_H_ANS
        elif cname in llm_col_set:
            cell.fill = _FILL_H_LLM
        else:
            cell.fill = _FILL_H_DEF
        cell.font      = _FONT_WH
        cell.alignment = _ALIGN_C
        cell.border    = _BORDER
    ws.row_dimensions[1].height = 22

    # ── 데이터 행 기본 스타일 ─────────────────────────────────────────
    product_col_num = col_num.get(cmap.product_col) if cmap.product_col else None
    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row:
            cell.font   = _FONT_NM
            cell.border = _BORDER
            cell.alignment = (
                _ALIGN_L if cell.column == product_col_num else _ALIGN_C
            )

    # ── 오답/에러 셀 마킹 ─────────────────────────────────────────────
    # flags 의 row 는 DataFrame 인덱스(0-base) → 엑셀 행 번호 = row + 2
    for flag in flags:
        excel_row = flag.row + 2   # 헤더(1) + 0-base → 2-base
        fill = _FILL_WRONG if flag.kind == "wrong" else _FILL_SKIP
        for col_name in flag.col_names:
            c_num = col_num.get(col_name)
            if c_num is None:
                continue
            cell = ws.cell(row=excel_row, column=c_num)
            cell.fill = fill
            if flag.kind == "wrong":
                cell.font = Font(bold=True, size=10, color="8B0045")

    # ── 열 너비 자동 조정 ─────────────────────────────────────────────
    for col_cells in ws.columns:
        letter = get_column_letter(col_cells[0].column)
        values = [str(c.value or "") for c in col_cells]
        max_len = max(len(v) for v in values) if values else 8
        ws.column_dimensions[letter].width = min(max_len * 1.25 + 2, 42)

    ws.freeze_panes = "A2"


def build_summary_df(stats: list[ModelStats], source_file: str) -> pd.DataFrame:
    """요약 DataFrame 생성."""
    rows = []
    for st in stats:
        rows.append({
            "모델":         st.name,
            "전체 행수":    st.total,
            "평가 행수":    st.evaluated,
            "에러 제외":    st.skipped,
            "정답":         st.correct,
            "오답":         st.wrong,
            "정확도(%)":    st.accuracy,
            "소스 파일":    pathlib.Path(source_file).name,
        })
    return pd.DataFrame(rows)


def style_summary_sheet(ws) -> None:
    """모델별요약 시트 스타일 적용."""
    for cell in ws[1]:
        cell.fill      = _FILL_H_DEF
        cell.font      = _FONT_WH
        cell.alignment = _ALIGN_C
        cell.border    = _BORDER
    ws.row_dimensions[1].height = 22

    acc_col = None
    for cell in ws[1]:
        if cell.value == "정확도(%)":
            acc_col = cell.column

    for row in ws.iter_rows(min_row=2, max_row=ws.max_row):
        for cell in row:
            cell.font      = _FONT_BL if cell.column == 1 else _FONT_NM
            cell.alignment = _ALIGN_C
            cell.border    = _BORDER

            if acc_col and cell.column == acc_col and cell.value is not None:
                try:
                    pct = float(cell.value)
                except (TypeError, ValueError):
                    continue
                if pct >= 80:
                    cell.fill = PatternFill("solid", fgColor=C_SUMMARY_GOOD)
                    cell.font = Font(bold=True, size=10, color="276221")
                elif pct >= 60:
                    cell.fill = PatternFill("solid", fgColor=C_SUMMARY_MID)
                    cell.font = Font(bold=True, size=10, color="7D6608")
                else:
                    cell.fill = PatternFill("solid", fgColor=C_SUMMARY_BAD)
                    cell.font = Font(bold=True, size=10, color="9C0006")

    for col_cells in ws.columns:
        letter = get_column_letter(col_cells[0].column)
        vals = [str(c.value or "") for c in col_cells]
        ws.column_dimensions[letter].width = max(len(v) for v in vals) * 1.4 + 4

    ws.freeze_panes = "A2"

# ════════════════════════════════════════════════════════════════════════
# 파일 I/O
# ════════════════════════════════════════════════════════════════════════

def load_file(path: pathlib.Path) -> pd.DataFrame:
    """CSV 또는 Excel 파일을 읽어 DataFrame 반환. 모든 컬럼은 문자열 처리."""
    suffix = path.suffix.lower()
    if suffix == ".csv":
        # 인코딩 자동 탐지 (UTF-8 → CP949 순)
        for enc in ("utf-8-sig", "utf-8", "cp949", "euc-kr"):
            try:
                return pd.read_csv(path, dtype=str, encoding=enc).fillna("")
            except (UnicodeDecodeError, ValueError):
                continue
        raise ValueError(f"CSV 인코딩을 인식할 수 없습니다: {path}")
    elif suffix in (".xlsx", ".xls"):
        return pd.read_excel(path, dtype=str).fillna("")
    else:
        raise ValueError(f"지원하지 않는 파일 형식입니다: {suffix}")


def process_one_file(
    input_path: pathlib.Path,
    writer: pd.ExcelWriter,
    sheet_prefix: str = "",
) -> list[ModelStats]:
    """
    단일 입력 파일을 처리하고 writer 에 시트를 추가한다.
    반환값: 이 파일에 대한 ModelStats 목록
    """
    print(f"\n{'─'*60}")
    print(f"📂 파일 로딩: {input_path.name}")

    df = load_file(input_path)
    print(f"  → {len(df):,}행 × {len(df.columns)}열 로딩 완료")

    # ── 컬럼 자동 감지 ────────────────────────────────────────────────
    cmap = detect_column_map(df)
    print(f"  → 컬럼 형식: {'3분류 (대/중/소)' if cmap.mode == '3level' else '단일 분류'}")
    print(f"  → 정답 컬럼: {cmap.answer_cols}")
    llm_names = [n for n, _ in cmap.llm_cols]
    print(f"  → LLM 모델 : {', '.join(llm_names)}")

    # ── 비교 실행 ─────────────────────────────────────────────────────
    print("📝 비교 중...")
    flags, stats = compare_dataframe(df, cmap)

    for st in stats:
        print(
            f"  [{st.name:8s}] 전체 {st.total:4d}행 | "
            f"에러 제외 {st.skipped:3d} | "
            f"정답 {st.correct:4d} | 오답 {st.wrong:4d} | "
            f"정확도 {st.accuracy:6.2f}%"
        )

    # ── Excel 시트 저장 ───────────────────────────────────────────────
    sheet_name = f"{sheet_prefix}채점결과" if sheet_prefix else "채점결과"
    sheet_name = sheet_name[:31]  # Excel 시트명 최대 31자
    df.to_excel(writer, sheet_name=sheet_name, index=False)

    # openpyxl 에서 시트를 꺼내 스타일 적용
    ws = writer.book[sheet_name]
    style_main_sheet(ws, df, flags, cmap)

    return stats


# ════════════════════════════════════════════════════════════════════════
# 범례 시트
# ════════════════════════════════════════════════════════════════════════

def add_legend_sheet(wb) -> None:
    """색상 범례 시트를 추가한다."""
    ws = wb.create_sheet("📖 범례")

    legend = [
        ("셀 배경색",         "의미",                      "비고"),
        ("─────────────",     "─────────────────────",     "───────────────────"),
        ("#FF89BF (분홍)",     "오답 — LLM 답변 ≠ 정답",   "직접 마킹 대상"),
        ("#D9D9D9 (회색)",     "에러 패스 — 비교 제외",    "코드 오류/Exception 등"),
        ("#C6EFCE (연초록)",   "정확도 80% 이상 (요약)",   "요약 시트 전용"),
        ("#FFEB9C (연노랑)",   "정확도 60–79% (요약)",     "요약 시트 전용"),
        ("#FFC7CE (연빨강)",   "정확도 60% 미만 (요약)",   "요약 시트 전용"),
        ("", "", ""),
        ("에러 패스 키워드", "", ""),
    ]
    for kw in ERROR_KEYWORDS:
        legend.append((f'  · "{kw}"', "", ""))
    legend.append(("  · 빈 셀, None, NaN, null 등", "", "완전 일치 시"))

    fills = {
        "#FF89BF (분홍)":   _FILL_WRONG,
        "#D9D9D9 (회색)":   _FILL_SKIP,
        "#C6EFCE (연초록)": PatternFill("solid", fgColor=C_SUMMARY_GOOD),
        "#FFEB9C (연노랑)": PatternFill("solid", fgColor=C_SUMMARY_MID),
        "#FFC7CE (연빨강)": PatternFill("solid", fgColor=C_SUMMARY_BAD),
    }

    for r_idx, row_vals in enumerate(legend, start=1):
        for c_idx, val in enumerate(row_vals, start=1):
            cell = ws.cell(row=r_idx, column=c_idx, value=val)
            cell.alignment = _ALIGN_L
            cell.font = _FONT_NM
            if r_idx == 1:
                cell.fill = _FILL_H_DEF
                cell.font = _FONT_WH
            elif val in fills:
                cell.fill = fills[val]

    for col in ["A", "B", "C"]:
        ws.column_dimensions[col].width = 32 if col == "A" else 28

# ════════════════════════════════════════════════════════════════════════
# main
# ════════════════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LLM 분류 정답 비교 및 오답 마킹 스크립트",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("-i", "--input",  nargs="+", help="입력 파일 경로 (xlsx/csv, 복수 지정 가능)")
    p.add_argument("-o", "--output", default=str(DEFAULT_OUTPUT), help="출력 xlsx 경로")
    p.add_argument("--all", dest="all_files", action="store_true",
                   help="data/master/ 내 기본 파일 전체 처리 (기본값)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ── 입력 파일 결정 ────────────────────────────────────────────────
    if args.input:
        inputs = [pathlib.Path(p) for p in args.input]
    else:
        inputs = [p for p in DEFAULT_INPUTS if p.exists()]
        if not inputs:
            print("❌ 처리할 파일이 없습니다.")
            print("   data/master/ 에 *_final.xlsx 파일이 있는지 확인하거나")
            print("   -i 옵션으로 입력 파일을 지정하세요.")
            sys.exit(1)

    # 존재 여부 검증
    for p in inputs:
        if not p.exists():
            print(f"❌ 파일 없음: {p}")
            sys.exit(1)

    output_path = pathlib.Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  LLM 오답 마킹 스크립트 시작")
    print(f"  처리 파일: {len(inputs)}개")
    print(f"  출력 경로: {output_path}")
    print("=" * 60)

    all_stats: list[ModelStats] = []

    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for idx, input_path in enumerate(inputs):
            prefix = f"{input_path.stem[:10]}_" if len(inputs) > 1 else ""
            stats = process_one_file(input_path, writer, sheet_prefix=prefix)
            all_stats.extend(stats)

        # ── 통합 요약 시트 ────────────────────────────────────────────
        print(f"\n{'─'*60}")
        print("📊 요약 시트 생성 중...")

        # 파일별 → 모델별로 stats 취합
        summary_rows: list[dict] = []
        for st in all_stats:
            summary_rows.append({
                "모델":       st.name,
                "전체 행수":  st.total,
                "평가 행수":  st.evaluated,
                "에러 제외":  st.skipped,
                "정답":       st.correct,
                "오답":       st.wrong,
                "정확도(%)":  st.accuracy,
            })
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_excel(writer, sheet_name="모델별요약", index=False)
        style_summary_sheet(writer.book["모델별요약"])

        # ── 범례 시트 ─────────────────────────────────────────────────
        add_legend_sheet(writer.book)

    print(f"\n✅ 완료! 결과 파일 저장됨:")
    print(f"   {output_path.resolve()}")
    print()

    # ── 콘솔 최종 요약 ────────────────────────────────────────────────
    print("┌" + "─" * 58 + "┐")
    print(f"│ {'모델':<10} {'평가행':>6} {'에러제외':>8} {'정답':>6} {'오답':>6} {'정확도':>8} │")
    print("├" + "─" * 58 + "┤")
    for st in all_stats:
        print(
            f"│ {st.name:<10} {st.evaluated:>6,} {st.skipped:>8,} "
            f"{st.correct:>6,} {st.wrong:>6,} {st.accuracy:>7.2f}% │"
        )
    print("└" + "─" * 58 + "┘")


if __name__ == "__main__":
    main()
