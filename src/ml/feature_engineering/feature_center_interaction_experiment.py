"""
feature_center_interaction_experiment.py
ML 트랙 Feature Engineering 실험: center_id × 연속형/이진형 변수 명시적 상호작용 피처 검증

배경: feature_external_interaction_concat.py(Day3, 프로덕션)는 "qty_lag1 * center_is_B"
형태의 명시적 곱셈 상호작용을 의도적으로 만들지 않기로 결정해뒀음(B센터 26주 데이터의
과적합 위험, 해당 스크립트 docstring 3번 항목 참고) — 단 "Day4 ML 성능 검증에서 B 오차가
크게 나오면 그때 재검토"라는 조건을 남겨뒀고, 실제로 B 회귀 WAPE가 61%(실제 수요>0 행
기준)로 크게 나온 상태를 확인해 재검토 조건이 충족되어 이 실험을 진행한다(팀 협의 완료).

핵심 결정 (팀 협의 완료, 재론 불필요):
    1) center 상호작용을 만드는 대상은 qty_lag1(판매 관성)과 평균온도(기온 편차) 2종뿐이다.
       공휴일_W0/W-1/W+1은 센터 구분 없이 원본 그대로 3개를 개별 피처로만 쓴다(center_is_B와
       곱한 상호작용 컬럼을 만들지 않음 — 이미 0/1 이진값이라 center_is_B와의 곱이 정보량을
       거의 더해주지 못한다는 판단, 팀 협의로 범위 축소함). 최초 스펙의 holiday_flag/
       temp_anomaly는 실제 데이터에 없는 컬럼명이라 공휴일_W0(+W-1/W+1)/평균온도로 교체함.
    2) center 인코딩은 원본 center_id(문자열/category)가 아니라 기존 파이프라인이 이미
       쓰는 center_is_B(0/1 수치형)를 재사용 — 새로 정의하지 않고 그대로 곱셈에 사용.
    3) 상호작용 생성 방식은 문자열 결합이 아니라 수치형 곱셈(center_is_B * 변수) —
       연속형 변수를 문자열로 결합하면 카디널리티가 폭발해 LightGBM 카테고리 처리에
       부적합하기 때문.
    4) 이 스크립트는 프로덕션 feature_table_final.parquet을 직접 덮어쓰지 않는다 — 상호작용
       피처가 실제로 성능을 개선하는지 먼저 ablation으로 검증한 뒤, 프로덕션
       feature_external_interaction_concat.py 반영 여부를 별도로 결정한다. 그래서 결과물은
       data/ml/feature_engineering_experiment/에 별도 저장한다.
    5) NaN 처리: 결측치는 인위적으로 채우지 않고 그대로 둔다(LightGBM 네이티브 결측 처리
       활용). qty_lag2/4처럼 한쪽 센터에서 구조적으로 전부 NaN이 되는 피처는 프로덕션이
       이미 학습 대상에서 제외하고 있으므로(get_base_model_excluded_cols), 이 스크립트가
       새로 추가하는 상호작용 피처도 동일 위험이 있는지 출력 검증 단계에서 센터별로 확인한다.
    6) center_id를 실제 학습 피처셋에서 배제하는 것(get_excluded_cols() 적용)과 LightGBM
       categorical 지정은 이 스크립트의 책임이 아니다 — 기존 파이프라인 관행과 동일하게
       feature-engineering 단계는 피처를 만들기만 하고, 배제/dtype 변환은 학습 스크립트
       (train_base_model.py, common.py의 select_feature_cols/prepare_X)가 담당한다.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

BASE_DIR = Path(__file__).resolve().parents[3]
FEATURE_TABLE_PATH = BASE_DIR / "data" / "ml" / "splits" / "feature_table_final.parquet"
OUT_DIR = BASE_DIR / "data" / "ml" / "feature_engineering_experiment"
OUT_PATH = OUT_DIR / "feature_table_with_center_interactions.parquet"

CENTER_COL = "center_id"
# 공휴일_W0/W-1/W+1은 상호작용 없이 원본 그대로 쓰는 참고용 컬럼이라 상호작용 대상에서 뺌.
INTERACTION_BASE_COLS = ["qty_lag1", "평균온도"]
HOLIDAY_COLS = ["공휴일_W0", "공휴일_W-1", "공휴일_W+1"]
NEW_INTERACTION_COLS = ["center_qty_lag1_inter", "center_temp_inter"]


# ----------------------------------------------------------------------
# 1) 환경 설정 및 라이브러리 임포트는 위에서 완료(pandas/numpy). pyarrow는
#    pandas.read_parquet/to_parquet가 내부적으로 사용(별도 명시 호출 불필요).
# ----------------------------------------------------------------------


def load_feature_table() -> pd.DataFrame:
    """P1: 프로덕션 feature_table_final.parquet 로드(읽기 전용, 이 스크립트는 절대 덮어쓰지 않음)."""
    df = pd.read_parquet(FEATURE_TABLE_PATH)
    print(f"[로드 완료] {FEATURE_TABLE_PATH}  shape={df.shape}")
    return df


def validate_input(df: pd.DataFrame) -> None:
    """로드 직후 기본 검증: shape, 상호작용 대상 컬럼 + 공휴일 컬럼 존재 여부, 결측치 비율, center_id 분포."""
    print("=" * 80)
    print("[입력 검증] 상호작용 대상 컬럼 및 공휴일 컬럼 존재 여부, 결측치 비율")
    missing_cols = [c for c in INTERACTION_BASE_COLS + HOLIDAY_COLS if c not in df.columns]
    if missing_cols:
        raise KeyError(f"필요 컬럼이 존재하지 않음: {missing_cols}")
    for col in INTERACTION_BASE_COLS + HOLIDAY_COLS:
        na_ratio = df[col].isna().mean()
        print(f"  {col:12s} 결측치 비율={na_ratio:.2%}")
    print(f"  center_id dtype={df[CENTER_COL].dtype}")
    print(f"  center_id 분포:\n{df[CENTER_COL].value_counts().to_string()}")


def add_center_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """center_id × 연속형 변수 명시적 곱셈 상호작용 피처 생성(팀 확정 방식, qty_lag1/평균온도
    2종만 대상). 공휴일_W0/W-1/W+1은 상호작용 없이 원본 그대로 유지(센터 구분 없이 개별
    피처로만 사용 — 팀 협의로 확정).
    center_is_B가 이미 있으면 재사용, 없으면(원본 feature_table_final.parquet에 바로
    적용하는 이번 케이스) 여기서 새로 만든다 — feature_external_interaction_concat.py의
    정의(df[CENTER_COL] == "B")와 동일하게 맞춤(재정의가 아니라 동일 로직 재사용)."""
    df = df.copy()
    if "center_is_B" not in df.columns:
        df["center_is_B"] = (df[CENTER_COL] == "B").astype(int)

    df["center_qty_lag1_inter"] = df["center_is_B"] * df["qty_lag1"]
    df["center_temp_inter"] = df["center_is_B"] * df["평균온도"]
    return df


def get_categorical_feature_cols(df: pd.DataFrame) -> list[str]:
    """LightGBM categorical_feature 인자에 넘길 컬럼 리스트. 원본 center_id는 제외한다
    (기존 프로덕션 설계와 동일 — get_excluded_cols가 center_id를 ID_COLS로 이미 제외하고
    대신 수치형 center_is_B를 피처로 씀, categorical_feature=['center_id']로 지정하지 않음)."""
    categorical_cols = [
        c for c in df.columns
        if df[c].dtype.name in ("object", "category") and c != CENTER_COL
    ]
    print(f"[범주형 컬럼] center_id 제외, LightGBM categorical_feature 후보({len(categorical_cols)}개): {categorical_cols}")
    return categorical_cols


def validate_output(df: pd.DataFrame) -> None:
    """P5 검증: 신규 상호작용 피처가 인위적으로 채워지지 않았는지, 한쪽 센터에서만
    구조적으로 전부 NaN이 되는 결함이 없는지 확인(qty_lag2/4가 그래서 제외됐던 것과
    같은 문제가 새 피처에도 있는지 미리 체크)."""
    print("=" * 80)
    print("[출력 검증] 신규 상호작용 피처 결측치 비율(전체 및 센터별)")
    centers = df[CENTER_COL].cat.categories if hasattr(df[CENTER_COL], "cat") else df[CENTER_COL].unique()
    for col in NEW_INTERACTION_COLS:
        na_ratio = df[col].isna().mean()
        print(f"  {col}: 전체 결측치 비율={na_ratio:.2%}")
        for center in centers:
            sub_na = df.loc[df[CENTER_COL] == center, col].isna().mean()
            flag = "  ⚠ 구조적 전부-NaN 의심 — 학습 피처 제외 검토 필요" if sub_na >= 0.999 else ""
            print(f"    센터 {center}: {sub_na:.2%}{flag}")


def main():
    df = load_feature_table()
    validate_input(df)

    df = add_center_interaction_features(df)
    categorical_cols = get_categorical_feature_cols(df)

    validate_output(df)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PATH, index=False)
    print("=" * 80)
    print(f"[저장 완료] {OUT_PATH}  shape={df.shape}")
    print("  (프로덕션 data/ml/splits/feature_table_final.parquet은 건드리지 않음)")

    # ------------------------------------------------------------------
    # 다음 단계(LightGBM 학습) 연계 가이드 — center_id 배제/categorical 지정은 이 스크립트가
    # 아니라 학습 단계(train_base_model.py 패턴)에서 처리한다:
    #   - center_id 배제: select_feature_cols(df, get_base_model_excluded_cols(df))를 호출하면
    #     ID_COLS에 포함된 center_id가 feature_cols에서 자동으로 빠진다(직접 컬럼을
    #     drop할 필요 없음 — get_excluded_cols()/get_base_model_excluded_cols() 재사용).
    #   - categorical 지정: 이 프로젝트는 LGBMClassifier/Regressor.fit()에 categorical_feature=
    #     [...] 인자를 명시적으로 넘기지 않는다. 대신 common.py의 prepare_X()가 feature_cols
    #     중 숫자/불리언이 아닌 컬럼(object/문자열 등)을 전부 pandas category dtype으로
    #     변환하고, LightGBM이 category dtype 컬럼을 자동으로 범주형으로 인식한다. 즉 위
    #     get_categorical_feature_cols()가 반환한 리스트는 참고용일 뿐, 실제로는
    #     select_feature_cols + prepare_X만 그대로 재사용하면 된다(별도 지정 불필요 —
    #     "나중에"가 아니라 "이미 기존 함수가 자동으로 처리").
    #   - 회귀 타겟: target_h1(원본)에 그 자리에서 log1p 적용해서 쓸 것.
    #     target_h1_qty_log1p 컬럼은 sold_flag==1(현재 주 기준)인 행만 값이 있고
    #     나머지가 NaN이라 그대로 쓰면 수요 0인 행(train의 대다수)이 학습에서 빠지는
    #     심각한 편향이 생김(common.py docstring 참고) — 재사용 금지.
    #   - 분류 타겟: (target_h1 > 0).astype(int). sold_flag는 "현재 주" 기준이라
    #     h1주 뒤 미래를 보는 이 타겟과 다르므로 착각하지 말 것.
    #   - 이 산출물은 아직 프로덕션 미반영 상태 — 상호작용 피처 있음/없음 ablation으로
    #     실제 WAPE/Bias 개선 여부를 확인한 뒤 feature_external_interaction_concat.py에
    #     반영할지 결정할 것.
    # ------------------------------------------------------------------


if __name__ == "__main__":
    main()
