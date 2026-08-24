"""
common.py
ML 트랙(Hurdle Base Model 단독 / 최종 평가) 공통 유틸.

A센터 전용 잔차(Residual) 보정 스테이지는 제거됨 — Ablation Test 결과 in-sample과
OOF 두 버전 모두에서 잔차모델이 Base 단독보다 A_test 전 지표(RMSE/MAE/WAPE/MASE)를
악화시킴을 확인했고(잔차 신호 자체가 노이즈에 가까움), Hurdle Model(분류+조건부회귀)
자체가 Bias를 크게 줄여주는 효과가 이미 확인되어 A/B 공통으로 Base(Hurdle) 단독
구조로 단순화함. 스크립트들이 동일하게 필요로 하는 경로 상수, feature 준비, 모델
저장/로드 로직을 한 곳에 모아 중복을 없앤다. get_excluded_cols / get_base_model_excluded_cols는
Day2/Day3 feature_engineering 모듈의 정의를 그대로 재사용(재정의 금지 — Day3
docstring에 명시된 두 함수의 차이가 원래 1단계/2단계 모델 feature set을 가르던
기준이었으나, 2단계가 제거된 지금도 Base Model은 get_base_model_excluded_cols를
계속 사용함 — qty_lag2/4가 B에는 구조적으로 없어 A+B 통합 모델에는 여전히 제외 대상).
"""

from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[4]
SPLIT_DIR = BASE_DIR / "data" / "ml" / "splits"
MODEL_DIR = BASE_DIR / "data" / "ml" / "models"
FEATURE_TABLE_PATH = SPLIT_DIR / "feature_table_final.parquet"
WALKFORWARD_FOLDS_PATH = SPLIT_DIR / "B_walkforward_folds.json"

sys.path.insert(0, str(BASE_DIR / "src" / "forecasting" / "archive" / "feature_engineering"))
from feature_lag_rolling_calendar import get_excluded_cols  # noqa: E402
from feature_external_interaction_concat import get_base_model_excluded_cols  # noqa: E402

CENTER_COL = "center_id"
WEEK_COL = "week_st"
# target_h1(원본 qty, NaN 없음 - train/val 기준. test/pool의 NaN은 데이터셋 마지막 주라
# h주 뒤 관측치가 아직 없는 우측절단(right-censoring)일 뿐 "미입고"가 아님, 실측 확인됨)을
# 그대로 회귀 타겟으로 쓴다. 기존 target_h1_qty_log1p 컬럼은 재사용하지 않는다 —
# 그 컬럼은 sold_flag==1인 행만 값이 있고 진짜 0(판매 안 됨, train의 73%)을 전부 NaN
# 처리해둔 다른 용도의 컬럼이라, 그대로 회귀 타겟으로 쓰면 "수요 0인 주"가 통째로
# 학습에서 빠지는 심각한 편향이 생긴다. log1p 변환은 여기서 target_h1에 직접 적용한다
# (log1p(0)=0이라 진짜 0도 정상적으로 포함됨). target_h2/h4_qty_log1p도 동일한 이유로
# 사용 금지(Day1 Step1 audit, 2026-08-17 확정) — 향후 h2/h4 non-Hurdle 타겟이 필요해도
# 아래 LEGACY_LOG_TARGET_COLS 중 하나를 그대로 갖다 쓰지 말고 np.log1p(target_h{h})를
# 그 자리에서 계산할 것. 컬럼 자체는 과거 실험 재현성을 위해 삭제하지 않고 보존한다.
LEGACY_LOG_TARGET_COLS = ["target_h1_qty_log1p", "target_h2_qty_log1p", "target_h4_qty_log1p"]
TARGET_COL = "target_h1"
assert TARGET_COL not in LEGACY_LOG_TARGET_COLS, (
    f"{TARGET_COL}은 legacy Hurdle 조건부회귀 전용 컬럼(raw==0 -> NaN)이라 "
    "final non-Hurdle 회귀 타겟으로 쓸 수 없음 - raw target에서 np.log1p()를 runtime에 계산할 것."
)

BASE_CLS_MODEL_PATH = MODEL_DIR / "base_model_cls.pkl"
BASE_MODEL_PATH = MODEL_DIR / "base_model_reg.pkl"

# 센터별 분리 학습 Hurdle 모델(통합 모델과 별도 비교용)
A_BASE_CLS_PATH = MODEL_DIR / "a_base_cls.pkl"
A_BASE_REG_PATH = MODEL_DIR / "a_base_reg.pkl"
B_BASE_CLS_PATH = MODEL_DIR / "b_base_cls.pkl"
B_BASE_REG_PATH = MODEL_DIR / "b_base_reg.pkl"

# hard 모드 기본값으로 전환: 튜닝 과정에서 A val split 기준 WAPE를 최소화하는 threshold를
# 0.2~0.5 범위에서 탐색해 cls_bundle["threshold"]에 저장한다(모델별로 캘리브레이션이
# 다를 수 있어 전역 상수가 아니라 그 분류기 번들에 귀속시킴). HURDLE_THRESHOLD는 그
# 키가 없는 옛 번들을 위한 하위 호환 기본값일 뿐, 튜닝된 번들에서는 쓰이지 않는다.
HURDLE_MODE = "hard"
HURDLE_THRESHOLD = 0.5


def select_feature_cols(df: pd.DataFrame, excluded: list[str]) -> list[str]:
    """excluded 컬럼 + 'split' 수동 제외 (split은 target_*도 ID도 아니라
    get_excluded_cols류 함수가 걸러내지 않으므로 항상 여기서 별도로 뺀다)."""
    return [c for c in df.columns if c not in excluded and c != "split"]


def apply_target_date_boundary_filter(df: pd.DataFrame, horizon_weeks: int = 1) -> pd.DataFrame:
    """target_date(=week_st + horizon_weeks주) <= 그 센터 train 구간의 마지막 관측주
    (train_end)만 남긴다. train_end는 df 안의 split=='train' 최대 week_st로 센터별
    동적 계산(하드코딩 금지 — A/B train_end가 우연히 같은 값이라도 데이터에서 직접
    구하는 편이 안전). 이 필터가 없으면 train 구간 맨 마지막 주(target_date가
    val/test 쪽으로 넘어가는 주)의 라벨이 팀 컨벤션상 학습에 포함되면 안 되는데
    포함되는 경계 케이스가 생긴다."""
    train_end_by_center = df.loc[df["split"] == "train"].groupby(CENTER_COL)[WEEK_COL].max()
    target_date = df[WEEK_COL] + pd.Timedelta(weeks=horizon_weeks)
    keep = target_date <= df[CENTER_COL].map(train_end_by_center)
    before = len(df)
    out = df[keep]
    print(f"  [target_date<=train_end 경계 필터] {before:,} -> {len(out):,}행 ({before - len(out):,}행 제외)")
    return out


def predict_base_hurdle(df: pd.DataFrame, cls_bundle: dict, reg_bundle: dict,
                         mode: str = HURDLE_MODE, threshold: float | None = None):
    """Hurdle Model 추론: 1단계 분류 확률 >= threshold면 2단계 조건부회귀값 채택,
    아니면 0(hard, 기본). threshold 결정 우선순위: (1) 함수 인자로 명시적으로 받은
    threshold(스칼라, search_threshold_*.py처럼 전체에 동일 값을 스윕하며 비교할 때 씀)
    -> (2) cls_bundle["threshold_by_center"](딕셔너리, 센터별로 다른 threshold를 쓰고
    싶을 때 — search_threshold_extended.py가 A/B 각각의 val 기준 WAPE 최소화로 찾아
    저장함. df["center_id"]로 행마다 다른 threshold를 매핑) -> (3) cls_bundle["threshold"]
    (스칼라 폴백) -> (4) HURDLE_THRESHOLD(옛 bundle 하위호환용 최종 폴백). soft 모드
    (확률 x 조건부회귀값)는 threshold 없이 연속적인 기댓값을 원할 때 대안으로 남겨둔다."""
    Xc = prepare_X(df, cls_bundle["feature_cols"])
    prob = cls_bundle["model"].predict_proba(Xc)[:, 1]

    Xr = prepare_X(df, reg_bundle["feature_cols"])
    cond_qty = np.clip(np.expm1(reg_bundle["model"].predict(Xr)), 0, None)

    if mode == "soft":
        return prob * cond_qty
    if mode == "hard":
        if threshold is not None:
            th = threshold
        elif cls_bundle.get("threshold_by_center"):
            fallback = cls_bundle.get("threshold", HURDLE_THRESHOLD)
            th = df[CENTER_COL].map(cls_bundle["threshold_by_center"]).fillna(fallback).to_numpy()
        else:
            th = cls_bundle.get("threshold", HURDLE_THRESHOLD)
        return np.where(prob >= th, cond_qty, 0.0)
    raise ValueError(f"알 수 없는 mode: {mode}")


def prepare_X(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """object dtype 뿐 아니라 pandas의 신규 string dtype(str/string[pyarrow] 등,
    parquet 읽을 때 pyarrow 엔진이 텍스트 컬럼을 이렇게 추론하는 경우가 있음)도
    LightGBM이 못 받으므로 함께 category로 변환. 숫자/불리언 dtype만 그대로 둔다."""
    X = df[feature_cols].copy()
    for col in X.columns:
        if not (pd.api.types.is_numeric_dtype(X[col]) or pd.api.types.is_bool_dtype(X[col])):
            X[col] = X[col].astype("category")
    return X


def save_model_bundle(path: Path, model, feature_cols: list[str], **extra) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "feature_cols": feature_cols, **extra}
    joblib.dump(bundle, path)
    print(f"  저장 완료 -> {path}")


def load_model_bundle(path: Path) -> dict:
    return joblib.load(path)


MASE_EPS = 1e-5


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, naive_pred: np.ndarray) -> dict:
    """RMSE/MAE/WAPE/MASE/Bias(%) 5종을 한 번에 계산한다. 학습 단계의 threshold
    탐색(train_base_model.py)과 최종 평가(evaluate_pipeline.py)가 동일한 정의를
    공유해야 하므로 여기 한 곳에만 둔다.
    naive_pred: 같은 행 집합에 대한 1스텝 지연 단순 예측값(=qty, 해당 행 자기 주 실측)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    naive_pred = np.asarray(naive_pred, dtype=float)

    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    mae = float(np.mean(np.abs(y_true - y_pred)))
    denom = np.sum(np.abs(y_true))
    wape = float(np.sum(np.abs(y_true - y_pred)) / denom * 100) if denom > 0 else np.nan
    bias = float(np.sum(y_pred - y_true) / denom * 100) if denom > 0 else np.nan

    mae_naive = float(np.mean(np.abs(y_true - naive_pred)))
    mase = float(mae / (mae_naive if mae_naive > 0 else MASE_EPS))

    return {"RMSE": rmse, "MAE": mae, "WAPE": wape, "MASE": mase, "Bias(%)": bias}
