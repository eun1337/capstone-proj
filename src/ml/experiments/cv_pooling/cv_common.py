"""
cv_common.py
A/B 센터 CV 구조 개편 실험(기존 방식 vs Option1/2/3) 공용 로더/학습/평가 헬퍼.

기존 프로덕션 모듈(src/ml/models/common.py, train_base_model.py,
feature_external_interaction_concat.py)의 함수를 import해서 재사용하며,
그 파일들은 전혀 수정하지 않는다. 이 실험의 산출물은 전부
data/ml/cv_experiment/ 아래에만 쓴다(기존 data/ml/splits/, data/ml/models/는
read-only로만 접근).

반품(returns) 데이터는 이 실험 전체에서 사용하지 않는다 — 출고(총판매수량,
qty) 기준으로만 split/feature/학습이 진행된다(build_b_full_history.py 참고).
"""

from pathlib import Path
import json
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[4]
SPLIT_DIR = BASE_DIR / "data" / "ml" / "splits"
MODELS_DIR = BASE_DIR / "data" / "ml" / "models"
CV_DIR = BASE_DIR / "data" / "ml" / "cv_experiment"

sys.path.insert(0, str(BASE_DIR / "src" / "ml" / "models"))
sys.path.insert(0, str(BASE_DIR / "src" / "ml" / "feature_engineering"))

from common import (  # noqa: E402
    FEATURE_TABLE_PATH,
    TARGET_COL,
    WEEK_COL,
    CENTER_COL,
    compute_metrics,
    get_base_model_excluded_cols,
    load_model_bundle,
    predict_base_hurdle,
    prepare_X,
    save_model_bundle,
    select_feature_cols,
)
from feature_external_interaction_concat import (  # noqa: E402
    add_holiday_calendar_features,
    add_climate_extreme_flags,
    add_covid_flag,
    add_interaction_features,
    add_weeks_since_active_filled,
)

A_FEAT_FILES = [
    SPLIT_DIR / "A_train_feat.parquet",
    SPLIT_DIR / "A_val_feat.parquet",
    SPLIT_DIR / "A_test_feat.parquet",
]
B_FULL_FEAT_PATH = CV_DIR / "B_full_feat.parquet"
BEST_HYPERPARAMS_PATH = MODELS_DIR / "tuning" / "best_hyperparams.json"
# hpo_pure.py가 만드는, 2024를 전혀 참조하지 않은 하이퍼파라미터(A/B train 마지막
# 4주만 내부 val로 사용). use_pure=True로 지정한 호출만 이 경로를 쓴다.
PURE_HYPERPARAMS_PATH = CV_DIR / "best_hyperparams_pure.json"

N_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 30
INTERNAL_ES_WEEKS = 4  # 기존 B_INTERNAL_VAL_WEEKS 관례(train_center_base_models.py)를 그대로 재사용
RANDOM_STATE = 42
FIXED_PARAMS = dict(subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1)
# best_hyperparams.json 없을 때만 쓰는 폴백 (기존 train_center_base_models.py와 동일 값)
FALLBACK_LEARNING_RATE = 0.03
FALLBACK_NUM_LEAVES = 63


def load_a_full() -> pd.DataFrame:
    """A_train/val/test_feat.parquet(Day2 산출물, 프로덕션 read-only)을 이어붙여
    2021-01~2024-12 연속 타임라인을 복원한 뒤, Day3(공휴일/기후/covid/상호작용) feature를
    이 실험 전용으로 다시 적용한다. Day3 함수들은 (center_id, week_st) 단위로만
    동작하는 순수 함수라 A만 따로 넣어도 production의 feature_table_final.parquet A행과
    동일한 값이 나온다."""
    frames = [pd.read_parquet(p) for p in A_FEAT_FILES]
    df = pd.concat(frames, ignore_index=True).sort_values([CENTER_COL, "sku_id", WEEK_COL]).reset_index(drop=True)
    df = _apply_day3(df)
    return df


def load_b_full() -> pd.DataFrame:
    """build_b_full_history.py 산출물(pre+post 레짐 통합, 반품 미포함, Day2+Day3 완료)을 읽는다."""
    if not B_FULL_FEAT_PATH.exists():
        raise FileNotFoundError(f"{B_FULL_FEAT_PATH} 없음 — build_b_full_history.py를 먼저 실행해야 함")
    return pd.read_parquet(B_FULL_FEAT_PATH)


def _apply_day3(df: pd.DataFrame) -> pd.DataFrame:
    df = add_holiday_calendar_features(df)
    df = add_climate_extreme_flags(df)
    df = add_covid_flag(df)
    df = add_interaction_features(df)
    df = add_weeks_since_active_filled(df)
    df = df.drop(columns=["공휴일"], errors="ignore")
    return df


def slice_by_date(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df[(df[WEEK_COL] >= pd.Timestamp(start)) & (df[WEEK_COL] <= pd.Timestamp(end))].copy()


def apply_boundary_filter(df: pd.DataFrame, train_end: str, horizon_weeks: int = 1) -> pd.DataFrame:
    """target_date(=week_st+horizon주) <= train_end만 남긴다 (common.py의
    apply_target_date_boundary_filter와 동일 원칙 — fold의 train_end가 이미
    알려진 상수라 'split' 컬럼 없이 직접 계산)."""
    boundary = pd.Timestamp(train_end)
    target_date = df[WEEK_COL] + pd.Timedelta(weeks=horizon_weeks)
    keep = target_date <= boundary
    before = len(df)
    out = df[keep]
    print(f"    [target_date<={train_end} 경계 필터] {before:,} -> {len(out):,}행")
    return out


def carve_internal_es_val(df: pd.DataFrame, es_weeks: int = INTERNAL_ES_WEEKS) -> tuple[pd.DataFrame, pd.DataFrame]:
    """df(어떤 fold의 train 구간 행들, A 단독 또는 A+B pooled)에서 시간순 마지막 es_weeks
    주를 조기종료(early-stopping)용 내부 val로 떼어낸다. A/B가 같은 주간 캘린더를
    쓰므로 week_st 기준으로 한 번만 계산하면 pooled 데이터에도 그대로 적용된다."""
    weeks = sorted(df[WEEK_COL].unique())
    es_val_weeks = set(weeks[-es_weeks:])
    es_val = df[df[WEEK_COL].isin(es_val_weeks)]
    train_fit = df[~df[WEEK_COL].isin(es_val_weeks)]
    return train_fit, es_val


def load_tuned_params(key: str, use_pure: bool = False) -> dict | None:
    """튜닝 산출물이 있으면 해당 모델의 best_params 반환, 없으면 None.
    use_pure=True면 hpo_pure.py가 만든 PURE_HYPERPARAMS_PATH(2024를 전혀 참조하지
    않은 버전)를 읽는다 — 기본값(False)은 기존 프로덕션 BEST_HYPERPARAMS_PATH."""
    path = PURE_HYPERPARAMS_PATH if use_pure else BEST_HYPERPARAMS_PATH
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data.get(key, {}).get("best_params")


def _b_sample_weight(df: pd.DataFrame, b_weight: float | None) -> np.ndarray | None:
    if b_weight is None:
        return None
    return np.where(df[CENTER_COL].to_numpy() == "B", b_weight, 1.0)


def train_hurdle(train_df: pd.DataFrame, es_val_df: pd.DataFrame, feature_cols: list[str],
                  label: str, b_weight: float | None = None, use_pure: bool = False) -> tuple[dict, dict]:
    """Hurdle(분류기+조건부회귀기) 1회 학습. train_base_model.py/train_center_base_models.py와
    동일한 패턴(2단계 LightGBM, early stopping) — best_hyperparams.json이 있으면 그
    하이퍼파라미터를 쓰고 없으면 기존 폴백 그리드 값(lr=0.03/leaves=63)을 쓴다.
    use_pure=True면 hpo_pure.py가 만든 2024 미참조 하이퍼파라미터를 대신 쓴다.
    (폴드마다 재튜닝은 하지 않음 — 실험 범위 밖, plan에 명시)"""
    tuned_cls = load_tuned_params("classifier", use_pure=use_pure)
    tuned_reg = load_tuned_params("regressor", use_pure=use_pure)

    y_cls_train = (train_df[TARGET_COL] > 0).astype(int)
    y_cls_val = (es_val_df[TARGET_COL] > 0).astype(int)
    n_pos, n_neg = int((y_cls_train == 1).sum()), int((y_cls_train == 0).sum())
    natural_spw = n_neg / max(n_pos, 1)

    X_cls_train = prepare_X(train_df, feature_cols)
    X_cls_val = prepare_X(es_val_df, feature_cols)
    sw_cls_train = _b_sample_weight(train_df, b_weight)

    if tuned_cls is not None:
        scale_pos_weight = natural_spw * tuned_cls["scale_pos_weight_multiplier"]
        lgb_params = {k: tuned_cls[k] for k in
                      ["num_leaves", "max_depth", "learning_rate", "min_child_samples",
                       "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]}
    else:
        scale_pos_weight = natural_spw
        lgb_params = dict(learning_rate=FALLBACK_LEARNING_RATE, num_leaves=FALLBACK_NUM_LEAVES,
                           **FIXED_PARAMS)
    cls_model = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS, scale_pos_weight=scale_pos_weight, subsample_freq=1,
        verbose=-1, random_state=RANDOM_STATE, n_jobs=-1, **lgb_params,
    )
    cls_model.fit(
        X_cls_train, y_cls_train, sample_weight=sw_cls_train, eval_set=[(X_cls_val, y_cls_val)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )
    print(f"    [{label}] 분류기 best_iter={cls_model.best_iteration_} scale_pos_weight={scale_pos_weight:.3f}")

    pos_train = train_df[train_df[TARGET_COL] > 0]
    pos_val = es_val_df[es_val_df[TARGET_COL] > 0]
    X_reg_train = prepare_X(pos_train, feature_cols)
    y_reg_train = np.log1p(pos_train[TARGET_COL])
    X_reg_val = prepare_X(pos_val, feature_cols)
    y_reg_val = np.log1p(pos_val[TARGET_COL])
    sw_reg_train = _b_sample_weight(pos_train, b_weight)

    if tuned_reg is not None:
        lgb_params = {k: tuned_reg[k] for k in
                      ["num_leaves", "max_depth", "learning_rate", "min_child_samples",
                       "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]}
    else:
        lgb_params = dict(learning_rate=FALLBACK_LEARNING_RATE, num_leaves=FALLBACK_NUM_LEAVES,
                           **FIXED_PARAMS)
    reg_model = lgb.LGBMRegressor(
        n_estimators=N_ESTIMATORS, objective="regression", subsample_freq=1,
        verbose=-1, random_state=RANDOM_STATE, n_jobs=-1, **lgb_params,
    )
    reg_model.fit(
        X_reg_train, y_reg_train, sample_weight=sw_reg_train, eval_set=[(X_reg_val, y_reg_val)],
        callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
    )
    print(f"    [{label}] 회귀기 best_iter={reg_model.best_iteration_}")

    cls_bundle = {"model": cls_model, "feature_cols": feature_cols}
    reg_bundle = {"model": reg_model, "feature_cols": feature_cols}
    return cls_bundle, reg_bundle


def compute_mape(y_true: np.ndarray, y_pred: np.ndarray, zero_handling: str = "exclude",
                  eps: float = 1e-8) -> tuple[float, int, int]:
    """MAPE = mean(|y_true-y_pred| / |y_true|) * 100. 수요 데이터가 희소해(약 70%+가
    실제 0) actual==0인 행이 매우 많으므로 zero_handling으로 처리 방식을 명시한다:
      - "exclude"(기본, 권장): actual==0 행을 MAPE 분모 계산에서 제외한다. "실제로
        수요가 있었던 주에 수량을 얼마나 정확히 맞췄는가"를 보는 지표가 되어
        WAPE(전체 물량 기준 오차)와 다른 관점을 준다. epsilon 방식보다 안전한 이유:
        eps를 분모로 쓰면 actual=0에 대한 항이 |pred|/eps 형태로 사실상 무한대에
        가까운 값이 되어(pred가 조금만 0이 아니어도) MAPE 전체가 그 소수의 행에
        완전히 지배당해 지표로서 의미가 없어진다.
      - "epsilon": actual==0 행도 포함하되 분모를 max(|y_true|, eps)로 클립 — 위
        문제를 그대로 안고 가는 옵션이라 참고용으로만 남겨둠.
    반환: (MAPE(%), 사용된 행수, 제외된 행수)"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if zero_handling == "exclude":
        mask = y_true != 0
        n_excluded = int((~mask).sum())
        yt, yp = y_true[mask], y_pred[mask]
        denom = np.abs(yt)
    elif zero_handling == "epsilon":
        n_excluded = 0
        yt, yp = y_true, y_pred
        denom = np.clip(np.abs(yt), eps, None)
    else:
        raise ValueError(f"알 수 없는 zero_handling: {zero_handling}")
    if len(yt) == 0:
        return np.nan, 0, n_excluded
    mape = float(np.mean(np.abs(yt - yp) / denom) * 100)
    return mape, len(yt), n_excluded


def predict_and_score(df: pd.DataFrame, cls_bundle: dict, reg_bundle: dict,
                       mape_zero_handling: str = "exclude") -> dict:
    """soft 모드(P(수요>0) x E[수량|수요>0])로 통일해서 평가 — 프로덕션 공식 기준점과
    동일 방식(evaluate_pipeline.py 기본값)이라 폴드마다 threshold 탐색을 추가하지 않는
    의도적 단순화. RMSE/MAE/WAPE/MASE/Bias(common.compute_metrics)에 더해 MAPE도
    함께 계산한다(actual==0 행 처리는 compute_mape 참고)."""
    valid = df[TARGET_COL].notna()
    d = df[valid]
    if len(d) == 0:
        return {"n": 0, "RMSE": np.nan, "MAE": np.nan, "WAPE": np.nan, "MASE": np.nan,
                "Bias(%)": np.nan, "MAPE": np.nan, "n_mape": 0, "n_mape_excluded_zero": 0}
    y_true = d[TARGET_COL].to_numpy()
    y_pred = predict_base_hurdle(d, cls_bundle, reg_bundle, mode="soft")
    naive_pred = d["qty"].to_numpy()
    m = compute_metrics(y_true, y_pred, naive_pred)
    mape, n_mape, n_excluded = compute_mape(y_true, y_pred, zero_handling=mape_zero_handling)
    m["MAPE"] = mape
    return {"n": len(d), **m, "n_mape": n_mape, "n_mape_excluded_zero": n_excluded}


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    return select_feature_cols(df, get_base_model_excluded_cols(df))
