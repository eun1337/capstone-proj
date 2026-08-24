"""
tweedie_experiment.py
Tweedie 단일모델 대조실험을 "2024 미유출" 정책에 맞게 재구성. 프로덕션
train_tweedie_model.py/search_tweedie_cutoff.py는 variance_power 선택과 cutoff
탐색 모두에 A `split=='val'`(2024-01~06)을 직접 사용한다(train_tweedie_model.py:85-145,
search_tweedie_cutoff.py:53-68) — 이 스크립트는 그 지점만 "pooled train(2021~2023
A + 2023-07~12 B)의 시간순 마지막 4주 내부 val"로 교체하고, 나머지 로직(feature
구성, tuned params 재사용 관례, cutoff 그리드, 최종 홀드아웃 방식)은 프로덕션과
동일하게 유지한다. 프로덕션 파일은 수정하지 않으며, 새로 학습한 모델은 별도
경로(data/ml/cv_experiment/models/tweedie_no_leak_reg.pkl)에 저장해 기존
tweedie_reg.pkl을 덮어쓰지 않는다.

python tweedie_experiment.py

핵심 설계:
    1) 학습 데이터: A는 cv_common.load_a_full()의 2021~2023(a_final_cv와 동일 소스/
       범위), B는 production feature_table_final.parquet의 B/train(2023-07~12,
       "B 학습체계 유지" 원칙 그대로 post 레짐만) — 둘을 pooled해서 A+B 통합
       Tweedie를 학습한다(production과 동일 컨셉).
    2) early-stopping + variance_power 선택: pooled train의 시간순 마지막 4주를
       내부 val로 분리(cv_common.carve_internal_es_val) — A val(2024H1) 미사용.
    3) tuned_hyperparams.json의 "tweedie" 키가 있으면 그 설정(8종 파라미터 +
       tweedie_variance_power + b_center_weight)을 그대로 써서 단일 학습, 없으면
       VARIANCE_POWER_GRID=[1.3,1.5,1.7] 스윕으로 폴백 — production과 동일 관례.
    4) cutoff 탐색도 같은 내부 val(A/B 각각 자기 쪽 마지막 4주)로 다시 계산 —
       production의 CUTOFF_GRID를 그대로 재사용.
    5) 최종 홀드아웃(1회, 재탐색 금지): A=2024 전체(val+test 통합, 새 정책),
       B=53-fold walk-forward. Tweedie(raw/cutoff)뿐 아니라 Hurdle Soft
       기준선(A: a_final_cv, B: production base_model)도 같은 표에 병기해
       "이 복잡도가 필요한가"라는 원 실험 취지를 유지한다. WAPE/RMSE/MAE/MAPE
       4개 지표 전부 산출.
"""

from pathlib import Path
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as cc  # noqa: E402
import folds  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(BASE_DIR / "src" / "forecasting" / "archive" / "models_hurdle"))
from common import (  # noqa: E402
    BASE_CLS_MODEL_PATH,
    BASE_MODEL_PATH,
    CENTER_COL,
    FEATURE_TABLE_PATH,
    TARGET_COL,
    WEEK_COL,
    compute_metrics,
    prepare_X,
)
from search_tweedie_cutoff import CUTOFF_GRID  # noqa: E402

MODEL_OUT_DIR = cc.CV_DIR / "models"
REPORT_DIR = cc.CV_DIR / "reports"
TWEEDIE_PATH = MODEL_OUT_DIR / "tweedie_no_leak_reg.pkl"

VARIANCE_POWER_GRID = [1.3, 1.5, 1.7]
LEARNING_RATE = 0.03
NUM_LEAVES = 63
N_ESTIMATORS = 1000
EARLY_STOPPING_ROUNDS = 30
RANDOM_STATE = 42
FIXED_PARAMS = dict(subsample=0.8, colsample_bytree=0.8, random_state=RANDOM_STATE, n_jobs=-1)

A_TRAIN_START = folds.A_FINAL_TRAIN_START
A_TRAIN_END = folds.A_FINAL_TRAIN_END

# hpo_pure.py가 만든 2024 완전 미유출 하이퍼파라미터를 쓴다(best_hyperparams.json
# 대신 best_hyperparams_pure.json) — "2024는 어떤 결정에도 유출되지 않는다"는
# 원칙을 하이퍼파라미터 선택 단계까지 완전히 지키기 위함.
USE_PURE_HYPERPARAMS = True


def load_pooled_train() -> pd.DataFrame:
    a_full = cc.load_a_full()
    a_train = cc.slice_by_date(a_full, A_TRAIN_START, A_TRAIN_END)

    prod = pd.read_parquet(FEATURE_TABLE_PATH)
    b_train = prod[(prod[CENTER_COL] == "B") & (prod["split"] == "train")].copy()

    pooled = pd.concat([a_train, b_train], ignore_index=True)
    pooled = cc.apply_boundary_filter(pooled, A_TRAIN_END, horizon_weeks=1)
    pooled = pooled[pooled[TARGET_COL].notna()]
    print(f"[Tweedie] pooled train(A 2021~2023 + B 2023-07~12): {len(pooled):,}행 "
          f"(A={int((pooled[CENTER_COL]=='A').sum()):,} / B={int((pooled[CENTER_COL]=='B').sum()):,})")
    return pooled


def train_tweedie(pooled: pd.DataFrame):
    train_fit, es_val = cc.carve_internal_es_val(pooled)
    print(f"[Tweedie] 내부 val(pooled 마지막 4주): {es_val[WEEK_COL].min().date()} ~ "
          f"{es_val[WEEK_COL].max().date()}, {len(es_val):,}행 / 학습: {len(train_fit):,}행")

    feature_cols = cc.get_feature_cols(train_fit)
    X_train = prepare_X(train_fit, feature_cols)
    y_train = train_fit[TARGET_COL]
    X_val = prepare_X(es_val, feature_cols)
    y_val = es_val[TARGET_COL]
    naive_val = es_val["qty"].to_numpy()

    tuned = cc.load_tuned_params("tweedie", use_pure=USE_PURE_HYPERPARAMS)

    if tuned is not None:
        src = "best_hyperparams_pure.json" if USE_PURE_HYPERPARAMS else "best_hyperparams.json"
        print(f"[Tweedie] {src}(tweedie) 사용: {tuned}")
        sw_train = np.where(train_fit[CENTER_COL].to_numpy() == "B", tuned["b_center_weight"], 1.0)
        lgb_params = {k: tuned[k] for k in
                      ["num_leaves", "max_depth", "learning_rate", "min_child_samples",
                       "subsample", "colsample_bytree", "reg_alpha", "reg_lambda"]}
        best_vp = tuned["tweedie_variance_power"]
        model = lgb.LGBMRegressor(
            objective="tweedie", tweedie_variance_power=best_vp, n_estimators=N_ESTIMATORS,
            subsample_freq=1, verbose=-1, random_state=RANDOM_STATE, n_jobs=-1, **lgb_params,
        )
        model.fit(
            X_train, y_train, sample_weight=sw_train, eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
        )
        pred_val = np.clip(model.predict(X_val), 0, None)
        m = compute_metrics(y_val.to_numpy(), pred_val, naive_val)
        print(f"  best_iter={model.best_iteration_}  내부val WAPE={m['WAPE']:.3f}%  Bias={m['Bias(%)']:+.3f}%")
    else:
        print("[Tweedie] best_hyperparams.json 없음 -> variance_power 스윕 폴백")
        best = None
        for vp in VARIANCE_POWER_GRID:
            cand = lgb.LGBMRegressor(
                objective="tweedie", tweedie_variance_power=vp, n_estimators=N_ESTIMATORS,
                learning_rate=LEARNING_RATE, num_leaves=NUM_LEAVES, verbose=-1, **FIXED_PARAMS,
            )
            cand.fit(
                X_train, y_train, eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(EARLY_STOPPING_ROUNDS, verbose=False), lgb.log_evaluation(0)],
            )
            pred_val = np.clip(cand.predict(X_val), 0, None)
            m = compute_metrics(y_val.to_numpy(), pred_val, naive_val)
            print(f"  [vp={vp}] best_iter={cand.best_iteration_}  내부val WAPE={m['WAPE']:.3f}%")
            if best is None or m["WAPE"] < best[1]:
                best = (cand, m["WAPE"], vp)
        model, _, best_vp = best

    MODEL_OUT_DIR.mkdir(parents=True, exist_ok=True)
    cc.save_model_bundle(TWEEDIE_PATH, model, feature_cols, variance_power=best_vp, tuned_params=tuned)
    return model, feature_cols


def internal_cutoff_val(pooled: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """cutoff 탐색용 내부 val — A/B 각각 자기 쪽 pooled train의 마지막 4주
    (production처럼 A val/B train-tail을 따로 쓰지 않고, 학습에 쓴 것과 동일한
    pooled 내부 val을 센터별로 나눠서 재사용 — 이미 2024를 안 보므로 안전)."""
    _, es_val = cc.carve_internal_es_val(pooled)
    a_val = es_val[es_val[CENTER_COL] == "A"]
    b_val = es_val[es_val[CENTER_COL] == "B"]
    return a_val, b_val


def sweep_cutoff(y_true, raw_pred, naive_pred, label) -> tuple[pd.DataFrame, float]:
    rows = []
    for c in CUTOFF_GRID:
        pred = np.where(raw_pred < c, 0.0, raw_pred)
        m = compute_metrics(y_true, pred, naive_pred)
        rows.append({"cutoff": c, **{k: round(v, 3) for k, v in m.items()}})
    table = pd.DataFrame(rows)
    best = table.loc[table["WAPE"].idxmin()]
    print(f"  [{label}] 최적 cutoff={best['cutoff']:.2f} (WAPE={best['WAPE']:.3f}%, Bias={best['Bias(%)']:+.3f}%)")
    return table, float(best["cutoff"])


def final_holdout(model, feature_cols, best_cutoff_a: float, best_cutoff_b: float) -> pd.DataFrame:
    a_full = cc.load_a_full()
    a_2024 = cc.slice_by_date(a_full, folds.FINAL_TEST_START, folds.FINAL_TEST_END)
    a_2024 = a_2024[a_2024[TARGET_COL].notna()]
    Xa = prepare_X(a_2024, feature_cols)
    pred_a_raw = np.clip(model.predict(Xa), 0, None)
    pred_a_cut = np.where(pred_a_raw < best_cutoff_a, 0.0, pred_a_raw)
    y_a = a_2024[TARGET_COL].to_numpy()
    naive_a = a_2024["qty"].to_numpy()

    prod = pd.read_parquet(FEATURE_TABLE_PATH)
    b_pool = prod[(prod[CENTER_COL] == "B") & (prod["split"] == "pool")].copy()
    b_pool = b_pool[b_pool[TARGET_COL].notna()]
    Xb = prepare_X(b_pool, feature_cols)
    pred_b_raw = np.clip(model.predict(Xb), 0, None)
    pred_b_cut = np.where(pred_b_raw < best_cutoff_b, 0.0, pred_b_raw)
    y_b = b_pool[TARGET_COL].to_numpy()
    naive_b = b_pool["qty"].to_numpy()

    rows = []
    for center, y, naive, raw, cut, n in [
        ("A", y_a, naive_a, pred_a_raw, pred_a_cut, len(a_2024)),
        ("B", y_b, naive_b, pred_b_raw, pred_b_cut, len(b_pool)),
    ]:
        for variant, pred in [("Tweedie(raw)", raw), ("Tweedie+cutoff", cut)]:
            m = compute_metrics(y, pred, naive)
            mape, n_mape, n_ex = cc.compute_mape(y, pred)
            rows.append({"center": center, "variant": variant, "n": n,
                         "WAPE": round(m["WAPE"], 3), "MAE": round(m["MAE"], 3),
                         "RMSE": round(m["RMSE"], 3), "MAPE": round(mape, 3)})

    # Hurdle Soft 기준선 병기 (a_final_cv / production base_model, 재학습 없음)
    a_cls = cc.load_model_bundle(MODEL_OUT_DIR / "a_final_cv_cls.pkl")
    a_reg = cc.load_model_bundle(MODEL_OUT_DIR / "a_final_cv_reg.pkl")
    m_a_hurdle = cc.predict_and_score(a_2024, a_cls, a_reg)
    rows.append({"center": "A", "variant": "Hurdle Soft(기준선)", "n": m_a_hurdle["n"],
                 "WAPE": round(m_a_hurdle["WAPE"], 3), "MAE": round(m_a_hurdle["MAE"], 3),
                 "RMSE": round(m_a_hurdle["RMSE"], 3), "MAPE": round(m_a_hurdle["MAPE"], 3)})

    b_cls = cc.load_model_bundle(BASE_CLS_MODEL_PATH)
    b_reg = cc.load_model_bundle(BASE_MODEL_PATH)
    m_b_hurdle = cc.predict_and_score(b_pool, b_cls, b_reg)
    rows.append({"center": "B", "variant": "Hurdle Soft(기준선)", "n": m_b_hurdle["n"],
                 "WAPE": round(m_b_hurdle["WAPE"], 3), "MAE": round(m_b_hurdle["MAE"], 3),
                 "RMSE": round(m_b_hurdle["RMSE"], 3), "MAPE": round(m_b_hurdle["MAPE"], 3)})

    return pd.DataFrame(rows)


def main():
    pooled = load_pooled_train()
    model, feature_cols = train_tweedie(pooled)

    a_cut_val, b_cut_val = internal_cutoff_val(pooled)
    Xa = prepare_X(a_cut_val, feature_cols)
    pred_a_val = np.clip(model.predict(Xa), 0, None)
    Xb = prepare_X(b_cut_val, feature_cols)
    pred_b_val = np.clip(model.predict(Xb), 0, None)

    print("=" * 80)
    print(f"[cutoff 탐색, 내부 val(2024 미포함)] A n={len(a_cut_val):,} / B n={len(b_cut_val):,}")
    _, best_cutoff_a = sweep_cutoff(a_cut_val[TARGET_COL].to_numpy(), pred_a_val, a_cut_val["qty"].to_numpy(), "A")
    _, best_cutoff_b = sweep_cutoff(b_cut_val[TARGET_COL].to_numpy(), pred_b_val, b_cut_val["qty"].to_numpy(), "B")

    print()
    print("=" * 80)
    print(f"[최종 홀드아웃 1회 적용] A cutoff={best_cutoff_a:.2f} / B cutoff={best_cutoff_b:.2f}")
    report_df = final_holdout(model, feature_cols, best_cutoff_a, best_cutoff_b)
    print(report_df.to_string(index=False))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_df.to_csv(REPORT_DIR / "tweedie_no_leak_report.csv", index=False, encoding="utf-8-sig")
    md_lines = [
        "# Tweedie 대조실험 (2024 미유출 버전)",
        "",
        f"채택 cutoff: A={best_cutoff_a:.2f}, B={best_cutoff_b:.2f}",
        "",
        "| " + " | ".join(report_df.columns) + " |",
        "| " + " | ".join(["---"] * len(report_df.columns)) + " |",
    ]
    for _, r in report_df.iterrows():
        md_lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    md_lines.append("")
    md_lines.append(
        "각주: 학습(variance_power 선택 포함)과 cutoff 탐색 모두 pooled train(A 2021~2023 + "
        "B 2023-07~12)의 시간순 마지막 4주 내부 val만 사용 — 2024 데이터는 최종 홀드아웃 1회 "
        "평가에만 등장. Hurdle Soft 기준선은 재학습 없이 기존 모델(A: a_final_cv, B: production "
        "base_model)을 그대로 평가한 값. 모델은 data/ml/cv_experiment/models/tweedie_no_leak_reg.pkl "
        "에 저장, 프로덕션 tweedie_reg.pkl은 미변경."
    )
    (REPORT_DIR / "tweedie_no_leak_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n저장 완료 -> {REPORT_DIR}/tweedie_no_leak_report.{{csv,md}}, 모델 -> {TWEEDIE_PATH}")


if __name__ == "__main__":
    main()
