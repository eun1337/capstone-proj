"""
run_es_lr_sensitivity.py
직전 실험(run_a_expanding_cv_lgbm_multih.py)에서 발견된 이상 신호 진단:
best_hyperparams_pure.json의 분류기 learning_rate(0.121)를 학습량이 다른 4개 Fold에
그대로 재사용했더니, train이 커질수록(Fold1: 143~182회 -> Fold3/4: 17~20회) 조기종료가
비정상적으로 빨리 걸렸다. "Fold1이 더 낫다"는 결론이 진짜인지, 하이퍼파라미터
미스매치로 인한 착시인지 가리기 위해 두 가지를 각 Fold(h=1, 분류기 단독)에서 스윕한다:

    1) Early-stopping 완화: early_stopping_rounds를 30(원래값)에서 60/100/150으로
       늘렸을 때 best_iteration/val logloss가 어떻게 바뀌는지.
    2) Learning rate 민감도: learning_rate를 0.01/0.03/0.05/0.08/0.121(pure 원래값)로
       바꿔가며(early_stopping_rounds=30 고정) best_iteration/val logloss가 어떻게
       바뀌는지 — 0.121이 학습량이 큰 폴드에서 유독 빨리 튀는지 확인.

분류기만 본다(회귀기 best_iter는 폴드 간 덜 요동쳤음 - 204/260/99/283 vs 분류기
143/31/17/18). feature_cols/데이터 전처리는 run_a_expanding_cv_lgbm_multih.py와
완전히 동일(재사용, import).

python run_es_lr_sensitivity.py

산출물: data/ml/cv_experiment/reports/es_lr_sensitivity_report.{csv,md}
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import cv_common as cc  # noqa: E402
import folds  # noqa: E402
import run_a_expanding_cv_lgbm_multih as prev  # noqa: E402

BASE_DIR = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(BASE_DIR / "src" / "forecasting" / "archive" / "models_hurdle"))
from common import prepare_X  # noqa: E402

TARGET_COL = "target_h1"
REPORT_DIR = cc.CV_DIR / "reports"

ES_GRID = [30, 60, 100, 150]
LR_GRID = [0.01, 0.03, 0.05, 0.08, 0.121]
BASE_PARAMS = dict(num_leaves=19, max_depth=15, min_child_samples=46,
                    subsample=0.5909124836035503, colsample_bytree=0.5917022549267169,
                    reg_alpha=5.472429642032198e-06, reg_lambda=0.00052821153945323)
SCALE_POS_WEIGHT_MULTIPLIER = 1.5798625466052894
N_ESTIMATORS = 1000
RANDOM_STATE = 42


def prep_fold(a_full: pd.DataFrame, f: dict):
    train_window = cc.slice_by_date(a_full, f["train_start"], f["train_end"])
    train_window = cc.apply_boundary_filter(train_window, f["train_end"], horizon_weeks=1)
    train_window = train_window[train_window[TARGET_COL].notna()]
    train_fit, es_val = cc.carve_internal_es_val(train_window)
    feature_cols = prev.get_feature_cols_a(train_fit)

    y_train = (train_fit[TARGET_COL] > 0).astype(int)
    y_val = (es_val[TARGET_COL] > 0).astype(int)
    X_train = prepare_X(train_fit, feature_cols)
    X_val = prepare_X(es_val, feature_cols)
    natural_spw = int((y_train == 0).sum()) / max(int((y_train == 1).sum()), 1)
    return X_train, y_train, X_val, y_val, natural_spw, len(train_fit)


def fit_classifier(X_train, y_train, X_val, y_val, natural_spw, learning_rate, early_stopping_rounds):
    scale_pos_weight = natural_spw * SCALE_POS_WEIGHT_MULTIPLIER
    model = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS, learning_rate=learning_rate, scale_pos_weight=scale_pos_weight,
        subsample_freq=1, verbose=-1, random_state=RANDOM_STATE, n_jobs=-1, **BASE_PARAMS,
    )
    model.fit(
        X_train, y_train, eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False), lgb.log_evaluation(0)],
    )
    return {
        "best_iteration": model.best_iteration_,
        "val_logloss": model.best_score_["valid_0"]["binary_logloss"],
        "hit_n_estimators_ceiling": model.best_iteration_ >= N_ESTIMATORS - 1,
    }


def main():
    print("=== A_full 로드 ===")
    a_full = cc.load_a_full()

    es_rows, lr_rows = [], []
    for f in folds.A_EXPANDING_H2_FOLDS:
        fold_id = f["fold"]
        print("=" * 80)
        print(f"[Fold {fold_id}] train {f['train_start']}~{f['train_end']}")
        X_train, y_train, X_val, y_val, natural_spw, n_train = prep_fold(a_full, f)
        print(f"  train n={n_train:,}, natural_scale_pos_weight={natural_spw:.3f}")

        print("  -- Early-stopping 완화 스윕 (learning_rate=0.121 고정) --")
        for es in ES_GRID:
            r = fit_classifier(X_train, y_train, X_val, y_val, natural_spw, learning_rate=0.121, early_stopping_rounds=es)
            print(f"    es_rounds={es:>3} -> best_iter={r['best_iteration']:>4} val_logloss={r['val_logloss']:.5f} "
                  f"{'(ceiling 도달!)' if r['hit_n_estimators_ceiling'] else ''}")
            es_rows.append({"fold": fold_id, "train_n": n_train, "early_stopping_rounds": es,
                             "learning_rate": 0.121, **r})

        print("  -- Learning rate 민감도 스윕 (early_stopping_rounds=30 고정) --")
        for lr in LR_GRID:
            r = fit_classifier(X_train, y_train, X_val, y_val, natural_spw, learning_rate=lr, early_stopping_rounds=30)
            print(f"    lr={lr:<6} -> best_iter={r['best_iteration']:>4} val_logloss={r['val_logloss']:.5f}")
            lr_rows.append({"fold": fold_id, "train_n": n_train, "learning_rate": lr,
                             "early_stopping_rounds": 30, **r})

    es_df = pd.DataFrame(es_rows)
    lr_df = pd.DataFrame(lr_rows)

    print()
    print("=" * 80)
    print("[Early-stopping 완화 스윕 결과]")
    print(es_df.to_string(index=False))
    print()
    print("[Learning rate 민감도 스윕 결과]")
    print(lr_df.to_string(index=False))

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    es_df.to_csv(REPORT_DIR / "es_sensitivity_report.csv", index=False, encoding="utf-8-sig")
    lr_df.to_csv(REPORT_DIR / "lr_sensitivity_report.csv", index=False, encoding="utf-8-sig")

    md_lines = ["# Early-stopping 완화 / Learning rate 민감도 진단 (A센터, 분류기 단독, h=1)", "",
                "## Early-stopping rounds 스윕 (learning_rate=0.121 고정)", "",
                "| " + " | ".join(es_df.columns) + " |",
                "| " + " | ".join(["---"] * len(es_df.columns)) + " |"]
    for _, r in es_df.iterrows():
        md_lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    md_lines += ["", "## Learning rate 스윕 (early_stopping_rounds=30 고정)", "",
                 "| " + " | ".join(lr_df.columns) + " |",
                 "| " + " | ".join(["---"] * len(lr_df.columns)) + " |"]
    for _, r in lr_df.iterrows():
        md_lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    md_lines += ["", "각주: 회귀기는 포함하지 않음(분류기 best_iteration만 폴드 간 크게 요동쳤음 — "
                      "143~182 -> 17~20). num_leaves/max_depth 등 나머지 하이퍼파라미터는 "
                      "best_hyperparams_pure.json 값 그대로 고정."]
    (REPORT_DIR / "es_lr_sensitivity_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\n저장 완료 -> {REPORT_DIR}/{{es_sensitivity_report.csv, lr_sensitivity_report.csv, es_lr_sensitivity_report.md}}")


if __name__ == "__main__":
    main()
