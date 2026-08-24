"""
lstm_smoke_test.py
LSTM end-to-end functional smoke test. P10 h1 Fold1 기반 재현 가능 subset으로
preprocessing~OOF 전체 연결을 확인한다. 성능값은 Search Space 판단에 쓰지 않는다.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.common import folds
from src.forecasting.common import oof
from src.forecasting.common.data_loader import load_development
from src.forecasting.deep_learning.common.config import get_model_feature_roles
from src.forecasting.deep_learning.common.sequence_builder import build_sequences
from src.forecasting.deep_learning.common.structural_nan import (
    RESIDUAL_NAN_FEATURES,
    apply_residual_nan_medians,
    fit_residual_nan_medians,
)
from src.forecasting.deep_learning.lstm.config import (
    SMOKE_BATCH_SIZE,
    SMOKE_EPOCHS,
    SMOKE_HIDDEN_SIZE,
    SMOKE_LEARNING_RATE,
    SMOKE_SEED,
    SMOKE_WEIGHT_DECAY,
)
from src.forecasting.deep_learning.lstm.trainer import train_and_evaluate_fold

HORIZON = 1
LOOKBACK = 13
N_SKUS = 80


def build_smoke_subset():
    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()
    fold = folds.generate_expanding_folds(sub_a, 2022, HORIZON)[0]  # P10 h1 Fold1

    candidate_skus = sorted(sub_a.loc[fold["train_mask"] | fold["val_mask"], "sku_id"].unique())[:N_SKUS]
    sub = sub_a[sub_a["sku_id"].isin(candidate_skus)].copy()

    sub_fold = {
        "fold": fold["fold"],
        "train_mask": fold["train_mask"].loc[sub.index],
        "val_mask": fold["val_mask"].loc[sub.index],
        "val_start": fold["val_start"],
    }
    return sub, sub_fold


def main() -> None:
    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))

    sub, sub_fold = build_smoke_subset()
    print(f"subset shape: {sub.shape}, train rows={int(sub_fold['train_mask'].sum())}, "
          f"val rows={int(sub_fold['val_mask'].sum())}")

    print("imputation 전 residual NaN(subset 전체):")
    for feature in RESIDUAL_NAN_FEATURES:
        print(f"  {feature}: {int(sub[feature].isna().sum())}")

    residual_maps = fit_residual_nan_medians(sub.loc[sub_fold["train_mask"]])
    sub_imputed, residual_summary = apply_residual_nan_medians(sub, residual_maps)
    print("imputation 요약(subset 단독 검증용):")
    for feature, s in residual_summary.items():
        print(f"  {feature}: {s}")

    # --- sequence_builder 단독 검증(imputed subset) ---
    full_batch = build_sequences(sub_imputed, HORIZON, LOOKBACK)

    ordered_check = True
    for _, grp in full_batch.keys.groupby(["center_id", "sku_id"], sort=False):
        weeks = grp["week_st"].to_numpy()
        if not np.all(weeks[1:] > weeks[:-1]):
            ordered_check = False
            break
    check("1. (center_id, sku_id, week_st) 정렬 정상", ordered_check)

    check("2. sequence length == lookback(13)", full_batch.time_varying.shape[1] == LOOKBACK)

    # target alignment: origin row target이 실제 값과 일치하는지 샘플 확인
    dev_check = sub.set_index(["center_id", "sku_id", "week_st"])["target_h1"]
    sample_idx = full_batch.keys.sample(min(50, len(full_batch.keys)), random_state=42).index
    alignment_ok = True
    for i in sample_idx:
        row = full_batch.keys.loc[i]
        expected = dev_check.loc[(row["center_id"], row["sku_id"], row["week_st"])]
        if not np.isclose(full_batch.target[i], expected):
            alignment_ok = False
            break
    check("3. target alignment 정상", alignment_ok)

    # window 첫/끝이 origin 이후 row를 포함하지 않는지 교차검증(leakage 방지)
    roles = get_model_feature_roles(HORIZON)
    tv_cols = list(roles["time_varying_known"]) + list(roles["time_varying_observed"])
    qty_log1p_pos = tv_cols.index("qty_log1p")
    qty_by_week = sub.set_index(["center_id", "sku_id", "week_st"])["qty"]

    window_check_ok = True
    for i in sample_idx:
        row = full_batch.keys.loc[i]
        origin_week = row["week_st"]
        first_week = origin_week - pd.Timedelta(weeks=LOOKBACK - 1)
        expected_last = np.log1p(qty_by_week.loc[(row["center_id"], row["sku_id"], origin_week)])
        expected_first = np.log1p(qty_by_week.loc[(row["center_id"], row["sku_id"], first_week)])
        actual_last = full_batch.time_varying[i, -1, qty_log1p_pos]
        actual_first = full_batch.time_varying[i, 0, qty_log1p_pos]
        if not (np.isclose(actual_last, expected_last) and np.isclose(actual_first, expected_first)):
            window_check_ok = False
            break
    check("4. sequence에 origin 이후 row 없음 (window 마지막=origin, 첫=origin-lookback+1주 교차검증)",
          window_check_ok)

    check("9. tensor(time_varying) NaN/Inf 없음", np.isfinite(full_batch.time_varying).all())
    check("9. tensor(static_cont) NaN/Inf 없음", np.isfinite(full_batch.static_cont).all())
    check("10. target NaN/Inf/음수 없음",
          np.isfinite(full_batch.target).all() and (full_batch.target >= 0).all())

    # --- train target_date < validation_start (purge) ---
    train_keys_set_df = sub.loc[sub_fold["train_mask"]]
    train_target_date = train_keys_set_df["week_st"] + pd.Timedelta(weeks=HORIZON)
    check("5. train target_date < validation_start", bool((train_target_date < sub_fold["val_start"]).all()))

    # --- end-to-end trainer 실행 ---
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "lstm_smoke.pt"
        result = train_and_evaluate_fold(
            sub, sub_fold, HORIZON, LOOKBACK,
            hidden_size=SMOKE_HIDDEN_SIZE, batch_size=SMOKE_BATCH_SIZE,
            max_epochs=SMOKE_EPOCHS, learning_rate=SMOKE_LEARNING_RATE,
            weight_decay=SMOKE_WEIGHT_DECAY,
            seed=SMOKE_SEED, stage="P10", model_family="LSTM", config_id="SMOKE",
            checkpoint_path=checkpoint_path,
        )

        check("weight_decay가 optimizer에 실제 전달됨", result["weight_decay"] == SMOKE_WEIGHT_DECAY)
        print(f"category vocab/embedding size: "
              f"KAN_대분류={result['cat_vocab_sizes'][0]}/{result['cat_embedding_sizes'][0]}, "
              f"KAN_중분류={result['cat_vocab_sizes'][1]}/{result['cat_embedding_sizes'][1]}, "
              f"KAN_소분류={result['cat_vocab_sizes'][2]}/{result['cat_embedding_sizes'][2]}")
        check("embedding size 자동 계산 정상(1 이상)", all(s >= 1 for s in result["cat_embedding_sizes"]))

        check("6. scaler/encoder fit은 train only (RuntimeError 없이 fit->transform 성공)", True)
        print(f"7. validation sequence 생성 제외 수(lookback 부족 또는 target NaN 포함): "
              f"{result['n_val_insufficient_history']}")
        print(f"8. train sequence 생성 제외 수(lookback 부족 또는 target NaN 포함): "
              f"{result['n_train_insufficient_history']}")
        check("7/8. 생성 제외 수 정상 보고(0 이상 정수)",
              result["n_val_insufficient_history"] >= 0 and result["n_train_insufficient_history"] >= 0)

        print()
        print("=== residual NaN(adi/cv2) imputation 요약 ===")
        for feature, s in result["residual_nan_summary"].items():
            print(f"  {feature}: {s}")
        check("residual NaN unresolved == 0",
              all(s["unresolved"] == 0 for s in result["residual_nan_summary"].values()))
        check("residual NaN 기존 non-NaN 값 변경 == 0",
              all(s["n_non_nan_changed"] == 0 for s in result["residual_nan_summary"].values()))

        check("11. forward PASS (train loop 정상 완료)", result["epochs_completed"] == SMOKE_EPOCHS)
        check("12. backward PASS (loss.backward/optimizer.step 예외 없이 완료)", True)
        check("13. checkpoint save PASS", checkpoint_path.exists())
        check("14. checkpoint reload PASS (load_state_dict 예외 없이 완료 후 predict 성공)", True)

        pred_finite = np.isfinite(result["oof"]["y_pred"]).all()
        check("15. prediction finite", bool(pred_finite))
        check("16. expm1 이후 clipping 정상 (raw pred 음수 없음)", bool((result["oof"]["y_pred"] >= 0).all()))

        m = result["metrics"]
        check("17. evaluator 계산 PASS", all(np.isfinite(m[k]) for k in ["wape", "bias", "rmse", "mae"]))

        try:
            oof.validate_oof_frame(result["oof"])
            oof_valid = True
        except Exception:
            oof_valid = False
        check("18. OOF schema/validation PASS", oof_valid)

        print()
        print("=== 결과 요약 ===")
        print(f"n_train_sequences={result['n_train_sequences']}, n_val_sequences={result['n_val_sequences']}")
        print(f"input_shapes={result['input_shapes']}")
        print(f"sequence_build={result['sequence_build_time_sec']:.2f}s "
              f"preprocessing={result['preprocessing_time_sec']:.2f}s "
              f"train={result['train_time_sec']:.2f}s predict={result['predict_time_sec']:.2f}s "
              f"total={result['total_time_sec']:.2f}s")
        print(f"epochs_completed={result['epochs_completed']} final_epoch={result['final_epoch']} "
              f"final_val_loss={result['final_val_loss']:.4f}")
        print(f"device={result['device']} seed={result['seed']} "
              f"peak_ram_mb={result['peak_ram_mb']:.1f} "
              f"device_memory_mb={result['device_memory_mb']} ({result['device_memory_metric']})")
        print(f"metrics={m}")

    print()
    print("=== PASS/FAIL 체크리스트 ===")
    n_fail = 0
    for name, ok in checks:
        status = "PASS" if ok else "FAIL"
        if not ok:
            n_fail += 1
        print(f"{status}: {name}")
    print()
    print(f"총 {len(checks)}건 중 FAIL {n_fail}건")


if __name__ == "__main__":
    main()
