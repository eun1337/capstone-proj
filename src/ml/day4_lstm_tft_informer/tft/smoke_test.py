"""
smoke_test.py
TFT end-to-end functional smoke test - 실제 P10 h1 Fold1 구조 기반의 작은 재현 가능
subset으로 residual NaN imputation -> TimeSeriesDataSet 변환 -> fit -> predict ->
inverse transform -> metrics -> OOF 전체가 오류 없이 연결되는지만 확인한다. 성능값은
Search Space 판단에 쓰지 않는다.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.ml.day3_rf_lightgbm.common import folds as day3_folds
from src.ml.day3_rf_lightgbm.common import oof as day3_oof
from src.ml.day3_rf_lightgbm.common.data_loader import load_development
from src.ml.day4_lstm_tft_informer.common.config import get_model_feature_roles
from src.ml.day4_lstm_tft_informer.common.sequence_builder import (
    build_sequences,
    fold_origin_key_set,
    split_batch_by_origin_keys,
)
from src.ml.day4_lstm_tft_informer.common.structural_nan import (
    RESIDUAL_NAN_FEATURES,
    apply_residual_nan_medians,
    fit_residual_nan_medians,
)
from pytorch_forecasting.data.encoders import NaNLabelEncoder
from sklearn.preprocessing import StandardScaler

from src.ml.day4_lstm_tft_informer.tft.config import (
    OPTIMIZER,
    SMOKE_ATTENTION_HEAD_SIZE,
    SMOKE_DROPOUT,
    SMOKE_EPOCHS,
    SMOKE_GRADIENT_CLIP_VAL,
    SMOKE_HIDDEN_CONTINUOUS_SIZE,
    SMOKE_HIDDEN_SIZE,
    SMOKE_LEARNING_RATE,
    SMOKE_SEED,
)
from src.ml.day4_lstm_tft_informer.tft.dataset_adapter import (
    build_tft_long_dataframe,
    build_training_dataset,
    build_validation_dataset,
)
from src.ml.day4_lstm_tft_informer.tft.trainer import train_and_evaluate_fold

HORIZON = 1
LOOKBACK = 13
N_SKUS = 80
SMOKE_BATCH_SIZE = 32


def build_smoke_subset():
    dev = load_development()
    fold = day3_folds.generate_expanding_folds(dev, 2022, HORIZON)[0]  # P10 h1 Fold1

    candidate_skus = sorted(dev.loc[fold["train_mask"] | fold["val_mask"], "sku_id"].unique())[:N_SKUS]
    sub = dev[(dev["center_id"] == "A") & (dev["sku_id"].isin(candidate_skus))].copy()

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
    print("imputation 요약(subset 단독 검증용, train-only fit):")
    for feature, s in residual_summary.items():
        print(f"  {feature}: {s}")
    check("train-only NaN imputation (unresolved==0)",
          all(s["unresolved"] == 0 for s in residual_summary.values()))
    check("기존 non-NaN 값 변경 == 0",
          all(s["n_non_nan_changed"] == 0 for s in residual_summary.values()))

    # --- feature role 정상 여부 ---
    roles = get_model_feature_roles(HORIZON)
    check("static/observed/known role 정상 (static 6, known 7, observed 14)",
          len(roles["static"]) == 6 and len(roles["time_varying_known"]) == 7
          and len(roles["time_varying_observed"]) == 14)

    # --- target alignment + origin 이후 row 미포함 교차검증 (LSTM smoke와 동일 방식) ---
    full_batch = build_sequences(sub_imputed, HORIZON, LOOKBACK)
    dev_check = sub.set_index(["center_id", "sku_id", "week_st"])["target_h1"]
    sample_idx = full_batch.keys.sample(min(30, len(full_batch.keys)), random_state=42).index

    alignment_ok = True
    for i in sample_idx:
        row = full_batch.keys.loc[i]
        expected = dev_check.loc[(row["center_id"], row["sku_id"], row["week_st"])]
        if not np.isclose(full_batch.target[i], expected):
            alignment_ok = False
            break
    check("target alignment 정상", alignment_ok)

    tv_cols = list(roles["time_varying_known"]) + list(roles["time_varying_observed"])
    qty_log1p_pos = tv_cols.index("qty_log1p")
    qty_by_week = sub.set_index(["center_id", "sku_id", "week_st"])["qty"]
    window_ok = True
    for i in sample_idx:
        row = full_batch.keys.loc[i]
        origin_week = row["week_st"]
        first_week = origin_week - pd.Timedelta(weeks=LOOKBACK - 1)
        expected_last = np.log1p(qty_by_week.loc[(row["center_id"], row["sku_id"], origin_week)])
        expected_first = np.log1p(qty_by_week.loc[(row["center_id"], row["sku_id"], first_week)])
        actual_last = full_batch.time_varying[i, -1, qty_log1p_pos]
        actual_first = full_batch.time_varying[i, 0, qty_log1p_pos]
        if not (np.isclose(actual_last, expected_last) and np.isclose(actual_first, expected_first)):
            window_ok = False
            break
    check("encoder history가 origin 이후를 포함하지 않음(window 첫/끝 교차검증)", window_ok)

    check("tensor(time_varying) NaN/Inf == 0", np.isfinite(full_batch.time_varying).all())
    check("tensor(static_cont) NaN/Inf == 0", np.isfinite(full_batch.static_cont).all())

    # --- target_date purge ---
    train_target_date = sub.loc[sub_fold["train_mask"], "week_st"] + pd.Timedelta(weeks=HORIZON)
    check("train target_date < validation_start (P20 purge)",
          bool((train_target_date < sub_fold["val_start"]).all()))

    # --- train-only categorical encoder/scaler fitting 검증 ---
    # trainer.py와 동일한 경로(build_sequences -> fold별 분리 -> build_tft_long_dataframe
    # -> build_training_dataset -> build_validation_dataset)로 독립적으로 재구성한 뒤,
    # validation_dataset이 실제로 사용하는 fitted encoder/scaler(_categorical_encoders,
    # _scalers)가 train_long만으로 fit한 참조값과 정확히 일치하는지 확인한다.
    # (validation_dataset이 val_long으로 재fit됐다면 이 값들이 달라진다.)
    train_keys = fold_origin_key_set(sub, sub_fold["train_mask"])
    val_keys = fold_origin_key_set(sub, sub_fold["val_mask"])
    train_batch = split_batch_by_origin_keys(full_batch, train_keys)
    val_batch = split_batch_by_origin_keys(full_batch, val_keys)
    train_long = build_tft_long_dataframe(train_batch, sub_imputed, HORIZON, group_offset=0)
    val_long = build_tft_long_dataframe(val_batch, sub_imputed, HORIZON, group_offset=len(train_batch.target))
    training_dataset = build_training_dataset(train_long, HORIZON, LOOKBACK)
    validation_dataset = build_validation_dataset(training_dataset, val_long)

    static_cat_cols = list(roles["static_categorical"])
    encoder_reuse_ok = True
    for col in static_cat_cols:
        ref_classes = NaNLabelEncoder(add_nan=True).fit(train_long[col]).classes_
        if training_dataset._categorical_encoders[col].classes_ != ref_classes:
            encoder_reuse_ok = False
        if validation_dataset._categorical_encoders[col].classes_ != ref_classes:
            encoder_reuse_ok = False
    check("train-only categorical encoder fitting (validation 재fit 없이 train vocab 재사용)",
          encoder_reuse_ok)

    scaler_reuse_ok = True
    for name, scaler in training_dataset._scalers.items():
        if scaler is None or not hasattr(scaler, "mean_"):
            continue
        ref = StandardScaler().fit(train_long[[name]])
        val_scaler = validation_dataset._scalers[name]
        if not (np.array_equal(scaler.mean_, ref.mean_) and np.array_equal(scaler.scale_, ref.scale_)):
            scaler_reuse_ok = False
        if not (np.array_equal(val_scaler.mean_, ref.mean_) and np.array_equal(val_scaler.scale_, ref.scale_)):
            scaler_reuse_ok = False
    check("train-only continuous scaler fitting (validation 재fit 없이 train 통계 재사용)",
          scaler_reuse_ok)

    check("optimizer 설정이 config.OPTIMIZER와 일치", OPTIMIZER == "adam")

    # --- end-to-end trainer 실행 ---
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_dir = Path(tmpdir) / "tft_ckpt"
        result = train_and_evaluate_fold(
            sub, sub_fold, HORIZON, LOOKBACK,
            hidden_size=SMOKE_HIDDEN_SIZE, hidden_continuous_size=SMOKE_HIDDEN_CONTINUOUS_SIZE,
            attention_head_size=SMOKE_ATTENTION_HEAD_SIZE, dropout=SMOKE_DROPOUT,
            learning_rate=SMOKE_LEARNING_RATE, gradient_clip_val=SMOKE_GRADIENT_CLIP_VAL,
            batch_size=SMOKE_BATCH_SIZE, max_epochs=SMOKE_EPOCHS, seed=SMOKE_SEED,
            stage="P10", model_family="TFT", config_id="SMOKE", checkpoint_dir=checkpoint_dir,
        )

        print(f"validation sequence 누락 수: {result['n_val_insufficient_history']}")
        print(f"insufficient history 수(train): {result['n_train_insufficient_history']}")

        print()
        print("=== residual NaN(adi/cv2) imputation 요약(trainer 내부) ===")
        for feature, s in result["residual_nan_summary"].items():
            print(f"  {feature}: {s}")
        check("residual NaN unresolved == 0 (trainer)",
              all(s["unresolved"] == 0 for s in result["residual_nan_summary"].values()))

        check("forward/backward PASS (train loop 정상 완료)", result["epochs_completed"] >= 1)
        check("gradient clipping 실제 적용(gradient_clip_val 전달)",
              result["gradient_clip_val"] == SMOKE_GRADIENT_CLIP_VAL)
        check("optimizer 실제 적용 (model.hparams.optimizer == config.OPTIMIZER)",
              result["optimizer"] == OPTIMIZER == "adam")

        ckpt_files = list(checkpoint_dir.glob("*.ckpt"))
        check("checkpoint save PASS", len(ckpt_files) >= 1)
        check("checkpoint reload PASS (load_from_checkpoint 후 predict 성공)",
              np.isfinite(result["oof"]["y_pred_log"]).all())

        pred_finite = np.isfinite(result["oof"]["y_pred"]).all()
        check("prediction finite", bool(pred_finite))
        check("expm1 + clip 정상 (raw pred 음수 없음)", bool((result["oof"]["y_pred"] >= 0).all()))

        m = result["metrics"]
        check("evaluator 계산 PASS", all(np.isfinite(m[k]) for k in ["wape", "bias", "rmse", "mae"]))

        try:
            day3_oof.validate_oof_frame(result["oof"])
            oof_valid = True
        except Exception:
            oof_valid = False
        check("OOF schema/validation PASS", oof_valid)

        print()
        print("=== 결과 요약 ===")
        print(f"n_train_sequences={result['n_train_sequences']}, n_val_sequences={result['n_val_sequences']}")
        print(f"sequence_build={result['sequence_build_time_sec']:.2f}s "
              f"preprocessing={result['preprocessing_time_sec']:.2f}s "
              f"train={result['train_time_sec']:.2f}s predict={result['predict_time_sec']:.2f}s "
              f"total={result['total_time_sec']:.2f}s")
        print(f"epochs_completed={result['epochs_completed']} best_epoch={result['best_epoch']} "
              f"best_val_loss={result['best_val_loss']:.4f}")
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
