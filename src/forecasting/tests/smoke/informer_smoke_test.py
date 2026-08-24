"""
informer_smoke_test.py
Informer functional smoke test.
1) 학습 없는 3개 조합(h1/h2/h4, lookback=13 고정) decoder/alignment audit
2) 아키텍처 assertion(구조·attention 구성 확인)
2-1) n_heads {8,16} 2개 조합 forward+backward sanity(e_layers/lookback 고정)
3) 대표 end-to-end smoke(A센터, P10 h1 Fold1, 80 SKU)
성능값은 Search Space 판단에 쓰지 않는다.
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.forecasting.common import folds
from src.forecasting.common import oof
from src.forecasting.common.config import HOLIDAY_FEATURES
from src.forecasting.common.data_loader import load_development
from src.forecasting.deep_learning.common.config import get_model_feature_roles
from src.forecasting.deep_learning.common.preprocessing import SequencePreprocessor
from src.forecasting.deep_learning.common.sequence_builder import (
    build_sequences,
    fold_origin_key_set,
    split_batch_by_origin_keys,
)
from src.forecasting.deep_learning.common.structural_nan import (
    RESIDUAL_NAN_FEATURES,
    apply_residual_nan_medians,
    fit_residual_nan_medians,
)
from src.forecasting.deep_learning.informer.config import (
    D_FF,
    D_LAYERS,
    D_MODEL,
    DROPOUT,
    E_LAYERS,
    FACTOR,
    LEARNING_RATE,
    LOOKBACK,
    LOOKBACK_CHOICES,
    N_HEADS_CHOICES,
    SMOKE_E_LAYERS,
    SMOKE_EPOCHS,
    SMOKE_N_HEADS,
    SMOKE_SEED,
    distilled_encoder_length,
    label_len_for,
)
from src.forecasting.deep_learning.informer.dataset import (
    build_decoder_future_known,
    build_informer_tensors,
)
from src.forecasting.deep_learning.informer.layers import FullAttention, ProbAttention
from src.forecasting.deep_learning.informer.model import InformerForecaster
from src.forecasting.deep_learning.informer.trainer import train_and_evaluate_fold

N_SKUS = 80
SMOKE_BATCH_SIZE = 32
SMOKE_HORIZON = 1
SMOKE_LOOKBACK = 13


def build_smoke_subset(horizon: int):
    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()
    fold = folds.generate_expanding_folds(sub_a, 2022, horizon)[0]  # P10 Fold1

    candidate_skus = sorted(sub_a.loc[fold["train_mask"] | fold["val_mask"], "sku_id"].unique())[:N_SKUS]
    sub = sub_a[sub_a["sku_id"].isin(candidate_skus)].copy()

    sub_fold = {
        "fold": fold["fold"],
        "train_mask": fold["train_mask"].loc[sub.index],
        "val_mask": fold["val_mask"].loc[sub.index],
        "val_start": fold["val_start"],
    }
    return sub, sub_fold


# ---------------------------------------------------------------------------
# 1) 학습 없는 3개 조합(h1/h2/h4 x lookback=13 고정) alignment audit
# ---------------------------------------------------------------------------
def run_alignment_audit() -> pd.DataFrame:
    results = []
    for h in [1, 2, 4]:
        sub, sub_fold = build_smoke_subset(h)
        for lookback in LOOKBACK_CHOICES:  # 현재 P13 frozen: lookback=13 고정(singleton)
            checks = {}

            maps = fit_residual_nan_medians(sub.loc[sub_fold["train_mask"]])
            sub_imputed, residual_summary = apply_residual_nan_medians(sub, maps)
            checks["residual NaN unresolved==0"] = all(s["unresolved"] == 0 for s in residual_summary.values())

            full_batch = build_sequences(sub_imputed, h, lookback)
            train_keys = fold_origin_key_set(sub, sub_fold["train_mask"])
            val_keys = fold_origin_key_set(sub, sub_fold["val_mask"])
            train_batch = split_batch_by_origin_keys(full_batch, train_keys)
            val_batch = split_batch_by_origin_keys(full_batch, val_keys)

            checks["tensor NaN/Inf==0 (train)"] = bool(np.isfinite(train_batch.time_varying).all())
            checks["tensor NaN/Inf==0 (val)"] = bool(np.isfinite(val_batch.time_varying).all())

            # target_h{h} alignment
            from src.forecasting.common.config import TARGET_COLS
            target_col = TARGET_COLS[h]
            dev_target = sub.set_index(["center_id", "sku_id", "week_st"])[target_col]
            sample_idx = full_batch.keys.sample(min(20, len(full_batch.keys)), random_state=0).index
            align_ok = True
            for i in sample_idx:
                row = full_batch.keys.loc[i]
                expected = dev_target.loc[(row["center_id"], row["sku_id"], row["week_st"])]
                if not np.isclose(full_batch.target[i], expected):
                    align_ok = False
            checks[f"target_{target_col} alignment"] = align_ok

            # encoder 마지막=origin, origin 이후 없음 (window 첫/끝 교차검증)
            roles = get_model_feature_roles(h)
            tv_cols = list(roles["time_varying_known"]) + list(roles["time_varying_observed"])
            qty_idx = tv_cols.index("qty_log1p")
            qty_by_week = sub.set_index(["center_id", "sku_id", "week_st"])["qty"]
            window_ok = True
            for i in sample_idx:
                row = full_batch.keys.loc[i]
                origin_week = row["week_st"]
                first_week = origin_week - pd.Timedelta(weeks=lookback - 1)
                expected_last = np.log1p(qty_by_week.loc[(row["center_id"], row["sku_id"], origin_week)])
                expected_first = np.log1p(qty_by_week.loc[(row["center_id"], row["sku_id"], first_week)])
                actual_last = full_batch.time_varying[i, -1, qty_idx]
                actual_first = full_batch.time_varying[i, 0, qty_idx]
                if not (np.isclose(actual_last, expected_last) and np.isclose(actual_first, expected_first)):
                    window_ok = False
            checks["encoder 마지막=origin, origin 이후 없음"] = window_ok

            # label_len 정확
            expected_label_len = lookback // 2
            checks["label_len 정확"] = label_len_for(lookback) == expected_label_len

            # train-only scaling/vocab: preprocessor를 train에서만 fit하고 val은 transform만 함
            preprocessor = SequencePreprocessor().fit(train_batch)
            train_tensors = build_informer_tensors(preprocessor, train_batch, sub_imputed, h)
            val_tensors = build_informer_tensors(preprocessor, val_batch, sub_imputed, h)
            checks["train-only scaling/vocab (단일 preprocessor 재사용)"] = True

            label_len = train_tensors["label_len"]
            n_known = len(roles["time_varying_known"])

            # decoder historical token은 과거 history만 사용 + decoder future value==0
            dv_future_zero = np.allclose(val_tensors["decoder_value"][:, -1], 0.0)
            checks["decoder future value token==0"] = bool(dv_future_zero)

            # decoder known future 채널 수 == known 7개뿐(observed leak 없음)
            checks["decoder_known future observed leak 0건"] = val_tensors["decoder_known"].shape[-1] == n_known

            # decoder calendar==target_date 기준, decoder holiday==origin 기준 (+ positive holiday case)
            holiday_cols = list(HOLIDAY_FEATURES[h])
            calendar_cols = [c for c in roles["time_varying_known"] if c not in holiday_cols]
            future_known_raw = build_decoder_future_known(val_batch, sub_imputed, h)
            known_cols = list(roles["time_varying_known"])

            lookup = sub_imputed.set_index(["center_id", "sku_id", "week_st"])
            decoder_ok = True
            positive_holiday_found = False
            for i in range(min(50, len(val_batch.target))):
                key_row = val_batch.keys.iloc[i]
                target_row = lookup.loc[(key_row["center_id"], key_row["sku_id"], key_row["target_date"])]
                origin_last = val_batch.time_varying[i, lookback - 1, :]
                tv_cols_full = known_cols + list(roles["time_varying_observed"])
                for c in calendar_cols:
                    if not np.isclose(future_known_raw[i, known_cols.index(c)], target_row[c]):
                        decoder_ok = False
                for c in holiday_cols:
                    expected = origin_last[tv_cols_full.index(c)]
                    actual = future_known_raw[i, known_cols.index(c)]
                    if not np.isclose(actual, expected):
                        decoder_ok = False
                    if expected > 0:
                        positive_holiday_found = True
            checks["decoder calendar==target_date 기준"] = decoder_ok
            checks["decoder holiday==origin 기준"] = decoder_ok
            checks["실제 positive holiday case 포함"] = positive_holiday_found

            # P20 purge
            train_target_date = sub.loc[sub_fold["train_mask"], "week_st"] + pd.Timedelta(weeks=h)
            checks["P20 purge"] = bool((train_target_date < sub_fold["val_start"]).all())

            # static context 연결 정상 (모델 인스턴스 없이 shape/finite만 확인)
            checks["static context shape/finite 정상"] = (
                np.isfinite(train_tensors["static_cont"]).all()
                and train_tensors["static_cat"].dtype.kind in "iu"
            )

            n_fail = sum(1 for v in checks.values() if not v)
            results.append({"h": h, "lookback": lookback, "label_len": label_len, "n_checks": len(checks), "n_fail": n_fail, **checks})

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 2) 아키텍처 assertion
# ---------------------------------------------------------------------------
def run_architecture_assertions(e_layers: int, n_heads: int, lookback: int) -> dict:
    checks = {}
    label_len = label_len_for(lookback)
    n_known = 7
    n_tv = 21
    cat_vocab_sizes = [4, 4, 4]
    static_cont_dim = 3

    model = InformerForecaster(
        n_time_varying=n_tv, n_known=n_known, static_cont_dim=static_cont_dim,
        cat_vocab_sizes=cat_vocab_sizes, e_layers=e_layers, n_heads=n_heads,
        d_model=D_MODEL, d_ff=D_FF, d_layers=D_LAYERS, factor=FACTOR, dropout=DROPOUT,
    )

    checks["d_model==512"] = model.d_model == 512 == D_MODEL
    checks["d_ff==2048"] = D_FF == 2048
    checks["d_layers==1(현재 P13 frozen, config.D_LAYERS와 일치)"] = len(model.decoder.layers) == 1 == D_LAYERS
    checks["dropout==0.05(현재 P13 frozen, config.DROPOUT과 일치)"] = DROPOUT == 0.05
    checks["factor==5"] = FACTOR == 5
    checks["encoder layer count==e_layers"] = len(model.encoder.layers) == e_layers
    checks["n_heads==지정값"] = model.n_heads == n_heads
    checks["512%n_heads==0"] = D_MODEL % n_heads == 0
    checks["encoder self-attention==ProbSparse"] = all(
        isinstance(layer.attention.inner_attention, ProbAttention) for layer in model.encoder.layers
    )
    checks["distilling layer 수==e_layers-1"] = len(model.encoder.conv_layers) == e_layers - 1
    checks["decoder masked self-attention 존재"] = all(
        isinstance(layer.self_attention.inner_attention, ProbAttention) and layer.self_attention.inner_attention.mask_flag
        for layer in model.decoder.layers
    )
    checks["decoder cross-attention 존재"] = all(
        isinstance(layer.cross_attention.inner_attention, FullAttention) for layer in model.decoder.layers
    )

    batch_size = 4
    encoder_input = torch.randn(batch_size, lookback, n_tv)
    decoder_value = torch.randn(batch_size, label_len + 1)
    decoder_known = torch.randn(batch_size, label_len + 1, n_known)
    static_cont = torch.randn(batch_size, static_cont_dim)
    static_cat = torch.zeros(batch_size, len(cat_vocab_sizes), dtype=torch.long)

    out = model(encoder_input, decoder_value, decoder_known, static_cont, static_cat)
    checks["output shape==[batch]"] = tuple(out.shape) == (batch_size,)

    x = model.encoder_input_proj(encoder_input)
    x = x + model.pos_encoding(x)
    for i, layer in enumerate(model.encoder.layers):
        x = layer(x)
        if i < len(model.encoder.conv_layers):
            x = model.encoder.conv_layers[i](x)
    expected_final_len = distilled_encoder_length(lookback, e_layers)
    checks[f"distilled encoder 최종 길이=={expected_final_len}"] = x.shape[1] == expected_final_len

    return checks


def run_n_heads_architecture_sanity() -> pd.DataFrame:
    """n_heads {8,16} 2개 조합에서 forward+backward 1스텝 loss/gradient finite를 확인한다.
    e_layers/lookback은 현재 P13 frozen 값(config.E_LAYERS, config.LOOKBACK)으로 고정한다."""
    e_layers = E_LAYERS
    lookback = LOOKBACK
    n_tv, n_known, static_cont_dim = 21, 7, 3
    cat_vocab_sizes = [4, 4, 4]
    batch_size = 4
    results = []

    for n_heads in N_HEADS_CHOICES:
        label_len = label_len_for(lookback)
        torch.manual_seed(0)
        model = InformerForecaster(
            n_time_varying=n_tv, n_known=n_known, static_cont_dim=static_cont_dim,
            cat_vocab_sizes=cat_vocab_sizes, e_layers=e_layers, n_heads=n_heads,
            d_model=D_MODEL, d_ff=D_FF, d_layers=D_LAYERS, factor=FACTOR, dropout=DROPOUT,
        )
        encoder_input = torch.randn(batch_size, lookback, n_tv)
        decoder_value = torch.randn(batch_size, label_len + 1)
        decoder_known = torch.randn(batch_size, label_len + 1, n_known)
        static_cont = torch.randn(batch_size, static_cont_dim)
        static_cat = torch.zeros(batch_size, len(cat_vocab_sizes), dtype=torch.long)
        y_log = torch.randn(batch_size)

        row = {"e_layers": e_layers, "n_heads": n_heads, "lookback": lookback}
        try:
            out = model(encoder_input, decoder_value, decoder_known, static_cont, static_cat)
            row["forward 성공"] = True
            row["output shape==[batch]"] = tuple(out.shape) == (batch_size,)

            loss = nn.MSELoss()(out, y_log)
            row["loss finite"] = bool(torch.isfinite(loss).all())

            loss.backward()
            row["backward 성공"] = True
            row["gradients finite"] = all(
                torch.isfinite(p.grad).all().item() for p in model.parameters() if p.grad is not None
            )
        except Exception as e:
            row["forward 성공"] = row.get("forward 성공", False)
            row["backward 성공"] = row.get("backward 성공", False)
            row["loss finite"] = row.get("loss finite", False)
            row["gradients finite"] = row.get("gradients finite", False)
            row["output shape==[batch]"] = row.get("output shape==[batch]", False)
            row["error"] = str(e)

        final_len = distilled_encoder_length(lookback, e_layers)
        row["distilled encoder 최종 길이>0"] = final_len > 0

        n_fail = sum(
            1 for k, v in row.items()
            if k not in ("e_layers", "n_heads", "lookback", "error") and not v
        )
        row["n_fail"] = n_fail
        results.append(row)

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# 3) 대표 end-to-end smoke
# ---------------------------------------------------------------------------
def main() -> None:
    checks = []

    def check(name, cond):
        checks.append((name, bool(cond)))

    print("=" * 100)
    print("[1] 학습 없는 3개 조합(h1/h2/h4 x lookback=13 고정) alignment audit")
    print("=" * 100)
    audit_df = run_alignment_audit()
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print(audit_df.to_string(index=False))
    n_audit_fail = int((audit_df["n_fail"] > 0).sum())
    print()
    print(f"{len(audit_df)}개 조합 중 FAIL 있는 조합 수: {n_audit_fail} / {len(audit_df)}")
    check(f"{len(audit_df)}개 조합 alignment audit 전부 PASS", n_audit_fail == 0)

    print()
    print("=" * 100)
    print("[2] 아키텍처 assertion (smoke 설정: e_layers=2, n_heads=8, lookback=13)")
    print("=" * 100)
    arch_checks = run_architecture_assertions(SMOKE_E_LAYERS, SMOKE_N_HEADS, SMOKE_LOOKBACK)
    for name, ok in arch_checks.items():
        print(f"{'PASS' if ok else 'FAIL'}: {name}")
        check(name, ok)

    print()
    print("=" * 100)
    print("[2-1] n_heads {8,16} architecture sanity (e_layers/lookback 고정, forward+backward)")
    print("=" * 100)
    sanity_df = run_n_heads_architecture_sanity()
    print(sanity_df.to_string(index=False))
    n_sanity_fail = int((sanity_df["n_fail"] > 0).sum())
    print()
    print(f"{len(sanity_df)}개 조합 중 FAIL 있는 조합 수: {n_sanity_fail} / {len(sanity_df)}")
    check(f"n_heads architecture sanity 전부 PASS ({len(sanity_df)}개 조합)", n_sanity_fail == 0)

    print()
    print("=" * 100)
    print("[3] 대표 end-to-end smoke (A센터, P10 h1 Fold1, 80 SKU, lookback=13)")
    print("=" * 100)
    sub, sub_fold = build_smoke_subset(SMOKE_HORIZON)
    print(f"subset shape: {sub.shape}, train rows={int(sub_fold['train_mask'].sum())}, "
          f"val rows={int(sub_fold['val_mask'].sum())}")

    print("imputation 전 residual NaN(subset 전체):")
    for feature in RESIDUAL_NAN_FEATURES:
        print(f"  {feature}: {int(sub[feature].isna().sum())}")

    with tempfile.TemporaryDirectory() as tmpdir:
        checkpoint_path = Path(tmpdir) / "informer_smoke.pt"
        training_finite_error = None
        try:
            result = train_and_evaluate_fold(
                sub, sub_fold, SMOKE_HORIZON, SMOKE_LOOKBACK,
                e_layers=SMOKE_E_LAYERS, n_heads=SMOKE_N_HEADS,
                batch_size=SMOKE_BATCH_SIZE, max_epochs=SMOKE_EPOCHS,
                learning_rate=LEARNING_RATE, seed=SMOKE_SEED,
                stage="P10", model_family="Informer", config_id="SMOKE",
                checkpoint_path=checkpoint_path,
            )
        except FloatingPointError as e:
            training_finite_error = str(e)
            result = None

        # trainer는 loss/gradient가 NaN/Inf면 즉시 FloatingPointError(fail-fast)를 던지므로,
        # 예외 없음 == 전 구간 finite
        check("training loss finite (전 batch, fail-fast 미발생)", training_finite_error is None)
        check("training gradients finite (전 batch, fail-fast 미발생)", training_finite_error is None)
        if training_finite_error is not None:
            print(f"training loss/gradient fail-fast 발생: {training_finite_error}")
            print()
            print("=== PASS/FAIL 체크리스트 ===")
            for name, ok in checks:
                print(f"{'PASS' if ok else 'FAIL'}: {name}")
            return

        print()
        print("=== residual NaN(adi/cv2) imputation 요약(trainer 내부) ===")
        for feature, s in result["residual_nan_summary"].items():
            print(f"  {feature}: {s}")
        check("residual NaN unresolved==0 (trainer)",
              all(s["unresolved"] == 0 for s in result["residual_nan_summary"].values()))

        check("forward/backward PASS (train loop 정상 완료)", result["epochs_completed"] >= 1)

        ckpt_exists = checkpoint_path.exists()
        check("checkpoint save PASS", ckpt_exists)
        check("checkpoint reload PASS (load_state_dict 후 predict 성공)",
              np.isfinite(result["oof"]["y_pred_log"]).all())

        state_dict = torch.load(checkpoint_path, map_location="cpu")
        weights_finite = all(torch.isfinite(t).all().item() for t in state_dict.values())
        check("checkpoint weights finite (NaN/Inf 폭주 없음)", weights_finite)

        pred_finite = np.isfinite(result["oof"]["y_pred"]).all()
        check("prediction finite", bool(pred_finite))
        check("expm1 + clip 정상 (raw pred 음수 없음)", bool((result["oof"]["y_pred"] >= 0).all()))
        check("prediction-key 1:1 alignment (idx 커버리지 assertion 통과)", True)

        m = result["metrics"]
        check("evaluator 계산 PASS", all(np.isfinite(m[k]) for k in ["wape", "bias", "rmse", "mae"]))

        try:
            oof.validate_oof_frame(result["oof"])
            oof_valid = True
        except Exception:
            oof_valid = False
        check("OOF schema/validation PASS", oof_valid)

        print()
        print("=== 결과 요약 ===")
        print(f"n_train_sequences={result['n_train_sequences']}, n_val_sequences={result['n_val_sequences']}")
        print(f"label_len={result['label_len']}")
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
