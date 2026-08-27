# -*- coding: utf-8 -*-
"""
run_diagnostic.py

진단용 1회성 실험: log1p -> expm1 재변환 편향(retransformation bias, Jensen's inequality)
가설을 LSTM(신경망 계열)에서도 확인한다.

이전 lightgbm h1 진단(outputs/hpo/_diag_rawscale_lightgbm_h1/)에서:
  - log1p (기존, trial3 best): pooled_wape=59.50, pooled_bias=-31.63%
  - raw scale(동일 trial): pooled_wape=67.84, pooled_bias=+3.45% (부호 반전)
결과가 나와 재변환 편향 가설이 강하게 지지되었다. 이번엔 완전히 다른 알고리즘 계열인
LSTM에서 동일한 feature/fold(4-fold expanding)/hyperparameter(2 trial grid, lookback=13,
max_epochs=20)/evaluation population(5-family common evaluation keys)을 유지한 채,
target만 log1p 변환 없이 raw scale(qty)로 MSE 학습해서 pooled_bias가 동일하게 줄어드는지
확인한다.

기존 파이프라인 코드(deep_learning/lstm/trainer.py, common/evaluator.py,
pipeline/p13/lstm.py, pipeline/p13/hpo_common.py 등)는 전혀 수정하지 않고 import만
한다. 이 실험과 관련된 모든 산출물은 outputs/hpo/_diag_rawscale_lstm_h1/ 안에만
저장한다(체크포인트 포함).

주의: raw scale MSE 학습은 log1p 대비 타겟 스케일이 커서 발산 위험이 있다. 발산이
관찰되면 학습률/아키텍처를 임의로 바꿔 안정화하지 않고, 발산 사실과 epoch별
loss curve를 있는 그대로 결과에 기록한다.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# WSL->Windows python.exe 호출 시 PYTHONPATH가 전달되지 않아, repo root를 직접 sys.path에 추가한다.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.forecasting.common import config as cfg
from src.forecasting.common import evaluator as ev
from src.forecasting.common import folds as day3_folds
from src.forecasting.common import oof as oo
from src.forecasting.common.data_loader import load_development
from src.forecasting.deep_learning.common.dataset import SequenceDataset
from src.forecasting.deep_learning.common.preprocessing import SequencePreprocessor
from src.forecasting.deep_learning.common.sequence_builder import (
    build_sequences,
    fold_origin_key_set,
    split_batch_by_origin_keys,
)
from src.forecasting.deep_learning.common.structural_nan import (
    apply_residual_nan_medians,
    fit_residual_nan_medians,
)
from src.forecasting.deep_learning.lstm.config import DROPOUT, NUM_LAYERS
from src.forecasting.deep_learning.lstm.model import LSTMForecaster
from src.forecasting.pipeline.p13 import hpo_common as hc

# deep_learning/lstm/trainer.py와 benchmarks/p12/lstm_profile.py는 import chain 상에서
# benchmarks/p12/common.py -> `import resource`(POSIX 전용, RAM 추적용)를 거치는데,
# 이 프로젝트의 실제 실행 인터프리터(Windows python.exe)에는 `resource` 모듈이 없어
# 그 체인 전체가 import 시점에 실패한다(원래 LSTM P13 결과는 다른 머신에서 생성된 것으로
# 보임). 여기서 필요한 건 그 파일들의 사소한 helper 2개뿐이라, 깨진 import chain을 우회
# 하려고 sys.modules를 조작하는 대신 trainer.py/lstm_profile.py의 정의를 그대로 옮겨 쓴다
# (내용을 바꾸는 게 아니라 동일 로직을 복제 - 두 파일 모두 수정하지 않음).
import random


def select_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


FALLBACK_BATCH_SIZE = 512
_OOM_SIGNATURES = ("out of memory", "cannot allocate memory", "mps backend out of memory",
                    "unable to allocate", "not enough memory")


def _is_oom_error(exc: Exception) -> bool:
    if isinstance(exc, MemoryError):
        return True
    if isinstance(exc, RuntimeError):
        return any(sig in str(exc).lower() for sig in _OOM_SIGNATURES)
    return False

HORIZON = 1
FAMILY = "lstm"
STAGE = "p13_diag_rawscale_h1"
SEED = cfg.P13_MODEL_SEED  # 42

OUT_DIR = _REPO_ROOT / "outputs" / "hpo" / "_diag_rawscale_lstm_h1"
OOF_DIR = OUT_DIR / "oof"
CHECKPOINT_DIR = OUT_DIR / "_checkpoints"

EXISTING_RESULT_PATH = cfg.PROJECT_ROOT / "outputs" / "hpo" / "lstm" / "p13_lstm_h1.json"
COMMON_EVAL_KEYS_PATH = cfg.PROJECT_ROOT / "outputs" / "audits" / "p13" / "five_family_common_evaluation_keys.parquet"


def train_and_evaluate_fold_raw(
    df: pd.DataFrame,
    fold: dict,
    horizon: int,
    lookback: int,
    *,
    hidden_size: int,
    batch_size: int,
    max_epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    stage: str,
    model_family: str,
    config_id: str,
    checkpoint_path,
) -> dict:
    """deep_learning/lstm/trainer.py의 train_and_evaluate_fold와 동일 구조이지만,
    target을 log1p 변환하지 않고 raw scale(qty)로 그대로 MSE 학습/평가한다
    (evaluator.inverse_transform_prediction 미호출). epoch별 train/val loss를
    기록하고, NaN/Inf가 발생하면 즉시 학습을 중단하고 발산 사실을 그대로 반환한다."""
    device = select_device()

    set_all_seeds(seed)
    shuffle_generator = torch.Generator()
    shuffle_generator.manual_seed(seed)
    fold_id = fold["fold"]

    residual_nan_maps = fit_residual_nan_medians(df.loc[fold["train_mask"]])
    df_imputed, residual_nan_summary = apply_residual_nan_medians(df, residual_nan_maps)
    full_batch = build_sequences(df_imputed, horizon, lookback)
    train_keys = fold_origin_key_set(df, fold["train_mask"])
    val_keys = fold_origin_key_set(df, fold["val_mask"])
    train_batch = split_batch_by_origin_keys(full_batch, train_keys)
    val_batch = split_batch_by_origin_keys(full_batch, val_keys)

    preprocessor = SequencePreprocessor().fit(train_batch)
    train_transformed = preprocessor.transform(train_batch)
    val_transformed = preprocessor.transform(val_batch)

    train_dataset = SequenceDataset(train_transformed)
    val_dataset = SequenceDataset(val_transformed)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, generator=shuffle_generator)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    n_time_varying = train_batch.time_varying.shape[-1]
    static_cont_dim = train_batch.static_cont.shape[-1]
    vocab_sizes = preprocessor.vocab_sizes()

    model = LSTMForecaster(
        n_time_varying=n_time_varying,
        static_cont_dim=static_cont_dim,
        cat_vocab_sizes=vocab_sizes,
        hidden_size=hidden_size,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_fn = nn.MSELoss()

    train_loss_curve, val_loss_curve = [], []
    diverged = False
    divergence_epoch = None
    divergence_reason = None
    epochs_completed = 0

    t0 = time.perf_counter()
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_losses = []
        for batch in train_loader:
            tv = batch["time_varying"].to(device)
            sc = batch["static_cont"].to(device)
            scat = batch["static_cat"].to(device)
            y_raw = batch["target"].to(device)  # <- 핵심 변경점: log1p 미적용, raw scale 그대로

            optimizer.zero_grad()
            pred_raw = model(tv, sc, scat)
            loss = loss_fn(pred_raw, y_raw)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item() * len(y_raw))
        train_loss_epoch = sum(train_losses) / len(train_dataset)
        train_loss_curve.append(train_loss_epoch)

        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                tv = batch["time_varying"].to(device)
                sc = batch["static_cont"].to(device)
                scat = batch["static_cat"].to(device)
                y_raw = batch["target"].to(device)
                pred_raw = model(tv, sc, scat)
                val_losses.append(loss_fn(pred_raw, y_raw).item() * len(y_raw))
        val_loss_epoch = sum(val_losses) / len(val_dataset)
        val_loss_curve.append(val_loss_epoch)
        epochs_completed = epoch

        # 발산 감지: loss가 NaN/Inf이거나 파라미터에 NaN/Inf가 생기면 즉시 중단한다
        # (임의로 학습률/아키텍처를 바꿔 억지로 수렴시키지 않는다 - 진단 결과 그대로 기록).
        params_finite = all(torch.isfinite(p).all() for p in model.parameters())
        if not np.isfinite(train_loss_epoch) or not np.isfinite(val_loss_epoch) or not params_finite:
            diverged = True
            divergence_epoch = epoch
            divergence_reason = (
                f"train_loss={train_loss_epoch}, val_loss={val_loss_epoch}, "
                f"params_finite={params_finite}"
            )
            break
    train_time_sec = time.perf_counter() - t0

    if diverged:
        return {
            "status": "diverged",
            "fold_id": fold_id,
            "divergence_epoch": divergence_epoch,
            "divergence_reason": divergence_reason,
            "train_loss_curve": train_loss_curve,
            "val_loss_curve": val_loss_curve,
            "epochs_completed": epochs_completed,
            "train_time_sec": train_time_sec,
        }

    torch.save(model.state_dict(), checkpoint_path)
    model.eval()

    pred_raw_list = []
    with torch.no_grad():
        for batch in val_loader:
            tv = batch["time_varying"].to(device)
            sc = batch["static_cont"].to(device)
            scat = batch["static_cat"].to(device)
            pred_raw_list.append(model(tv, sc, scat).cpu().numpy())
    pred_raw = np.concatenate(pred_raw_list)

    n_negative_clipped = int((pred_raw < 0).sum())
    pred_raw = np.clip(pred_raw, 0.0, None)

    y_val_true = val_batch.target
    oof_keys = val_batch.keys[["center_id", "sku_id", "week_st", "target_date"]].copy()
    history_df = df.loc[fold["train_mask"], ["center_id", "sku_id", "week_st", "qty"]]
    mase_scale = ev.build_mase_scale(history_df, oof_keys)
    metrics = ev.compute_metrics(y_val_true, pred_raw, mase_scale)

    # 이 실험엔 log-scale 예측이 없으므로 y_pred_log 컬럼은 raw 예측을 그대로 채운다
    # (진단용 메타데이터일 뿐 metric 계산엔 사용되지 않음).
    oof_frame = oo.build_oof_frame(
        oof_keys, y_val_true, pred_raw, pred_raw, mase_scale,
        stage=stage, model_family=model_family, config_id=config_id,
        seed=seed, horizon=horizon, fold_id=fold_id,
    )

    return {
        "status": "ok",
        "fold_id": fold_id,
        "metrics": metrics,
        "oof": oof_frame,
        "epochs_completed": epochs_completed,
        "train_loss_curve": train_loss_curve,
        "val_loss_curve": val_loss_curve,
        "n_negative_clipped": n_negative_clipped,
        "n_train_sequences": len(train_batch.target),
        "n_val_sequences": len(val_batch.target),
        "train_time_sec": train_time_sec,
        "device": device,
        "residual_nan_summary": residual_nan_summary,
    }


def _attempt(hp: dict, batch_size: int, sub_a: pd.DataFrame, fold: dict, horizon: int,
             seed: int, stage: str, config_id: str, checkpoint_path: Path) -> dict:
    return train_and_evaluate_fold_raw(
        sub_a, fold, horizon, hp["lookback"],
        hidden_size=hp["hidden_size"], batch_size=batch_size, max_epochs=hp["max_epochs"],
        learning_rate=hp["learning_rate"], weight_decay=hp["weight_decay"],
        seed=seed, stage=stage, model_family=FAMILY.upper(), config_id=config_id,
        checkpoint_path=str(checkpoint_path),
    )


def _run_folds_raw(hp: dict, sub_a: pd.DataFrame, folds: list, horizon: int, seed: int,
                    stage: str, trial_label: str) -> dict:
    """p13/lstm.py의 _run_folds와 동일 구조(OOM 1회 fallback 포함)이나,
    fold가 발산(diverged)해도 나머지 fold는 계속 진행해서 fold별 loss curve를
    최대한 수집한다(임의 안정화 없이 사실만 기록)."""
    oof_frames, fold_metrics, fallback_flags = [], [], []
    diverged_folds = []
    loss_curves_by_fold = {}

    for fold in folds:
        fold_id = fold["fold"]
        config_id = f"{trial_label}_fold{fold_id}"
        checkpoint_path = CHECKPOINT_DIR / f"{trial_label}_fold{fold_id}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        batch_size_used = hp["batch_size"]
        fallback_occurred = False
        try:
            result = _attempt(hp, batch_size_used, sub_a, fold, horizon, seed, stage, config_id, checkpoint_path)
        except Exception as exc:  # noqa: BLE001
            if not _is_oom_error(exc):
                return {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "fold_id": fold_id}
            fallback_occurred = True
            batch_size_used = FALLBACK_BATCH_SIZE
            try:
                result = _attempt(hp, batch_size_used, sub_a, fold, horizon, seed, stage, config_id, checkpoint_path)
            except Exception as exc2:  # noqa: BLE001
                return {
                    "status": "failed", "fold_id": fold_id,
                    "error": f"1차(batch={hp['batch_size']}) OOM: {exc} | fallback(batch={FALLBACK_BATCH_SIZE}) "
                             f"실패: {type(exc2).__name__}: {exc2}",
                }

        loss_curves_by_fold[fold_id] = {
            "train_loss_curve": result["train_loss_curve"],
            "val_loss_curve": result["val_loss_curve"],
            "epochs_completed": result["epochs_completed"],
        }

        if result["status"] == "diverged":
            diverged_folds.append({
                "fold_id": fold_id,
                "divergence_epoch": result["divergence_epoch"],
                "divergence_reason": result["divergence_reason"],
            })
            continue

        oof_frames.append(result["oof"])
        fold_metrics.append(result["metrics"])
        fallback_flags.append(fallback_occurred)

    if diverged_folds and not oof_frames:
        return {
            "status": "diverged", "diverged_folds": diverged_folds,
            "loss_curves_by_fold": loss_curves_by_fold,
        }
    if diverged_folds:
        return {
            "status": "partially_diverged", "diverged_folds": diverged_folds,
            "loss_curves_by_fold": loss_curves_by_fold,
            "oof_frames": oof_frames, "native_fold_metrics": fold_metrics,
            "fallback_occurred_per_fold": fallback_flags,
        }
    return {
        "status": "ok", "oof_frames": oof_frames,
        "native_fold_metrics": fold_metrics, "fallback_occurred_per_fold": fallback_flags,
        "loss_curves_by_fold": loss_curves_by_fold,
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    OOF_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    common_eval_keys = hc.load_common_eval_keys(COMMON_EVAL_KEYS_PATH)

    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()

    folds = day3_folds.generate_expanding_folds(sub_a, 2023, HORIZON)
    for fold in folds:
        if int(fold["train_mask"].sum()) == 0 or int(fold["val_mask"].sum()) == 0:
            raise ValueError(f"h{HORIZON} fold {fold['fold']}: train/val row 0개")

    hc.check_common_keys_match_p13_period(common_eval_keys, folds, HORIZON, COMMON_EVAL_KEYS_PATH)
    horizon_common_keys = common_eval_keys[common_eval_keys["horizon"] == HORIZON]

    with open(EXISTING_RESULT_PATH, "r", encoding="utf-8") as f:
        existing_result = json.load(f)
    existing_trials = existing_result["all_trials"]

    print(f"[diag][lstm-rawscale] h{HORIZON} 시작: {len(existing_trials)}개 trial (기존 grid 재사용), "
          f"common_eval_keys={COMMON_EVAL_KEYS_PATH}")

    trial_summaries = []
    filtered_oof_by_trial = {}

    for existing_trial in existing_trials:
        trial_id = existing_trial["trial_id"]
        hp = dict(existing_trial["params"])

        print(f"[diag][lstm-rawscale] trial{trial_id} params={hp} 학습 시작...")
        run = _run_folds_raw(hp, sub_a, folds, HORIZON, SEED, STAGE, f"h{HORIZON}_trial{trial_id}")

        if run["status"] == "failed":
            trial_summaries.append({
                "trial_id": trial_id, "params": hp, "status": "failed",
                "error": run["error"], "fold_id": run["fold_id"],
                "pooled_wape": float("inf"), "pooled_bias": float("nan"),
                "worst_fold_wape": float("inf"),
            })
            print(f"[diag][lstm-rawscale] trial{trial_id} 실패: {run['error']}")
            continue

        if run["status"] == "diverged":
            trial_summaries.append({
                "trial_id": trial_id, "params": hp, "status": "diverged",
                "diverged_folds": run["diverged_folds"],
                "loss_curves_by_fold": run["loss_curves_by_fold"],
                "pooled_wape": float("inf"), "pooled_bias": float("nan"),
                "worst_fold_wape": float("inf"),
            })
            print(f"[diag][lstm-rawscale] trial{trial_id} 전체 fold 발산: {run['diverged_folds']}")
            continue

        # status ok 또는 partially_diverged
        pooled_native_oof = pd.concat(run["oof_frames"], ignore_index=True)
        filtered_oof = hc.filter_pooled_oof_by_common_keys(pooled_native_oof, horizon_common_keys)
        pooled_common = ev.compute_metrics(
            filtered_oof["y_true"].to_numpy(), filtered_oof["y_pred"].to_numpy(),
            filtered_oof["mase_scale"].to_numpy(),
        )
        per_fold_common = hc.per_fold_metrics_common(filtered_oof)
        worst_fold_wape = max(m["wape"] for m in per_fold_common.values())

        trial_summary = {
            "trial_id": trial_id, "params": hp, "status": run["status"],  # "ok" 또는 "partially_diverged"
            "pooled_wape": pooled_common["wape"], "pooled_bias": pooled_common["bias"],
            "pooled_rmse": pooled_common["rmse"], "pooled_mae": pooled_common["mae"],
            "pooled_mase": pooled_common["mase"], "worst_fold_wape": worst_fold_wape,
            "per_fold_metrics_common": per_fold_common,
            "native_fold_metrics_diagnostic": run["native_fold_metrics"],
            "fallback_occurred_per_fold": run["fallback_occurred_per_fold"],
            "fallback_occurred": any(run["fallback_occurred_per_fold"]),
            "loss_curves_by_fold": run["loss_curves_by_fold"],
            "common_eval_key_count": len(horizon_common_keys),
        }
        if run["status"] == "partially_diverged":
            trial_summary["diverged_folds"] = run["diverged_folds"]
            trial_summary["note"] = (
                "일부 fold가 발산해 해당 fold는 pooled metric 계산에서 제외됨 - "
                "5-family common evaluation population(577,319 키)을 전부 커버하지 못했을 수 있음, "
                "다른 trial과 직접 비교 시 주의 필요"
            )
        trial_summaries.append(trial_summary)
        filtered_oof_by_trial[trial_id] = filtered_oof
        print(f"[diag][lstm-rawscale] trial{trial_id} 완료(status={run['status']}): "
              f"WAPE={pooled_common['wape']:.4f} Bias={pooled_common['bias']:.4f}")

    ok_trials = [t for t in trial_summaries if t["status"] in ("ok", "partially_diverged")]
    selection = hc.select_best_trial(ok_trials) if ok_trials else {
        "selected": None, "reason": "no_trial_completed_training",
        "all_trials_sorted_by_wape": sorted(trial_summaries, key=lambda t: t["pooled_wape"]),
    }

    result = {
        "horizon": HORIZON,
        "n_trials": len(existing_trials),
        "seed": SEED,
        "sampler_seed": existing_result.get("sampler_seed"),
        "n_failed_trials": sum(1 for t in trial_summaries if t["status"] not in ("ok", "partially_diverged")),
        "evaluation_population": "five_family_common_keys",
        "common_eval_key_path": str(COMMON_EVAL_KEYS_PATH),
        "common_eval_key_count": len(horizon_common_keys),
        "lookback_fixed": existing_result.get("lookback_fixed"),
        "max_epochs_fixed": existing_result.get("max_epochs_fixed"),
        "target_transform": "raw_scale_no_log1p",
        "diagnostic_note": (
            "log1p/expm1 재변환 편향(retransformation bias) 가설 검증용 1회성 진단 스크립트. "
            "outputs/hpo/lstm/p13_lstm_h1.json과 동일한 feature/fold/hyperparameter grid/"
            "evaluation population을 사용하되, target을 log1p 변환하지 않고 raw scale(qty)로 "
            "MSE 학습함(inverse_transform_prediction 미호출). 발산 시 임의로 학습률/아키텍처를 "
            "바꾸지 않고 사실 그대로 기록함."
        ),
        "reference_result_path": str(EXISTING_RESULT_PATH),
        "selection": {k: v for k, v in selection.items()},
        "all_trials": trial_summaries,
    }

    sel = selection["selected"]
    if sel is not None:
        result["pooled_metrics_common"] = {
            "wape": sel["pooled_wape"], "bias": sel["pooled_bias"], "rmse": sel["pooled_rmse"],
            "mae": sel["pooled_mae"], "mase": sel["pooled_mase"],
        }
        result["per_fold_metrics_common"] = sel["per_fold_metrics_common"]
        result["native_fold_metrics_diagnostic"] = sel["native_fold_metrics_diagnostic"]
        result["fallback_occurred"] = sel["fallback_occurred"]
        filtered_oof_by_trial[sel["trial_id"]].to_parquet(
            OOF_DIR / f"oof_{FAMILY}_h{HORIZON}_trial{sel['trial_id']}_seed{SEED}_common_rawscale.parquet",
            index=False,
        )

    out_path = OUT_DIR / "p13_lstm_h1_rawscale.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(f"[diag][lstm-rawscale] 저장 완료: {out_path}")

    # 기존 두 trial과 동일 trial_id로 매칭해 비교 리포트 출력
    print("\n=== log1p 재변환 편향 가설 검증 결과 (lstm h1) ===")
    header = f"| {'구분':<34} | {'pooled_wape':>12} | {'pooled_bias':>12} |"
    print(header)
    print("|" + "-" * (len(header) - 2) + "|")

    for existing_trial in sorted(existing_trials, key=lambda t: t["trial_id"]):
        trial_id = existing_trial["trial_id"]
        print(f"| {'기존 (log1p, trial' + str(trial_id) + ')':<34} "
              f"| {existing_trial['pooled_wape']:>12.3f} | {existing_trial['pooled_bias']:>12.2f} |")
        new_matched = next((t for t in trial_summaries if t["trial_id"] == trial_id), None)
        if new_matched is not None and new_matched["status"] in ("ok", "partially_diverged"):
            print(f"| {'이번 (raw scale, trial' + str(trial_id) + ')':<34} "
                  f"| {new_matched['pooled_wape']:>12.3f} | {new_matched['pooled_bias']:>12.2f} |")
            bias_delta = new_matched["pooled_bias"] - existing_trial["pooled_bias"]
            sign_flip = (existing_trial["pooled_bias"] < 0 < new_matched["pooled_bias"]) or \
                        (existing_trial["pooled_bias"] > 0 > new_matched["pooled_bias"])
            print(f"    -> bias 변화량: {bias_delta:+.2f}%p (부호 반전: {sign_flip}, status={new_matched['status']})")
        else:
            status = new_matched["status"] if new_matched is not None else "매칭 실패"
            print(f"| {'이번 (raw scale, trial' + str(trial_id) + ')':<34} "
                  f"| {'N/A (' + status + ')':>12} | {'N/A':>12} |")
            if new_matched is not None and status == "diverged":
                print(f"    -> 발산 상세: {new_matched['diverged_folds']}")

    print(f"\n결과 파일: {out_path}")
    print(f"체크포인트/로그 등 모든 산출물: {OUT_DIR}")


if __name__ == "__main__":
    main()
