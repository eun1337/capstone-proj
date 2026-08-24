"""
dl_coverage_audit.py
LSTM/TFT/Informer가 P10(A센터, validation_year=2022)의 모든 validation origin을
horizon(1/2/4) x fold(Q1~Q4) x lookback(13/26) 조건에서 실제로 얼마나 커버하는지
학습 없이 전수 확인하는 coverage audit. 모델/공통 코드는 전혀 수정하지 않고 기존
production 경로(src.forecasting.common.folds, src.forecasting.deep_learning.common의
sequence_builder·structural_nan·preprocessing, lstm/common dataset, tft/dataset_adapter,
informer/dataset)를 그대로 import해서 사용한다. 학습(optimizer/epoch/HPO)과 성능 비교는
하지 않는다.

expected origin은 fold generator의 train_mask/val_mask를 그대로 source of truth로
쓴다(임의 날짜 재계산 금지). sequence_builder는 skip 이유를 별도로 저장하지 않으므로,
missing key의 이유(insufficient_contiguous_history / missing_target /
unresolved_nonfinite / other_unexpected)는 이 audit 스크립트 내부에서 원본 imputed
history를 다시 대조해 재구성한다(production builder 자체는 변경하지 않음).
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.common import folds as day3_folds
from src.forecasting.common.config import TARGET_COLS
from src.forecasting.common.data_loader import load_development
from src.forecasting.deep_learning.common.config import get_model_feature_roles
from src.forecasting.deep_learning.common.preprocessing import SequencePreprocessor
from src.forecasting.deep_learning.common.sequence_builder import (
    CENTER_COL,
    SKU_COL,
    WEEK_COL,
    build_sequences,
    fold_origin_key_set,
    split_batch_by_origin_keys,
)
from src.forecasting.deep_learning.common.structural_nan import (
    apply_residual_nan_medians,
    fit_residual_nan_medians,
)
from src.forecasting.deep_learning.informer.dataset import build_informer_tensors
from src.forecasting.deep_learning.tft.dataset_adapter import (
    build_tft_long_dataframe,
    build_training_dataset,
    build_validation_dataset,
)

VALIDATION_YEAR = 2022
HORIZONS = (1, 2, 4)
LOOKBACKS = (13, 26)
MODELS = ("LSTM", "TFT", "Informer")

OUTPUT_DIR = Path("outputs/audits")
SUMMARY_PATH = OUTPUT_DIR / "dl_coverage_summary.csv"
MISSING_KEYS_PATH = OUTPUT_DIR / "dl_coverage_missing_keys.csv"

KEY_COLS3 = [CENTER_COL, SKU_COL, WEEK_COL]


def _key_set(df: pd.DataFrame, cols=KEY_COLS3) -> set:
    return set(df[cols].itertuples(index=False, name=None))


def _compute_run_length_map(df: pd.DataFrame) -> dict:
    """sequence_builder.build_sequences와 동일한 run_length(연속 주간 카운트) 로직을
    audit에서 독립적으로 재구성한다(production 코드는 변경하지 않음)."""
    ordered = df.sort_values([CENTER_COL, SKU_COL, WEEK_COL])
    run_length_map = {}
    groups = ordered.groupby([CENTER_COL, SKU_COL], sort=False, observed=True)
    for (center_id, sku_id), grp in groups:
        weeks = grp[WEEK_COL].to_numpy()
        n = len(weeks)
        day_diff = np.full(n, np.nan)
        if n > 1:
            day_diff[1:] = (weeks[1:] - weeks[:-1]) / np.timedelta64(1, "D")
        run_length = np.ones(n, dtype=int)
        for i in range(1, n):
            run_length[i] = run_length[i - 1] + 1 if day_diff[i] == 7 else 1
        for w, rl in zip(weeks, run_length):
            run_length_map[(center_id, sku_id, pd.Timestamp(w))] = int(rl)
    return run_length_map


def _window_finite(df_indexed: pd.DataFrame, center_id, sku_id, week_st, lookback, tv_cols) -> bool:
    """key가 속한 (center,sku) 이력에서 week_st로 끝나는 lookback 윈도우가 실제로
    tv_cols 전부 finite인지 재확인한다(build_sequences와 동일한 판정 기준). df_indexed는
    [center_id, sku_id]로 set_index된 df이며 week_st는 일반 컬럼으로 남아있다."""
    if (center_id, sku_id) not in df_indexed.index:
        return False
    grp = df_indexed.loc[[(center_id, sku_id)]].sort_values(WEEK_COL)
    weeks = grp[WEEK_COL].to_numpy()
    pos = np.searchsorted(weeks, np.datetime64(week_st))
    if pos >= len(weeks) or weeks[pos] != np.datetime64(week_st):
        return False
    start = pos - lookback + 1
    if start < 0:
        return False
    window = grp.iloc[start: pos + 1][tv_cols].to_numpy(dtype=float)
    return bool(np.isfinite(window).all())


def classify_missing_reason(target_lookup: pd.Series, lookback, key, run_length_map, df_indexed, tv_cols) -> str:
    """target_lookup은 [center_id,sku_id,week_st] MultiIndex Series(target_col 값)로,
    fold당 한 번만 만들어 재사용한다 - 매 missing key마다 전체 df를 boolean mask로
    스캔하면 missing key 수 x N(전체 row 수)만큼 비용이 들어 audit 자체가 병목이 되므로
    인덱스 기반 조회로 바꿈(production sequence_builder/structural_nan은 변경하지 않음)."""
    center_id, sku_id, week_st = key
    try:
        target_val = target_lookup.loc[(center_id, sku_id, week_st)]
    except KeyError:
        return "other_unexpected"
    if pd.isna(target_val):
        return "missing_target"

    run_length = run_length_map.get((center_id, sku_id, pd.Timestamp(week_st)))
    if run_length is None or run_length < lookback:
        return "insufficient_contiguous_history"

    if not _window_finite(df_indexed, center_id, sku_id, week_st, lookback, tv_cols):
        return "unresolved_nonfinite"

    return "other_unexpected"


def _decode_tft_covered_keys(dataset, val_batch, group_offset: int) -> set:
    """TFT TimeSeriesDataSet이 실제로 만든 sample의 origin_id를 디코드해 원본
    (center_id, sku_id, week_st) key set으로 되돌린다. len(dataset)==len(val_batch)일
    때는 구조적으로 1:1이 보장되므로(직접 검증됨) 호출부에서 이 함수를 건너뛰고
    generated_val_keys를 그대로 재사용해도 된다 - 개수가 다를 때만 필요한 진단 경로."""
    enc = dataset._categorical_encoders["__group_id__origin_id"]
    groups_encoded = dataset.data["groups"][:, 0].numpy()
    decoded = enc.inverse_transform(groups_encoded)
    covered_positions = sorted(set(int(v) - group_offset for v in decoded))
    keys = set()
    for pos in covered_positions:
        if 0 <= pos < len(val_batch.target):
            row = val_batch.keys.iloc[pos]
            keys.add((row[CENTER_COL], row[SKU_COL], row[WEEK_COL]))
    return keys


def main() -> None:
    t_audit_start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dev = load_development()
    sub_a = dev[dev["center_id"] == "A"].copy()

    summary_rows = []
    missing_rows = []

    for horizon in HORIZONS:
        target_col = TARGET_COLS[horizon]
        roles = get_model_feature_roles(horizon)
        tv_cols = list(roles["time_varying_known"]) + list(roles["time_varying_observed"])

        folds = day3_folds.generate_expanding_folds(sub_a, VALIDATION_YEAR, horizon)
        for fold in folds:
            fold_id = fold["fold"]
            t_fold = time.perf_counter()
            print(f"[h{horizon} fold{fold_id}] structural NaN fit/apply...", flush=True)

            expected_train_keys = fold_origin_key_set(sub_a, fold["train_mask"])
            expected_val_keys = fold_origin_key_set(sub_a, fold["val_mask"])

            try:
                residual_maps = fit_residual_nan_medians(sub_a.loc[fold["train_mask"]])
                sub_imputed, residual_summary = apply_residual_nan_medians(sub_a, residual_maps)
                residual_unresolved = sum(s["unresolved"] for s in residual_summary.values())
            except ValueError as e:
                print(f"  [FAIL] structural NaN 처리 실패: {e}", flush=True)
                for lookback in LOOKBACKS:
                    for model in MODELS:
                        summary_rows.append({
                            "model": model, "horizon": horizon, "fold": fold_id, "lookback": lookback,
                            "expected_train_origins": len(expected_train_keys),
                            "expected_val_origins": len(expected_val_keys),
                            "error": str(e), "audit_pass": False,
                        })
                continue

            run_length_map = _compute_run_length_map(sub_imputed)
            df_indexed_for_window = sub_imputed.set_index([CENTER_COL, SKU_COL]).sort_index()
            target_lookup = sub_imputed.set_index([CENTER_COL, SKU_COL, WEEK_COL])[target_col].sort_index()

            for lookback in LOOKBACKS:
                t_combo = time.perf_counter()
                print(f"  [lookback={lookback}] build_sequences...", flush=True)
                try:
                    full_batch = build_sequences(sub_imputed, horizon, lookback)
                except ValueError as e:
                    print(f"    [FAIL] build_sequences 실패: {e}", flush=True)
                    for model in MODELS:
                        summary_rows.append({
                            "model": model, "horizon": horizon, "fold": fold_id, "lookback": lookback,
                            "expected_train_origins": len(expected_train_keys),
                            "expected_val_origins": len(expected_val_keys),
                            "error": str(e), "audit_pass": False,
                        })
                    continue

                train_batch = split_batch_by_origin_keys(full_batch, expected_train_keys)
                val_batch = split_batch_by_origin_keys(full_batch, expected_val_keys)

                generated_train_keys = _key_set(train_batch.keys)
                generated_val_keys = _key_set(val_batch.keys)

                missing_train_keys = expected_train_keys - generated_train_keys
                missing_val_keys = expected_val_keys - generated_val_keys
                unexpected_extra_train = generated_train_keys - expected_train_keys
                unexpected_extra_val = generated_val_keys - expected_val_keys

                reason_counts = {"insufficient_contiguous_history": 0, "missing_target": 0,
                                  "unresolved_nonfinite": 0, "other_unexpected": 0}
                for key in missing_val_keys:
                    reason = classify_missing_reason(
                        target_lookup, lookback, key, run_length_map, df_indexed_for_window, tv_cols,
                    )
                    reason_counts[reason] += 1
                    missing_rows.append({
                        "model": "common", "horizon": horizon, "fold": fold_id, "lookback": lookback,
                        "center_id": key[0], "sku_id": key[1], "week_st": key[2],
                        "target_date": key[2] + pd.Timedelta(weeks=horizon), "reason": reason,
                    })

                train_nonfinite = int((~np.isfinite(train_batch.time_varying)).sum()
                                       + (~np.isfinite(train_batch.static_cont)).sum())
                val_nonfinite = int((~np.isfinite(val_batch.time_varying)).sum()
                                     + (~np.isfinite(val_batch.static_cont)).sum())
                target_nonfinite = int((~np.isfinite(val_batch.target)).sum())

                common_stats = {
                    "expected_train_origins": len(expected_train_keys),
                    "generated_train_sequences": len(train_batch.target),
                    "train_missing_count": len(missing_train_keys),
                    "train_coverage_rate": (len(train_batch.target) / len(expected_train_keys)) if expected_train_keys else np.nan,
                    "expected_val_origins": len(expected_val_keys),
                    "generated_val_sequences": len(val_batch.target),
                    "val_missing_count": len(missing_val_keys),
                    "val_coverage_rate": (len(val_batch.target) / len(expected_val_keys)) if expected_val_keys else np.nan,
                    "insufficient_contiguous_history": reason_counts["insufficient_contiguous_history"],
                    "missing_target": reason_counts["missing_target"],
                    "unresolved_nonfinite": reason_counts["unresolved_nonfinite"],
                    "other_unexpected": reason_counts["other_unexpected"],
                    "unexpected_extra_keys": len(unexpected_extra_train) + len(unexpected_extra_val),
                    "train_input_nonfinite_count": train_nonfinite,
                    "val_input_nonfinite_count": val_nonfinite,
                    "target_nonfinite_count": target_nonfinite,
                    "residual_nan_fit_train_only": True,
                    "residual_nan_unresolved_before_sequence": residual_unresolved,
                }

                # --- LSTM adapter ---
                print(f"    [LSTM] preprocessing/dataset...", flush=True)
                preprocessor = SequencePreprocessor().fit(train_batch)
                lstm_val_nonfinite = 0
                lstm_dropped = 0
                try:
                    lstm_train_t = preprocessor.transform(train_batch)
                    lstm_val_t = preprocessor.transform(val_batch)
                    lstm_output_count = len(lstm_val_t["target"])
                    lstm_input_count = len(val_batch.target)
                except ValueError as e:
                    lstm_output_count = 0
                    lstm_input_count = len(val_batch.target)
                    lstm_val_nonfinite = lstm_input_count
                lstm_dropped = lstm_input_count - lstm_output_count
                lstm_val_keys = generated_val_keys if lstm_dropped == 0 else set()

                summary_rows.append({
                    "model": "LSTM", "horizon": horizon, "fold": fold_id, "lookback": lookback,
                    **common_stats,
                    "adapter_input_count": lstm_input_count,
                    "adapter_output_count": lstm_output_count,
                    "adapter_dropped_count": lstm_dropped,
                    "adapter_nonfinite_count": lstm_val_nonfinite,
                    "audit_pass": (lstm_dropped == 0 and lstm_val_nonfinite == 0
                                   and len(unexpected_extra_train) == 0 and len(unexpected_extra_val) == 0
                                   and reason_counts["unresolved_nonfinite"] == 0
                                   and reason_counts["other_unexpected"] == 0),
                })

                # --- Informer adapter ---
                print(f"    [Informer] tensor build...", flush=True)
                informer_dropped = 0
                informer_nonfinite = 0
                try:
                    informer_train_t = build_informer_tensors(preprocessor, train_batch, sub_imputed, horizon)
                    informer_val_t = build_informer_tensors(preprocessor, val_batch, sub_imputed, horizon)
                    informer_output_count = len(informer_val_t["target"])
                    informer_input_count = len(val_batch.target)
                except (ValueError, KeyError) as e:
                    informer_output_count = 0
                    informer_input_count = len(val_batch.target)
                    informer_nonfinite = informer_input_count
                informer_dropped = informer_input_count - informer_output_count
                informer_val_keys = generated_val_keys if informer_dropped == 0 else set()

                summary_rows.append({
                    "model": "Informer", "horizon": horizon, "fold": fold_id, "lookback": lookback,
                    **common_stats,
                    "adapter_input_count": informer_input_count,
                    "adapter_output_count": informer_output_count,
                    "adapter_dropped_count": informer_dropped,
                    "adapter_nonfinite_count": informer_nonfinite,
                    "audit_pass": (informer_dropped == 0 and informer_nonfinite == 0
                                   and len(unexpected_extra_train) == 0 and len(unexpected_extra_val) == 0
                                   and reason_counts["unresolved_nonfinite"] == 0
                                   and reason_counts["other_unexpected"] == 0),
                })

                # --- TFT adapter ---
                print(f"    [TFT] long dataframe + TimeSeriesDataSet (느림)...", flush=True)
                tft_dropped = 0
                tft_nonfinite = 0
                try:
                    train_long = build_tft_long_dataframe(train_batch, sub_imputed, horizon, group_offset=0)
                    val_long = build_tft_long_dataframe(val_batch, sub_imputed, horizon, group_offset=len(train_batch.target))
                    training_dataset = build_training_dataset(train_long, horizon, lookback)
                    validation_dataset = build_validation_dataset(training_dataset, val_long)
                    tft_output_count = len(validation_dataset)
                    tft_input_count = len(val_batch.target)
                    tft_dropped = tft_input_count - tft_output_count
                    if tft_dropped == 0:
                        tft_val_keys = generated_val_keys
                    else:
                        tft_val_keys = _decode_tft_covered_keys(validation_dataset, val_batch, len(train_batch.target))
                    if not np.isfinite(val_long.select_dtypes("number").to_numpy()).all():
                        tft_nonfinite = 1
                except (ValueError, KeyError) as e:
                    tft_output_count = 0
                    tft_input_count = len(val_batch.target)
                    tft_dropped = tft_input_count
                    tft_val_keys = set()
                    tft_nonfinite = tft_input_count

                summary_rows.append({
                    "model": "TFT", "horizon": horizon, "fold": fold_id, "lookback": lookback,
                    **common_stats,
                    "adapter_input_count": tft_input_count,
                    "adapter_output_count": tft_output_count,
                    "adapter_dropped_count": tft_dropped,
                    "adapter_nonfinite_count": tft_nonfinite,
                    "audit_pass": (tft_dropped == 0 and tft_nonfinite == 0
                                   and len(unexpected_extra_train) == 0 and len(unexpected_extra_val) == 0
                                   and reason_counts["unresolved_nonfinite"] == 0
                                   and reason_counts["other_unexpected"] == 0),
                })

                # --- 모델 간 key-set 비교 ---
                lstm_vs_tft = lstm_val_keys.symmetric_difference(tft_val_keys)
                lstm_vs_informer = lstm_val_keys.symmetric_difference(informer_val_keys)
                tft_vs_informer = tft_val_keys.symmetric_difference(informer_val_keys)
                for diff_name, diff_set in (
                    ("LSTM_vs_TFT", lstm_vs_tft),
                    ("LSTM_vs_Informer", lstm_vs_informer),
                    ("TFT_vs_Informer", tft_vs_informer),
                ):
                    for key in diff_set:
                        missing_rows.append({
                            "model": diff_name, "horizon": horizon, "fold": fold_id, "lookback": lookback,
                            "center_id": key[0], "sku_id": key[1], "week_st": key[2],
                            "target_date": key[2] + pd.Timedelta(weeks=horizon), "reason": "model_key_diff",
                        })

                print(f"    combo 완료: {time.perf_counter()-t_combo:.1f}s "
                      f"(val expected={len(expected_val_keys)}, generated={len(val_batch.target)})", flush=True)

            print(f"[h{horizon} fold{fold_id}] 완료: {time.perf_counter()-t_fold:.1f}s", flush=True)

    summary_df = pd.DataFrame(summary_rows)
    missing_df = pd.DataFrame(missing_rows)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    missing_df.to_csv(MISSING_KEYS_PATH, index=False)

    print()
    print(f"총 소요시간: {time.perf_counter()-t_audit_start:.1f}s")
    print(f"summary rows: {len(summary_df)} -> {SUMMARY_PATH}")
    print(f"missing/diff rows: {len(missing_df)} -> {MISSING_KEYS_PATH}")


if __name__ == "__main__":
    main()
