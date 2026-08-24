"""
family_registry.py

P13 이후 단계에서 사용하는 model family 공통 인터페이스.

family별 P13 실행, Final retrain, 2024 prediction 경로를 연결해
seed robustness, B robustness와 이후 pipeline에서 동일한 registry를 재사용한다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.forecasting.common import folds as day3_folds

# macOS OpenMP 충돌 방지를 위해 torch 계열을 LightGBM보다 먼저 import한다.
import src.forecasting.pipeline.p13.informer as p13_informer
import src.forecasting.pipeline.p13.lstm as p13_lstm
import src.forecasting.pipeline.p13.tft as p13_tft
import src.forecasting.pipeline.p13.lightgbm as p13_lgbm
import src.forecasting.pipeline.p13.rf as p13_rf

FAMILIES = ("rf", "lightgbm", "lstm", "tft", "informer")
ROW_BASED_FAMILIES = ("rf", "lightgbm")
DL_FAMILIES = ("lstm", "tft", "informer")

_MODULES = {
    "rf": p13_rf, "lightgbm": p13_lgbm,
    "lstm": p13_lstm, "tft": p13_tft, "informer": p13_informer,
}


def module_for(family: str):
    if family not in _MODULES:
        raise ValueError(f"알 수 없는 family: {family!r} (허용값: {FAMILIES})")
    return _MODULES[family]


def run_single_config(
    family: str, sub_a: pd.DataFrame, horizon: int, hp: dict, seed: int,
    trial_label: str, stage: str = "p13_seed_robustness", checkpoint_root: Path | None = None,
) -> dict:
    """하나의 family/configuration을 2023 P13 4-fold에서 학습·평가한다.
    row-based family와 sequence-based family의 실행 방식 차이는 이 함수에서 처리한다.
    """
    mod = module_for(family)
    folds = day3_folds.generate_expanding_folds(sub_a, 2023, horizon)

    if family in ROW_BASED_FAMILIES:
        fold_data = [(f["fold"], sub_a.loc[f["train_mask"]], sub_a.loc[f["val_mask"]]) for f in folds]
        return mod._run_folds(hp, fold_data, horizon, seed, stage, trial_label)

    if checkpoint_root is None:
        raise ValueError(f"{family}: DL family는 checkpoint_root가 필요함")
    return mod._run_folds(hp, sub_a, folds, horizon, seed, stage, trial_label, checkpoint_root)


def fit_final_model_for_family(family: str, full_df: pd.DataFrame, horizon: int, hp: dict, seed: int,
                                model_artifact_path: Path) -> dict:
    """선택된 family/configuration을 validation 없이 전체 Final retrain 데이터로 학습한다."""
    mod = module_for(family)
    if family == "rf":
        return mod.rf_trainer.fit_final_model(
            full_df, horizon, n_estimators=hp["n_estimators"], max_features=hp["max_features"],
            min_samples_leaf=hp["min_samples_leaf"], seed=seed, model_artifact_path=model_artifact_path,
        )
    if family == "lightgbm":
        return mod.lgbm_trainer.fit_final_model(
            full_df, horizon, num_leaves=hp["num_leaves"], min_child_samples=hp["min_child_samples"],
            feature_fraction=hp["feature_fraction"], bagging_fraction=hp["bagging_fraction"],
            bagging_freq=hp["bagging_freq"], lambda_l1=hp["lambda_l1"], lambda_l2=hp["lambda_l2"],
            seed=seed, model_artifact_path=model_artifact_path,
            fixed_params=mod.P13_FIXED_PARAMS,
        )
    if family == "lstm":
        return mod.lstm_trainer.fit_final_model(
            full_df, horizon, hp["lookback"], hidden_size=hp["hidden_size"], batch_size=hp["batch_size"],
            max_epochs=hp["max_epochs"], learning_rate=hp["learning_rate"], weight_decay=hp["weight_decay"],
            seed=seed, model_artifact_path=model_artifact_path,
        )
    if family == "tft":
        return mod.tft_trainer.fit_final_model(
            full_df, horizon, hp["lookback"], hidden_size=hp["hidden_size"],
            hidden_continuous_size=hp["hidden_continuous_size"], attention_head_size=hp["attention_head_size"],
            dropout=hp["dropout"], learning_rate=hp["learning_rate"], gradient_clip_val=hp["gradient_clip_val"],
            batch_size=hp["batch_size"], max_epochs=hp["max_epochs"], seed=seed,
            checkpoint_dir=model_artifact_path,
        )
    if family == "informer":
        return mod.informer_trainer.fit_final_model(
            full_df, horizon, hp["lookback"], e_layers=hp["e_layers"], n_heads=hp["n_heads"],
            batch_size=hp["batch_size"], max_epochs=hp["max_epochs"], learning_rate=hp["learning_rate"],
            seed=seed, model_artifact_path=model_artifact_path,
        )
    raise ValueError(f"알 수 없는 family: {family!r}")


def build_2024_prediction_batch(horizon: int, lookback: int, dev: pd.DataFrame, holdout_df: pd.DataFrame):
    """2024 DL holdout prediction용 sequence batch를 생성한다.
    Final retrain population으로 residual NaN imputation을 fit하고,
    pre-2024 development history와 2024 holdout을 연결해 lookback context를 보존한다.
    2025로 target이 넘어가는 origin은 제외하고 실제 2024 prediction 대상만 반환한다.
    """
    from src.forecasting.common import folds as day3_folds
    from src.forecasting.deep_learning.common.sequence_builder import (
        build_sequences,
        fold_origin_key_set,
        split_batch_by_origin_keys,
    )
    from src.forecasting.deep_learning.common.structural_nan import (
        apply_residual_nan_medians,
        fit_residual_nan_medians,
    )

    final_retrain_population = dev.loc[day3_folds.final_retrain_mask(dev, horizon)]
    residual_nan_maps = fit_residual_nan_medians(final_retrain_population)
    combined = pd.concat([dev, holdout_df], ignore_index=True)
    combined_imputed, _ = apply_residual_nan_medians(combined, residual_nan_maps)

    full_batch = build_sequences(combined_imputed, horizon, lookback)

    eligible_mask = day3_folds.holdout_2024_mask(holdout_df, horizon)
    expected_keys = fold_origin_key_set(holdout_df, eligible_mask)
    holdout_batch = split_batch_by_origin_keys(full_batch, expected_keys)

    generated_keys = set(holdout_batch.keys[["center_id", "sku_id", "week_st"]]
                        .itertuples(index=False, name=None))
    missing_keys = expected_keys - generated_keys
    # skipped_keys와 expected_keys의 datetime type을 맞춰 비교한다.
    skipped_keys_normalized = {(c, s, pd.Timestamp(w)) for c, s, w in full_batch.skipped_keys}
    missing_info = {
        "n_expected": len(expected_keys), "n_generated": len(generated_keys),
        "n_missing": len(missing_keys),
        "missing_reason_sample": [k for k in skipped_keys_normalized if k in missing_keys][:20],
    }

    if holdout_batch.time_varying.shape[0] > 0 and holdout_batch.time_varying.shape[1] != lookback:
        raise RuntimeError(f"holdout sequence length={holdout_batch.time_varying.shape[1]}가 lookback={lookback}과 다름")
    dup = holdout_batch.keys.duplicated(subset=["center_id", "sku_id", "week_st"]).sum()
    if dup:
        raise RuntimeError(f"2024 holdout prediction key 중복 {dup}건 발견")
    max_ts = holdout_batch.keys["week_st"].max() if len(holdout_batch.keys) else None
    if max_ts is not None and max_ts > holdout_df["week_st"].max():
        raise RuntimeError("holdout batch origin이 holdout_df 범위를 벗어남")

    return holdout_batch, combined_imputed, missing_info


def predict_with_final_model_for_family(family: str, metadata: dict, holdout_df: pd.DataFrame, horizon: int) -> dict:
    """저장된 Final model artifact로 2024 holdout prediction을 생성한다.
    재학습이나 model selection은 수행하지 않으며,
    DL family는 pre-2024 history를 연결해 lookback context를 구성한다.
    """
    import joblib
    import torch

    from src.forecasting.common import evaluator as ev
    from src.forecasting.common.config import TARGET_COLS
    from src.forecasting.common.data_loader import load_development
    from src.forecasting.deep_learning.informer.dataset import InformerSequenceDataset, build_informer_tensors
    from src.forecasting.deep_learning.tft.dataset_adapter import build_tft_long_dataframe, build_validation_dataset
    from pytorch_forecasting import TemporalFusionTransformer

    hp = metadata["config"]
    model_artifact_path = Path(metadata["model_artifact_path"])
    target_col = TARGET_COLS[horizon]

    if family in ("rf", "lightgbm"):
        eligible = day3_folds.holdout_2024_mask(holdout_df, horizon)
        eligible_df = holdout_df.loc[eligible]
        bundle = joblib.load(model_artifact_path)
        preprocessor, model = bundle["preprocessor"], bundle["model"]
        X = preprocessor.transform(eligible_df)
        pred_log = model.predict(X) if family == "rf" else model.predict(X, num_iteration=None)
        raw_pred = np.clip(ev.inverse_transform_prediction(pred_log), 0, None)
        y_true = eligible_df[target_col].to_numpy(dtype=float)
        keys = eligible_df[["center_id", "sku_id", "week_st"]].copy()
        keys["target_date"] = keys["week_st"] + pd.Timedelta(weeks=horizon)
        return {"y_true": y_true, "y_pred": raw_pred, "keys": keys, "missing_info": None}

    if hp["lookback"] != 13:
        raise RuntimeError(f"{family} 2024 holdout: lookback={hp['lookback']}가 frozen spec 13과 다름")

    dev = load_development()
    holdout_batch, combined_imputed, missing_info = build_2024_prediction_batch(
        horizon, hp["lookback"], dev, holdout_df,
    )

    if family == "lstm":
        mod = module_for(family)
        checkpoint = torch.load(model_artifact_path, map_location="cpu", weights_only=False)
        preprocessor = checkpoint["preprocessor"]
        transformed = preprocessor.transform(holdout_batch)
        model = mod.lstm_trainer.LSTMForecaster(
            n_time_varying=holdout_batch.time_varying.shape[-1], static_cont_dim=holdout_batch.static_cont.shape[-1],
            cat_vocab_sizes=preprocessor.vocab_sizes(), hidden_size=checkpoint["hidden_size"],
            num_layers=mod.lstm_trainer.NUM_LAYERS, dropout=mod.lstm_trainer.DROPOUT,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        with torch.no_grad():
            pred_log = model(
                torch.as_tensor(transformed.time_varying, dtype=torch.float32),
                torch.as_tensor(transformed.static_cont, dtype=torch.float32),
                torch.as_tensor(transformed.static_cat, dtype=torch.long),
            ).numpy()
        raw_pred = np.clip(ev.inverse_transform_prediction(pred_log), 0, None)
        return {"y_true": holdout_batch.target, "y_pred": raw_pred, "keys": holdout_batch.keys,
               "missing_info": missing_info}

    if family == "tft":
        training_dataset_path = model_artifact_path.parent / "training_dataset.pkl"
        if not training_dataset_path.exists():
            raise FileNotFoundError(f"TFT final retrain training_dataset이 없음: {training_dataset_path}")
        # PyTorch>=2.6 호환을 위해 직접 저장한 training_dataset을 weights_only=False로 로드한다.
        training_dataset = torch.load(str(training_dataset_path), weights_only=False)

        holdout_long = build_tft_long_dataframe(holdout_batch, combined_imputed, horizon, group_offset=0)
        predict_dataset = build_validation_dataset(training_dataset, holdout_long)
        predict_dataloader = predict_dataset.to_dataloader(train=False, batch_size=256, num_workers=0)

        model = TemporalFusionTransformer.load_from_checkpoint(model_artifact_path)
        model.eval()
        with torch.no_grad():
            prediction_result = model.predict(predict_dataloader, mode="prediction", return_index=True)
        raw_pred_log = prediction_result.output.squeeze(-1).cpu().numpy()
        origin_positions = prediction_result.index["origin_id"].to_numpy()

        n_holdout = len(holdout_batch.target)
        if len(origin_positions) != n_holdout or set(origin_positions.tolist()) != set(range(n_holdout)):
            raise AssertionError(
                f"TFT 2024 prediction origin_id가 holdout_batch와 1:1 대응하지 않음: "
                f"n_pred={len(origin_positions)}, n_holdout={n_holdout}"
            )
        pred_log = np.empty(n_holdout, dtype=float)
        pred_log[origin_positions] = raw_pred_log
        raw_pred = np.clip(ev.inverse_transform_prediction(pred_log), 0, None)
        return {"y_true": holdout_batch.target, "y_pred": raw_pred, "keys": holdout_batch.keys,
               "missing_info": missing_info}

    if family == "informer":
        checkpoint = torch.load(model_artifact_path, map_location="cpu", weights_only=False)
        preprocessor = checkpoint["preprocessor"]
        if checkpoint["label_len"] != 6:
            raise RuntimeError(f"Informer 2024 holdout: label_len={checkpoint['label_len']}가 frozen spec 6과 다름")

        tensors = build_informer_tensors(preprocessor, holdout_batch, combined_imputed, horizon)
        dataset = InformerSequenceDataset(tensors)
        loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

        mod = module_for(family)
        model = mod.informer_trainer.InformerForecaster(
            n_time_varying=holdout_batch.time_varying.shape[-1], n_known=tensors["decoder_known"].shape[-1],
            static_cont_dim=holdout_batch.static_cont.shape[-1], cat_vocab_sizes=preprocessor.vocab_sizes(),
            e_layers=checkpoint["e_layers"], n_heads=checkpoint["n_heads"],
            d_model=mod.informer_trainer.D_MODEL, d_ff=mod.informer_trainer.D_FF,
            d_layers=mod.informer_trainer.D_LAYERS, factor=mod.informer_trainer.FACTOR,
            dropout=mod.informer_trainer.DROPOUT,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()
        pred_log = mod.informer_trainer._run_epoch_predict(model, loader, "cpu", len(dataset))
        raw_pred = np.clip(ev.inverse_transform_prediction(pred_log), 0, None)
        return {"y_true": holdout_batch.target, "y_pred": raw_pred, "keys": holdout_batch.keys,
               "missing_info": missing_info}

    raise ValueError(f"알 수 없는 family: {family!r}")
