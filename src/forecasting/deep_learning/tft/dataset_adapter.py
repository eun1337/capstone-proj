"""
dataset_adapter.py

TFT용 TimeSeriesDataSet 변환.

각 origin을 lookback encoder와 1-step decoder로 구성해 Direct h1/h2/h4 모델을 지원한다.
decoder의 calendar feature는 target_date 행에서 가져오고, horizon별 holiday feature는
이미 shift된 origin 값을 사용한다. target은 log1p scale을 그대로 사용하며,
validation은 train에서 fit한 TimeSeriesDataSet 설정을 재사용한다.
"""

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder, TorchNormalizer

from src.forecasting.common.config import HOLIDAY_FEATURES
from src.forecasting.deep_learning.common.config import get_model_feature_roles
from src.forecasting.deep_learning.common.sequence_builder import SequenceBatch

GROUP_COL = "origin_id"
TIME_COL = "time_idx"
TARGET_COL = "target_log1p"


def build_tft_long_dataframe(batch: SequenceBatch, df: pd.DataFrame, horizon: int, group_offset: int = 0):
    """SequenceBatch를 lookback encoder와 1-step decoder 구조의 long DataFrame으로 변환한다.
    decoder calendar feature는 target_date 행에서 조회하고,
    holiday와 observed feature는 origin의 마지막 timestep 값을 사용한다.
    """
    roles = get_model_feature_roles(horizon)
    known_cols = list(roles["time_varying_known"])
    observed_cols = list(roles["time_varying_observed"])
    tv_cols = known_cols + observed_cols
    static_cat_cols = list(roles["static_categorical"])
    static_cont_cols = list(roles["static_continuous"])

    holiday_cols = list(HOLIDAY_FEATURES[horizon])
    calendar_known_cols = [c for c in known_cols if c not in holiday_cols]

    n = len(batch.target)
    lookback = batch.time_varying.shape[1]
    n_tv = len(tv_cols)

    # Encoder rows: origin 순서 → timestep 순서
    origin_ids_enc = np.repeat(np.arange(n) + group_offset, lookback)
    time_idx_enc = np.tile(np.arange(lookback), n)
    tv_flat = batch.time_varying.reshape(-1, n_tv)
    static_cont_enc = np.repeat(batch.static_cont, lookback, axis=0)
    static_cat_enc = np.repeat(batch.static_cat, lookback, axis=0)

    encoder_df = pd.DataFrame(tv_flat, columns=tv_cols)
    encoder_df[GROUP_COL] = origin_ids_enc
    encoder_df[TIME_COL] = time_idx_enc
    for j, col in enumerate(static_cont_cols):
        encoder_df[col] = static_cont_enc[:, j]
    for j, col in enumerate(static_cat_cols):
        encoder_df[col] = static_cat_enc[:, j]
    encoder_df[TARGET_COL] = 0.0

    # Decoder rows: calendar는 target_date, holiday/observed는 origin 마지막 timestep 사용
    target_lookup = df[["center_id", "sku_id", "week_st"] + calendar_known_cols]
    keys_target = batch.keys[["center_id", "sku_id", "target_date"]].reset_index(drop=True)
    merged = keys_target.merge(
        target_lookup, left_on=["center_id", "sku_id", "target_date"],
        right_on=["center_id", "sku_id", "week_st"], how="left",
    )
    if len(merged) != n or merged[calendar_known_cols].isna().any().any():
        raise KeyError("decoder future known(calendar) 조회 실패: target_date가 df에 없는 origin 존재")

    last_step = batch.time_varying[:, -1, :]
    decoder_df = pd.DataFrame({
        GROUP_COL: np.arange(n) + group_offset,
        TIME_COL: np.full(n, lookback),
    })
    for col in calendar_known_cols:
        decoder_df[col] = merged[col].to_numpy()
    for col in holiday_cols:
        decoder_df[col] = last_step[:, tv_cols.index(col)]
    for col in observed_cols:
        decoder_df[col] = last_step[:, tv_cols.index(col)]
    for j, col in enumerate(static_cont_cols):
        decoder_df[col] = batch.static_cont[:, j]
    for j, col in enumerate(static_cat_cols):
        decoder_df[col] = batch.static_cat[:, j]
    decoder_df[TARGET_COL] = np.log1p(batch.target)

    long_df = pd.concat([encoder_df, decoder_df], ignore_index=True)
    long_df = long_df.sort_values([GROUP_COL, TIME_COL], kind="mergesort").reset_index(drop=True)
    for col in static_cat_cols:
        long_df[col] = long_df[col].astype(str)
    return long_df


def build_training_dataset(long_df: pd.DataFrame, horizon: int, lookback: int) -> TimeSeriesDataSet:
    """train long DataFrame으로 TimeSeriesDataSet과 categorical encoder를 fit한다.
    validation의 미확인 category는 NaNLabelEncoder(add_nan=True)로 처리한다.
    """
    roles = get_model_feature_roles(horizon)
    known_cols = list(roles["time_varying_known"])
    observed_cols = list(roles["time_varying_observed"])
    static_cat_cols = list(roles["static_categorical"])
    static_cont_cols = list(roles["static_continuous"])

    return TimeSeriesDataSet(
        long_df,
        time_idx=TIME_COL,
        target=TARGET_COL,
        group_ids=[GROUP_COL],
        max_encoder_length=lookback,
        min_encoder_length=lookback,
        max_prediction_length=1,
        min_prediction_length=1,
        static_categoricals=static_cat_cols,
        static_reals=static_cont_cols,
        time_varying_known_reals=known_cols,
        time_varying_unknown_reals=observed_cols,
        categorical_encoders={col: NaNLabelEncoder(add_nan=True) for col in static_cat_cols},
        target_normalizer=TorchNormalizer(method="identity"),
        add_relative_time_idx=False,
        add_target_scales=False,
        add_encoder_length=False,
        allow_missing_timesteps=False,
    )


def build_validation_dataset(training_dataset: TimeSeriesDataSet, val_long_df: pd.DataFrame) -> TimeSeriesDataSet:
    """train-fit encoder/normalizer를 그대로 재사용해 validation을 transform만 한다."""
    return TimeSeriesDataSet.from_dataset(training_dataset, val_long_df, predict=False, stop_randomization=True)
