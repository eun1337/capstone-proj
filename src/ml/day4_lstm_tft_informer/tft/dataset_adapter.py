"""
dataset_adapter.py
Day4 common(sequence_builder, structural_nan, feature role)을 pytorch_forecasting의
TimeSeriesDataSet으로 변환한다. 각 origin을 독립된 (lookback encoder + 1 decoder) 그룹으로
취급해 Direct h1/h2/h4 별도 모델 원칙을 유지한다. decoder row의 known feature 중
캘린더(ISO_주차/월/분기/covid_flag)는 실제 target_date 행에서 읽고, horizon-specific
공휴일(target_h{h}_공휴일_*)은 이미 origin 시점에 horizon-shift되어 있는 값이므로
target_date 행이 아니라 origin(encoder 마지막 timestep) 값을 그대로 쓴다(target_date
행에서 읽으면 t+h 기준 정보를 다시 t+h만큼 미루는 double-shift가 됨). target은
log1p(raw target_h{h})를 미리 계산해 TorchNormalizer(method="identity")로 재정규화
없이 그대로 사용한다. train에서만 TimeSeriesDataSet을 만들고(categorical encoder fit),
validation은 TimeSeriesDataSet.from_dataset()으로 train-fit 상태를 그대로 재사용한다.
"""

import numpy as np
import pandas as pd
from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data.encoders import NaNLabelEncoder, TorchNormalizer

from src.ml.day3_rf_lightgbm.common.config import HOLIDAY_FEATURES
from src.ml.day4_lstm_tft_informer.common.config import get_model_feature_roles
from src.ml.day4_lstm_tft_informer.common.sequence_builder import SequenceBatch

GROUP_COL = "origin_id"
TIME_COL = "time_idx"
TARGET_COL = "target_log1p"


def build_tft_long_dataframe(batch: SequenceBatch, df: pd.DataFrame, horizon: int, group_offset: int = 0):
    """이미 fold 경계로 필터링된 SequenceBatch(예: split_batch_by_origin_keys 결과)를
    (lookback encoder + 1 decoder) long dataframe으로 펼친다. df는 decoder row의 known
    feature를 실제 target_date 행에서 조회하기 위한 원본(구조적 NaN 이미 imputed)이다.
    batch가 fold 경계를 이미 만족하므로 모든 target_date가 df 안에 실제 행으로 존재한다.
    encoder/decoder row는 reshape/repeat/merge로 한 번에 만든다(원래는 origin당 Python
    dict를 append하는 반복문이었으나, 전체 A센터 스케일(origin 수십만개)에서 병목이 되어
    벡터화함 - 작은 subset에서 기존 per-origin 구현과 shape/row order/값(allclose)/
    TimeSeriesDataSet len/covered key set이 전부 동일함을 확인한 뒤 교체함)."""
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

    # --- encoder rows: (n*lookback, ...), origin 오름차순 -> timestep 오름차순 순서 ---
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

    # --- decoder rows: (n, ...). calendar는 target_date 실제 행과 merge로 조회,
    # holiday/observed는 origin(encoder 마지막 timestep) 값을 그대로 슬라이스 ---
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
    """train origin들의 long dataframe으로만 TimeSeriesDataSet을 fit한다(categorical
    encoder/target normalizer가 여기서 확정됨). KAN 3종은 validation의 미확인 값을
    허용하도록 NaNLabelEncoder(add_nan=True)를 명시한다."""
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
