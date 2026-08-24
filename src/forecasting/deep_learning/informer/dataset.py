"""
dataset.py

Informer encoder/decoder 입력 생성.

SequenceBatch를 encoder sequence와 decoder 입력으로 변환한다.
decoder calendar feature는 target_date 행에서 가져오고, horizon별 holiday feature는
이미 shift된 origin 값을 사용한다. scaling은 train-fit SequencePreprocessor를 재사용하며,
sample index를 함께 반환해 prediction과 원본 key의 정렬을 보장한다.
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.forecasting.common.config import HOLIDAY_FEATURES
from src.forecasting.deep_learning.common.config import get_model_feature_roles
from src.forecasting.deep_learning.common.preprocessing import SequencePreprocessor
from src.forecasting.deep_learning.common.sequence_builder import SequenceBatch
from src.forecasting.deep_learning.informer.config import label_len_for


def build_decoder_future_known(batch: SequenceBatch, df, horizon: int) -> np.ndarray:
    """decoder future 시점의 known feature를 구성한다.
    calendar feature는 target_date 행에서, holiday feature는 origin 마지막 timestep에서 가져온다.
    """
    roles = get_model_feature_roles(horizon)
    known_cols = list(roles["time_varying_known"])
    tv_cols = known_cols + list(roles["time_varying_observed"])
    holiday_cols = list(HOLIDAY_FEATURES[horizon])

    lookup = df.set_index(["center_id", "sku_id", "week_st"])
    lookback = batch.time_varying.shape[1]
    n = len(batch.target)

    out = np.empty((n, len(known_cols)), dtype=float)
    for i in range(n):
        key_row = batch.keys.iloc[i]
        target_row = lookup.loc[(key_row["center_id"], key_row["sku_id"], key_row["target_date"])]
        for j, col in enumerate(known_cols):
            if col in holiday_cols:
                out[i, j] = batch.time_varying[i, lookback - 1, tv_cols.index(col)]
            else:
                out[i, j] = float(target_row[col])
    return out


def _scale_future_known(preprocessor: SequencePreprocessor, batch: SequenceBatch, future_known_raw: np.ndarray) -> np.ndarray:
    """future known feature에 encoder와 동일한 train-fit scaling을 적용한다."""
    n_known = future_known_raw.shape[1]
    n_tv = batch.time_varying.shape[-1]
    pad = np.zeros((len(batch.target), n_tv - n_known), dtype=float)
    future_row = np.concatenate([future_known_raw, pad], axis=1)[:, None, :]
    extended_tv = np.concatenate([batch.time_varying, future_row], axis=1)

    extended_batch = SequenceBatch(
        keys=batch.keys, static_cont=batch.static_cont, static_cat=batch.static_cat,
        time_varying=extended_tv, target=batch.target,
        n_insufficient_history=0, skipped_keys=[],
    )
    transformed = preprocessor.transform(extended_batch)
    return transformed["time_varying"][:, -1, :n_known]


def build_informer_tensors(preprocessor: SequencePreprocessor, batch: SequenceBatch, df, horizon: int) -> dict:
    """Informer encoder/decoder 입력과 static feature, target tensor를 구성한다.
    decoder past는 encoder의 마지막 label_len 구간을 사용하고,
    future 구간은 known feature만 채우며 value는 0으로 둔다.
    """
    lookback = batch.time_varying.shape[1]
    label_len = label_len_for(lookback)

    roles = get_model_feature_roles(horizon)
    known_cols = list(roles["time_varying_known"])
    tv_cols = known_cols + list(roles["time_varying_observed"])
    qty_log1p_idx = tv_cols.index("qty_log1p")
    n_known = len(known_cols)

    transformed = preprocessor.transform(batch)
    encoder_input = transformed["time_varying"]  # (N, lookback, n_tv)

    future_known_raw = build_decoder_future_known(batch, df, horizon)
    future_known_scaled = _scale_future_known(preprocessor, batch, future_known_raw)  # (N, n_known)

    decoder_value_hist = encoder_input[:, lookback - label_len: lookback, qty_log1p_idx]  # (N, label_len)
    decoder_value_future = np.zeros((len(batch.target), 1), dtype=np.float32)
    decoder_value = np.concatenate([decoder_value_hist, decoder_value_future], axis=1)  # (N, label_len+1)

    decoder_known_hist = encoder_input[:, lookback - label_len: lookback, :n_known]  # (N, label_len, n_known)
    decoder_known_future = future_known_scaled[:, None, :].astype(np.float32)
    decoder_known = np.concatenate([decoder_known_hist, decoder_known_future], axis=1)  # (N, label_len+1, n_known)

    if not np.isfinite(decoder_value).all():
        raise ValueError("decoder_value에 NaN/Inf가 존재함")
    if not np.isfinite(decoder_known).all():
        raise ValueError("decoder_known에 NaN/Inf가 존재함")

    return {
        "encoder_input": encoder_input.astype(np.float32),
        "decoder_value": decoder_value.astype(np.float32),
        "decoder_known": decoder_known.astype(np.float32),
        "static_cont": transformed["static_cont"],
        "static_cat": transformed["static_cat"],
        "target": transformed["target"],
        "label_len": label_len,
    }


class InformerSequenceDataset(Dataset):
    """Informer 입력을 tensor로 변환하고 원본 key 정렬을 위한 sample index를 함께 반환한다."""

    def __init__(self, tensors: dict):
        self.encoder_input = torch.as_tensor(tensors["encoder_input"], dtype=torch.float32)
        self.decoder_value = torch.as_tensor(tensors["decoder_value"], dtype=torch.float32)
        self.decoder_known = torch.as_tensor(tensors["decoder_known"], dtype=torch.float32)
        self.static_cont = torch.as_tensor(tensors["static_cont"], dtype=torch.float32)
        self.static_cat = torch.as_tensor(tensors["static_cat"], dtype=torch.long)
        self.target = torch.as_tensor(tensors["target"], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.target)

    def __getitem__(self, idx: int) -> dict:
        return {
            "idx": idx,
            "encoder_input": self.encoder_input[idx],
            "decoder_value": self.decoder_value[idx],
            "decoder_known": self.decoder_known[idx],
            "static_cont": self.static_cont[idx],
            "static_cat": self.static_cat[idx],
            "target": self.target[idx],
        }
