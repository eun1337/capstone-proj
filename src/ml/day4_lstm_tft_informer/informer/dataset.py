"""
dataset.py
SequenceBatch(lookback encoder 시퀀스 + raw target)를 Informer encoder/decoder 입력
텐서로 변환한다. decoder future known(calendar 4 + holiday 3)은 TFT의
tft/dataset_adapter.py와 동일한 규칙으로 만든다 - calendar는 실제 target_date 행에서,
horizon-specific holiday는 이미 origin 시점에 horizon-shift되어 있으므로 target_date
행이 아니라 origin(encoder 마지막 timestep) 값을 그대로 쓴다(안 그러면 double-shift).
scaling은 common/preprocessing.py의 SequencePreprocessor를 그대로 재사용한다 - decoder
future known raw 값을 encoder와 같은 21채널 자리에 끼워 넣은 "확장 배치"를 만들어
동일한 preprocessor.transform()을 한 번 더 호출하는 방식으로, 새 scaling 로직을
따로 만들지 않고 train-fit 통계를 그대로 재사용한다. Dataset은 원본 batch.keys와의
명시적 위치 정렬을 위해 매 sample마다 정수 idx를 함께 반환한다(DataLoader/prediction
순서를 암묵적으로 가정하지 않기 위함).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

from src.ml.day3_rf_lightgbm.common.config import HOLIDAY_FEATURES
from src.ml.day4_lstm_tft_informer.common.config import get_model_feature_roles
from src.ml.day4_lstm_tft_informer.common.preprocessing import SequencePreprocessor
from src.ml.day4_lstm_tft_informer.common.sequence_builder import SequenceBatch
from src.ml.day4_lstm_tft_informer.informer.config import label_len_for


def build_decoder_future_known(batch: SequenceBatch, df, horizon: int) -> np.ndarray:
    """origin마다 decoder future(=target_date) 시점의 known 7개(raw, 미scale) 값을
    만든다. calendar 4개는 target_date 실제 행, holiday 3개는 origin(encoder 마지막
    timestep) 값을 그대로 쓴다(TFT와 동일한 double-shift 방지 규칙)."""
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
    """future_known_raw(N, n_known)를 encoder와 동일 scaling으로 변환하기 위해,
    encoder 시퀀스 끝에 known 채널만 채운(관측 채널은 0 placeholder) 가짜 timestep을
    붙인 확장 SequenceBatch를 만들고 기존 SequencePreprocessor.transform()을 그대로
    호출한다. scaling은 채널별 (x-mean)/std로 lookback 차원과 무관하게 적용되므로
    encoder 구간 결과는 원본과 완전히 동일하다."""
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
    """encoder_input/decoder_value/decoder_known/static_cont/static_cat/target 텐서를
    만든다. label_len = floor(lookback/2). decoder past 구간은 encoder의 마지막
    label_len timestep을 그대로 슬라이스한 값(scaled, encoder와 동일)이고, decoder
    future 구간은 known만 target_date/origin 기준으로 채우고 value는 0-padding한다."""
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
    """build_informer_tensors() 결과를 텐서로 감싼다. __getitem__은 원본 batch.keys
    행 순서를 그대로 보존하는 정수 idx를 함께 반환해, DataLoader/prediction 결과를
    key와 명시적으로 재정렬할 수 있게 한다."""

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
