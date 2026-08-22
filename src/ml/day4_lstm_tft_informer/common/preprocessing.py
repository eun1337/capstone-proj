"""
preprocessing.py
DL 공통 입력 전처리 - SequenceBatch의 연속형(static/time-varying) feature를 train-fit
z-score로 표준화하고, KAN static categorical 3종을 train-fit vocabulary(미확인 값은 <UNK>)로
정수 인덱스화한다. train에서만 fit하고 validation/holdout은 transform만 한다.
"""

import numpy as np

from src.ml.day4_lstm_tft_informer.common.sequence_builder import SequenceBatch

UNK_TOKEN = "<UNK>"


class SequencePreprocessor:
    """train SequenceBatch로 fit(), 이후 임의 SequenceBatch에 transform()을 반복 적용한다."""

    def __init__(self):
        self._tv_mean = None
        self._tv_std = None
        self._static_cont_mean = None
        self._static_cont_std = None
        self._cat_vocabs = None

    def fit(self, batch: SequenceBatch) -> "SequencePreprocessor":
        if not np.isfinite(batch.time_varying).all():
            raise ValueError("time_varying(fit 대상)에 NaN/Inf가 존재함")
        if not np.isfinite(batch.static_cont).all():
            raise ValueError("static_cont(fit 대상)에 NaN/Inf가 존재함")

        n_tv = batch.time_varying.shape[-1]
        flat_tv = batch.time_varying.reshape(-1, n_tv)
        self._tv_mean = flat_tv.mean(axis=0)
        tv_std = flat_tv.std(axis=0)
        self._tv_std = np.where(tv_std < 1e-8, 1.0, tv_std)

        self._static_cont_mean = batch.static_cont.mean(axis=0)
        static_cont_std = batch.static_cont.std(axis=0)
        self._static_cont_std = np.where(static_cont_std < 1e-8, 1.0, static_cont_std)

        n_static_cat = batch.static_cat.shape[1]
        self._cat_vocabs = []
        for j in range(n_static_cat):
            vocab = {UNK_TOKEN: 0}
            for value in sorted(set(batch.static_cat[:, j].tolist())):
                vocab[value] = len(vocab)
            self._cat_vocabs.append(vocab)
        return self

    def _check_fitted(self) -> None:
        if self._cat_vocabs is None:
            raise RuntimeError("fit()을 먼저 호출해야 함")

    def transform(self, batch: SequenceBatch) -> dict:
        self._check_fitted()

        tv = (batch.time_varying - self._tv_mean) / self._tv_std
        static_cont = (batch.static_cont - self._static_cont_mean) / self._static_cont_std

        n, n_static_cat = batch.static_cat.shape
        static_cat_idx = np.zeros((n, n_static_cat), dtype=np.int64)
        for j in range(n_static_cat):
            vocab = self._cat_vocabs[j]
            static_cat_idx[:, j] = [vocab.get(value, 0) for value in batch.static_cat[:, j]]

        if not np.isfinite(tv).all():
            raise ValueError("transform 결과 time_varying에 NaN/Inf가 존재함")
        if not np.isfinite(static_cont).all():
            raise ValueError("transform 결과 static_cont에 NaN/Inf가 존재함")

        return {
            "time_varying": tv.astype(np.float32),
            "static_cont": static_cont.astype(np.float32),
            "static_cat": static_cat_idx,
            "target": batch.target.astype(np.float32),
        }

    def vocab_sizes(self) -> list[int]:
        self._check_fitted()
        return [len(vocab) for vocab in self._cat_vocabs]
