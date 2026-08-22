"""
sequence_builder.py
(center_id, sku_id) 그룹을 week_st 오름차순으로 정렬한 뒤, 각 origin row(=lookback
윈도우의 마지막 주)마다 encoder 길이=lookback 시퀀스와 horizon별 raw target을 만든다.
origin 이후 미래 row는 encoder에 절대 포함하지 않는다. 연속된 lookback주 이력이 없거나
target이 NaN인 origin은 건너뛰고 개수만 센다. adi/cv2 expanding의 구조적 residual NaN은
이 함수를 부르기 전에 common/structural_nan.py로 이미 채워져 있어야 하며, 그럼에도
lookback 윈도우 안에 NaN/Inf가 남아 있으면 조용히 skip하지 않고 즉시 실패한다(imputation
누락 감지). qty_log1p는 raw qty에서 이 함수 안에서 다시 계산한다(기존 qty_log1p
컬럼에 재적용하지 않음).
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.ml.day4_lstm_tft_informer.common.config import (
    TARGET_COLS,
    get_model_feature_roles,
    validate_horizon,
    validate_lookback,
)

CENTER_COL = "center_id"
SKU_COL = "sku_id"
WEEK_COL = "week_st"
QTY_COL = "qty"


@dataclass
class SequenceBatch:
    keys: pd.DataFrame  # center_id, sku_id, week_st(origin), target_date
    static_cont: np.ndarray  # (N, n_static_cont) float
    static_cat: np.ndarray  # (N, n_static_cat) object (raw category 값)
    time_varying: np.ndarray  # (N, lookback, n_time_varying) float
    target: np.ndarray  # (N,) float, raw target_h{horizon}
    n_insufficient_history: int
    skipped_keys: list  # 건너뛴 origin의 (center_id, sku_id, week_st) 키 목록


def build_sequences(df: pd.DataFrame, horizon: int, lookback: int) -> SequenceBatch:
    validate_horizon(horizon)
    validate_lookback(lookback)

    roles = get_model_feature_roles(horizon)
    static_cat_cols = list(roles["static_categorical"])
    static_cont_cols = list(roles["static_continuous"])
    tv_cols = list(roles["time_varying_known"]) + list(roles["time_varying_observed"])
    target_col = TARGET_COLS[horizon]

    required = static_cat_cols + static_cont_cols + tv_cols + [
        target_col, CENTER_COL, SKU_COL, WEEK_COL, QTY_COL,
    ]
    missing = [c for c in dict.fromkeys(required) if c not in df.columns]
    if missing:
        raise KeyError(f"df에 필수 컬럼 누락: {missing}")

    ordered = df.sort_values([CENTER_COL, SKU_COL, WEEK_COL]).reset_index(drop=True).copy()
    ordered["qty_log1p"] = np.log1p(ordered[QTY_COL].to_numpy(dtype=float))

    keys_rows = []
    static_cont_list, static_cat_list, tv_list, target_list = [], [], [], []
    n_insufficient = 0
    skipped_keys = []

    groups = ordered.groupby([CENTER_COL, SKU_COL], sort=False, observed=True).indices
    for (center_id, sku_id), idx in groups.items():
        grp = ordered.iloc[idx]
        n = len(grp)
        weeks = grp[WEEK_COL].to_numpy()

        day_diff = np.full(n, np.nan)
        if n > 1:
            day_diff[1:] = (weeks[1:] - weeks[:-1]) / np.timedelta64(1, "D")
        run_length = np.ones(n, dtype=int)
        for i in range(1, n):
            run_length[i] = run_length[i - 1] + 1 if day_diff[i] == 7 else 1

        tv_arr = grp[tv_cols].to_numpy(dtype=float)
        static_cont_arr = grp[static_cont_cols].to_numpy(dtype=float)
        static_cat_arr = grp[static_cat_cols].to_numpy(dtype=object)
        target_arr = grp[target_col].to_numpy(dtype=float)
        week_arr = grp[WEEK_COL].to_numpy()

        for i in range(n):
            if run_length[i] < lookback or np.isnan(target_arr[i]):
                n_insufficient += 1
                skipped_keys.append((center_id, sku_id, week_arr[i]))
                continue
            window = tv_arr[i - lookback + 1: i + 1]
            if not np.isfinite(window).all():
                bad_cols = [tv_cols[j] for j in range(len(tv_cols)) if not np.isfinite(window[:, j]).all()]
                raise ValueError(
                    f"lookback 윈도우에 NaN/Inf가 남아있음(imputation 누락 가능): "
                    f"center_id={center_id}, sku_id={sku_id}, week_st={week_arr[i]}, features={bad_cols}"
                )
            if not np.isfinite(static_cont_arr[i]).all():
                raise ValueError(
                    f"static_cont에 NaN/Inf가 존재함: center_id={center_id}, sku_id={sku_id}, "
                    f"week_st={week_arr[i]}"
                )
            keys_rows.append((center_id, sku_id, week_arr[i]))
            static_cont_list.append(static_cont_arr[i])
            static_cat_list.append(static_cat_arr[i])
            tv_list.append(window)
            target_list.append(target_arr[i])

    keys_df = pd.DataFrame(keys_rows, columns=[CENTER_COL, SKU_COL, WEEK_COL])
    keys_df[WEEK_COL] = pd.to_datetime(keys_df[WEEK_COL])
    keys_df["target_date"] = keys_df[WEEK_COL] + pd.Timedelta(weeks=horizon)

    return SequenceBatch(
        keys=keys_df,
        static_cont=np.asarray(static_cont_list, dtype=float).reshape(-1, len(static_cont_cols)),
        static_cat=np.asarray(static_cat_list, dtype=object).reshape(-1, len(static_cat_cols)),
        time_varying=np.asarray(tv_list, dtype=float).reshape(-1, lookback, len(tv_cols)),
        target=np.asarray(target_list, dtype=float),
        n_insufficient_history=n_insufficient,
        skipped_keys=skipped_keys,
    )


def split_batch_by_origin_keys(batch: SequenceBatch, origin_keys: set) -> SequenceBatch:
    """batch.keys의 (center_id, sku_id, week_st)가 origin_keys(집합)에 속하는 시퀀스만
    골라 서브셋 SequenceBatch를 만든다. 서브셋 자체는 별도 스킵이 없으므로
    n_insufficient_history/skipped_keys는 0/빈 리스트로 둔다(전체 스킵 집계는
    원본 batch.skipped_keys를 이용해 호출부에서 따로 계산)."""
    keys_tuples = list(batch.keys[[CENTER_COL, SKU_COL, WEEK_COL]].itertuples(index=False, name=None))
    mask = np.array([k in origin_keys for k in keys_tuples])
    return SequenceBatch(
        keys=batch.keys.loc[mask].reset_index(drop=True),
        static_cont=batch.static_cont[mask],
        static_cat=batch.static_cat[mask],
        time_varying=batch.time_varying[mask],
        target=batch.target[mask],
        n_insufficient_history=0,
        skipped_keys=[],
    )


def fold_origin_key_set(df: pd.DataFrame, mask: pd.Series) -> set:
    """df.loc[mask]의 (center_id, sku_id, week_st) 키 집합을 만든다 - fold의 train_mask/
    val_mask를 시퀀스 origin 분리에 쓰기 위한 헬퍼."""
    sub = df.loc[mask, [CENTER_COL, SKU_COL, WEEK_COL]]
    return set(sub.itertuples(index=False, name=None))
