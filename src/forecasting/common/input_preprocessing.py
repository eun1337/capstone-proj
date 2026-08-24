"""
input_preprocessing.py
RF/LightGBM 공통 모델 입력 dtype 정규화. ISO_주차/existed_before_regime/is_warmup/
coldstart_flag 4개 컬럼의 dtype만 변환하며, NaN/카테고리/scaling/target 처리는 하지 않는다.
"""

import pandas as pd

DTYPE_NORMALIZE_COLS = {
    "ISO_주차": "int64",
    "existed_before_regime": "int8",
    "is_warmup": "int8",
    "coldstart_flag": "int8",
}


def normalize_model_input_dtypes(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """feature_cols에 포함된 경우에만 위 4개 컬럼의 dtype을 변환한 복사본을 반환한다."""
    out = df.copy()
    for col, dtype in DTYPE_NORMALIZE_COLS.items():
        if col in feature_cols and col in out.columns:
            out[col] = out[col].astype(dtype)
    return out
