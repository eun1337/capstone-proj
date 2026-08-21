"""
statistical_utils.py

ARIMA/ARIMAX 통계모델 공통 유틸리티.

auto_arima 탐색 설정, 최소 history 기준, 수렴·AICc 판정,
log1p 예측값 역변환, cold-start fallback 및 실제값 조회 helper를 제공한다.
모델 학습이나 forecast-origin별 실행 로직은 포함하지 않는다.
"""

import numpy as np
import pandas as pd

CENTER_COL = "center_id"
WEEK_COL = "week_st"
QTY_COL = "qty"
SUBCAT_COL = "KAN_소분류"
MIDCAT_COL = "KAN_중분류"
LARGECAT_COL = "KAN_대분류"

SHORT_HISTORY_MAX = 5  # n_obs<=5 -> own-history mean, n_obs>=6(MIN_HISTORY)부터 ARIMA 후보
MIN_HISTORY = SHORT_HISTORY_MAX + 1

AUTO_ARIMA_KWARGS = dict(
    seasonal=False,
    start_p=0, max_p=5,
    d=None, max_d=2,
    start_q=0, max_q=5,
    stepwise=True,
    information_criterion="aicc",
    test="kpss",
    suppress_warnings=False,
    error_action="warn",
    trace=False,
)

FORECAST_N_PERIODS = 4
FORECAST_STEP_TO_HORIZON = {0: "h1", 1: "h2", 3: "h4"}


def extract_valid_fit_candidates(ret) -> list:
    """return_valid_fits=True 반환 형식을 안전하게 처리한다(pmdarima 2.1.1 실측:
    fit-완료 ARIMA 객체들의 flat tuple). 인식할 수 없는 형식이면 TypeError."""
    if isinstance(ret, (tuple, list)) and len(ret) > 0 and all(hasattr(m, "order") for m in ret):
        return list(ret)
    if isinstance(ret, tuple) and len(ret) == 2 and hasattr(ret[1], "__iter__"):
        candidates = list(ret[1])
        if all(hasattr(m, "order") for m in candidates):
            return candidates
    raise TypeError(f"return_valid_fits=True 반환 형식을 인식할 수 없음: {type(ret)}")


def candidate_convergence(model) -> "bool | str":
    res = getattr(model, "arima_res_", None)
    mle_retvals = getattr(res, "mle_retvals", None) if res is not None else None
    if isinstance(mle_retvals, dict) and "converged" in mle_retvals:
        return bool(mle_retvals["converged"])
    return "Unknown"


def candidate_aicc(model) -> float:
    """pmdarima ARIMA 객체의 공식 aicc() API를 사용한다(내부 arima_res_.aicc 직접 접근 금지 —
    information_criterion="aicc" 탐색 기준과 동일한 값을 얻기 위함)."""
    try:
        return float(model.aicc())
    except Exception:
        return np.nan


def expm1_clip(value_log1p: float) -> tuple[float, float]:
    """log1p 예측값을 raw로 역변환하고 0으로 clip한다. (unclipped, clipped) 반환."""
    raw = float(np.expm1(value_log1p))
    return raw, max(raw, 0.0)


def _assert_single_center(df: pd.DataFrame, fn_name: str) -> None:
    centers = df[CENTER_COL].unique()
    if len(centers) > 1:
        raise ValueError(f"{fn_name}()는 단일 center 데이터만 입력받아야 함(A/B 혼합 금지): centers={list(centers)}")


def compute_coldstart_means(df: pd.DataFrame, origin: pd.Timestamp) -> dict:
    """origin까지(week_st<=origin) 관측된 행으로 center/대/중/소분류별 raw qty 평균을
    계산한다(미래 정보 누출 없음). df는 단일 center여야 한다(A/B 혼합 평균 방지)."""
    _assert_single_center(df, "compute_coldstart_means")
    hist = df[df[WEEK_COL] <= origin]
    return {
        "center": float(hist[QTY_COL].mean()) if len(hist) else np.nan,
        "sub": hist.groupby(SUBCAT_COL, observed=True)[QTY_COL].mean(),
        "mid": hist.groupby(MIDCAT_COL, observed=True)[QTY_COL].mean(),
        "large": hist.groupby(LARGECAT_COL, observed=True)[QTY_COL].mean(),
    }


def build_sku_category_map(df: pd.DataFrame) -> pd.DataFrame:
    """sku_id -> (소/중/대분류) 정적 매핑. df는 단일 center여야 한다(A/B 혼합 금지)."""
    _assert_single_center(df, "build_sku_category_map")
    return (
        df[["sku_id", SUBCAT_COL, MIDCAT_COL, LARGECAT_COL]]
        .drop_duplicates("sku_id")
        .set_index("sku_id")
    )


def coldstart_fallback(sku_cat: pd.Series, coldstart_means: dict) -> tuple[float, str]:
    """소분류 -> 중분류 -> 대분류 -> center 순으로 값이 있는 첫 단계를 채택."""
    sub_val = coldstart_means["sub"].get(sku_cat[SUBCAT_COL], np.nan)
    if pd.notna(sub_val):
        return float(sub_val), "subcategory"
    mid_val = coldstart_means["mid"].get(sku_cat[MIDCAT_COL], np.nan)
    if pd.notna(mid_val):
        return float(mid_val), "midcategory"
    large_val = coldstart_means["large"].get(sku_cat[LARGECAT_COL], np.nan)
    if pd.notna(large_val):
        return float(large_val), "category"
    center_val = coldstart_means["center"]
    return (float(center_val) if pd.notna(center_val) else 0.0), "center"


def lookup_actual(week_arr: np.ndarray, qty_arr: np.ndarray, target_week: pd.Timestamp) -> float:
    """target_week에 해당하는 실측 qty를 찾는다. 없으면 NaN."""
    idx = np.searchsorted(week_arr, np.datetime64(target_week))
    if idx < len(week_arr) and week_arr[idx] == np.datetime64(target_week):
        return float(qty_arr[idx])
    return np.nan
