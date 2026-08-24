"""
08_sku_sarima_sarimax.py
SKU 단위 SARIMA / SARIMAX(S4 Operational-Full) 파이프라인 — day2_statistical_models 신규 트랙.

배경 및 전제(2026-08-21으로 확정):
    기존 통계 트랙(arima_baseline.py/arimax_exog.py/sarima_baseline.py)은 "SKU 수천 개에
    ARIMA를 개별 적용하는 것은 비현실적"이라는 판단 하에 센터 전체 합산 단일 시계열로
    Baseline을 잡았고, sarima_baseline.py는 s=13을 도메인 근거 없음으로 폐기, B센터는
    표본 부족(26주)으로 SARIMA 적용 대상에서 아예 제외했었다. 해당 크랙은 arima_development_orders.parquet(SKU x center 단위로 이미 auto_arima가 선택한
    비계절 (p,d,q))를 입력으로 삼아, 그 결론과 별개로 SKU 단위 개별 SARIMA/SARIMAX를
    새로 구축하는 것을 명시적으로 확정함 — 기존 결론을 대체하지 않고 별도
    트랙으로 공존(기존 스크립트/산출물은 건드리지 않음).

Zero-Arbitrary Rule:
    - 비계절 (p,d,q) 및 절편(with_intercept)은 arima_development_orders.parquet의
      development_status=="selected" 값을 그대로 상속하고, 이 파이프라인에서 재탐색하지 않는다.
    - 계절주기 m 후보군(2026-08-21 "SARIMA/SARIMAX Strict Implementation Protocol" 최종 확정):
      개별 SKU 단위 ACF 사전 필터링을 적용하지 않고 도메인 주기 후보 {13,26,52} 전체를 AICc
      최소화 탐색에 개방한다(계절성이 불리한 SKU는 모델이 스스로 P=0,Q=0을 선택하도록 둠).
      단, 관측 이력 104주(m=52 두 주기) 미만이면 m=52를 식별할 표본이 안 되므로 {13,26}으로
      제한한다(m_candidates_for_history). 이 규칙은 A/B를 구분하지 않는 단일 길이 기준이며,
      B는 Post-regime 최대 26주라 이 규칙만으로도 자동으로 항상 {13,26}에 걸린다(중복 규칙
      불필요). 이전 버전의 ACF+periodogram 교차검증 사전 필터는 위 확정 프로토콜로 대체되어
      제거함(A/B 전체가 baseline(0,0,0,0)으로 붕괴하는 문제가 있었음 — 팀 확인 후 폐기).
    - S4 exog 중 공휴일_W0/W-1/W+1은 우선 data/ml/day2_statistical_models/exog/
      statistical_holiday_weekly.parquet(Specification 16절 지정 경로)을 로드해서 쓰고,
      그 파일이 아직 전달되지 않은 경우에만 feature_external_interaction_concat.py의
      KOREAN_LUNAR_HOLIDAYS(팀 확정 공휴일 캘린더)로 직접 계산하는 폴백을 쓰며 이 사실을
      preflight 체크리스트에 경고로 남긴다(조용히 넘어가지 않음).
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
import time
import warnings
from pathlib import Path

# 2026-08-21: CPU 부하가 100%인데 워커는 10개뿐인 현상 확인 — numpy/scipy/statsmodels가
# 프로세스당 내부적으로 BLAS 멀티스레딩(OpenMP/MKL/OpenBLAS)을 쓰는데, 여기에 joblib
# 프로세스 병렬화(n_jobs=10)가 겹쳐 스레드 과다경쟁(oversubscription)이 발생하는 전형적
# 패턴. numpy/scipy를 import하기 전에 반드시 설정해야 실제로 적용됨(import 시점에 스레드
# 풀을 구성하므로). 결과/파라미터에는 전혀 영향 없는 순수 실행환경 설정.
for _env_var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_env_var] = "1"

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from joblib.externals.loky import get_reusable_executor
from pmdarima.arima import nsdiffs
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tools.sm_exceptions import ConvergenceWarning

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[3]
DATA_DIR = BASE_DIR / "data"
ARIMA_ORDERS_PATH = DATA_DIR / "ml" / "day2_statistical_models" / "arima_development_orders.parquet"
DEV_PATH = DATA_DIR / "development_2021_2023.parquet"
HOLDOUT_PATH = DATA_DIR / "holdout_2024.parquet"
RESULT_DIR = DATA_DIR / "ml" / "day2_statistical_models" / "sarima_sarimax_sku"
# Specification 16절 지정 경로. 없으면 KOREAN_LUNAR_HOLIDAYS 직접계산으로 폴백(preflight 경고).
HOLIDAY_WEEKLY_PATH = DATA_DIR / "ml" / "day2_statistical_models" / "exog" / "statistical_holiday_weekly.parquet"

# 2026-08-21: n_jobs=-1(16개) + BLAS 멀티스레드 상태에서 메모리 누수/워커 반복재시작이
# 실측됐었음(CPU 부하 100%인데 워커는 10개뿐 -> BLAS 스레드 과다경쟁으로 확인). 이후
# BLAS 스레드를 프로세스당 1개로 고정(위 os.environ)했더니 SKU당 CPU시간이 줄어(25.7s->
# 21.3s), 오버서브스크립션 문제가 해소됐다고 판단해 워커 수를 다시 코어수(16)로 복원.
# gc.collect() + 청크마다 워커 강제재활용은 그대로 유지(메모리 누수 대응은 별도 조치).
N_JOBS = 16

# 원본 01_build_arima_orders.py와 동일 컨벤션: joblib 등 병렬화 없이 순차 루프 +
# CHECKPOINT_EVERY마다 원자적(tmp파일+os.replace) checkpoint 저장, 재실행 시 이미 처리된
# (center_id, sku_id)는 건너뛴다. 최종 audit 통과 후에만 정식 저장하고 checkpoint는 삭제.
CHECKPOINT_EVERY = 100  # 2026-08-21: 50개는 가시성은 좋았지만 청크마다(워커 강제 재활용
# 포함) 재시작 오버헤드가 너무 잦았음. 메모리가 안정화된 것을 확인한 뒤 100개로 완화.
DEV_MAX_WEEK = pd.Timestamp("2023-12-25")  # 2024 데이터 누출 방지 안전장치(원본과 동일 상수)

sys.path.insert(0, str(BASE_DIR / "src" / "ml" / "feature_engineering"))
from feature_external_interaction_concat import KOREAN_LUNAR_HOLIDAYS  # noqa: E402

CENTER_COL, SKU_COL, WEEK_COL, QTY_COL, TARGET_COL = "center_id", "sku_id", "week_st", "qty", "qty_log1p"

# Step1: B는 Post-regime(2023-07-03~2023-12-25)만 사용 — 고정값.
B_POST_REGIME_START = pd.Timestamp("2023-07-03")

# Step2: 계절 그리드 범위(고정값). D는 그리드서치 대상이 아니라
# nsdiffs(OCSB 검정, max_D=1)로 m마다 1회 자동 결정한다(2026-08-21 문서: "D=자동
# 결정(max_D=1)" — AICc로 D까지 같이 그리드서치하던 이전 버전을 이 방식으로 교체).
# 2026-08-21 "Apply Optimized Seasonal Grid & Accelerated Execution" 확정: P,Q 범위를
# [0,2]에서 [0,1]로 축소(150주 시계열 대비 과적합 방지 + 연산 가속, 팀 합의).
SEASONAL_P_RANGE = range(0, 2)
SEASONAL_Q_RANGE = range(0, 2)
SEASONAL_MAX_D = 1

# m 후보 도메인 전체 개방(2026-08-21 Strict Implementation Protocol 확정, ACF 사전 필터링
# 금지). 관측 이력이 MIN_HISTORY_FOR_M52(m=52 두 주기) 미만이면 m=52를 뺀다 — A/B를 구분하지
# 않는 단일 길이 기준이며, B는 Post-regime 최대 26주라 이 규칙만으로 항상 {13,26}에 걸린다.
M_DOMAIN_CANDIDATES = [13, 26, 52]
MIN_HISTORY_FOR_M52 = 104


def m_candidates_for_history(n_obs: int) -> list[int]:
    return M_DOMAIN_CANDIDATES if n_obs >= MIN_HISTORY_FOR_M52 else [m for m in M_DOMAIN_CANDIDATES if m != 52]

# Step3: S4 Operational-Full exog 7종(고정값). 공휴일 3종은 런타임에 계산해 채운다.
EXOG_COLS = ["ccsi_lag_m1", "cpi_y1_prev", "cpi_y2_prev_yoy", "covid_flag",
             "공휴일_W0", "공휴일_W-1", "공휴일_W+1"]

HORIZONS = [1, 2, 4]
MAX_HORIZON = 4


# ---------------------------------------------------------------------------
# 공통 유틸
# ---------------------------------------------------------------------------

def load_center_week_series(df: pd.DataFrame) -> pd.DataFrame:
    """statistical_preanalysis.py(Day1 Step4)의 동일 정의를 그대로 가져옴 — 그 모듈은
    matplotlib(플로팅 전용, 이 파이썬 환경에 미설치)까지 import하므로 전체 모듈을 불러오는
    대신 필요한 순수 pandas 함수 2개만 동일하게 복제(재정의 아님, 원본과 1:1 동일)."""
    holdout_start = pd.Timestamp("2024-01-01")
    dev = df[df[WEEK_COL] < holdout_start]
    return dev.groupby([CENTER_COL, WEEK_COL], observed=True)[QTY_COL].sum().reset_index()


def compute_periodogram(qty: np.ndarray) -> pd.DataFrame:
    """statistical_preanalysis.py와 동일 정의(scipy.signal.periodogram, fs=1.0, detrend=constant)."""
    from scipy.signal import periodogram
    freqs, power = periodogram(qty, fs=1.0, detrend="constant")
    keep = freqs > 0
    return pd.DataFrame({
        "period_weeks": 1.0 / freqs[keep],
        "power": power[keep],
    }).sort_values("period_weeks").reset_index(drop=True)


def week_holiday_flags(weeks: pd.Series) -> pd.DataFrame:
    """KOREAN_LUNAR_HOLIDAYS(팀 확정 캘린더)를 그대로 재사용해, 임의의 주(과거/미래 모두)에
    대해 W0(그 주 자체)/W-1(그 주 이전 1주)/W+1(그 주 다음 1주)가 명절 포함 주인지 계산한다.
    feature_external_interaction_concat.add_holiday_calendar_features와 동일 로직이되,
    target_h{h} 파생 없이 임의 주 자체를 직접 기준점으로 잡는다는 점만 다르다(재정의 아님 —
    같은 상수/같은 window 정의 재사용, target_date 자리에 원하는 주를 그대로 대입한 것)."""
    holiday_dates = pd.Series(KOREAN_LUNAR_HOLIDAYS)
    holiday_week_starts = holiday_dates - pd.to_timedelta(holiday_dates.dt.weekday, unit="D")
    holiday_weeks = set(holiday_week_starts)
    idx = weeks.index
    return pd.DataFrame({
        "공휴일_W0": weeks.isin(holiday_weeks).astype(float).to_numpy(),
        "공휴일_W-1": (weeks - pd.Timedelta(weeks=1)).isin(holiday_weeks).astype(float).to_numpy(),
        "공휴일_W+1": (weeks + pd.Timedelta(weeks=1)).isin(holiday_weeks).astype(float).to_numpy(),
    }, index=idx)


_HOLIDAY_WEEKLY_CACHE: pd.DataFrame | None = None
_HOLIDAY_WEEKLY_LOADED = False
HOLIDAY_COLS = ["공휴일_W0", "공휴일_W-1", "공휴일_W+1"]


def _load_holiday_weekly_once() -> pd.DataFrame | None:
    """Specification 16절 지정 경로(statistical_holiday_weekly.parquet)를 1회만 로드해
    캐시한다. 파일이 없으면 None(호출부가 week_holiday_flags 폴백으로 처리)."""
    global _HOLIDAY_WEEKLY_CACHE, _HOLIDAY_WEEKLY_LOADED
    if not _HOLIDAY_WEEKLY_LOADED:
        _HOLIDAY_WEEKLY_LOADED = True
        if HOLIDAY_WEEKLY_PATH.exists():
            df = pd.read_parquet(HOLIDAY_WEEKLY_PATH)
            df[WEEK_COL] = pd.to_datetime(df[WEEK_COL])
            _HOLIDAY_WEEKLY_CACHE = df.set_index(WEEK_COL)[HOLIDAY_COLS]
    return _HOLIDAY_WEEKLY_CACHE


def get_holiday_flags(weeks: pd.Series) -> pd.DataFrame:
    """statistical_holiday_weekly.parquet이 있으면 그 값을 조회해서 쓰고(지정 파일이 아직
    없는 주가 있으면 그 행만 개별적으로 KOREAN_LUNAR_HOLIDAYS 계산으로 보충), 파일 자체가
    없으면 전체를 week_holiday_flags()로 계산한다(둘 다 preflight 체크리스트에서 어느 쪽이
    쓰였는지 명시적으로 로그를 남김 — 조용한 폴백 금지)."""
    cache = _load_holiday_weekly_once()
    if cache is None:
        return week_holiday_flags(weeks)
    out = pd.DataFrame(index=weeks.index, columns=HOLIDAY_COLS, dtype=float)
    for col in HOLIDAY_COLS:
        out[col] = weeks.map(cache[col])
    missing = out.isna().any(axis=1)
    if missing.any():
        out.loc[missing, HOLIDAY_COLS] = week_holiday_flags(weeks[missing]).to_numpy()
    return out.astype(float)


def aicc(res) -> float:
    """statsmodels SARIMAXResults에 aicc가 없어 직접 계산: AICc = AIC + 2k(k+1)/(n-k-1)."""
    n = float(res.nobs)
    k = float(len(res.params))
    denom = n - k - 1
    if denom <= 0:
        return np.inf
    return float(res.aic + (2 * k * (k + 1)) / denom)


def fit_sarimax(y: np.ndarray, order: tuple, seasonal_order: tuple, trend, exog: np.ndarray | None = None):
    """수렴 경고를 캡처해 (res, converged) 반환. 실패 시 (None, False)."""
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            res = SARIMAX(
                y, exog=exog, order=order, seasonal_order=seasonal_order, trend=trend,
                simple_differencing=True, enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
        return res, converged
    except Exception:
        return None, False


# ---------------------------------------------------------------------------
# Step 1: 사전 합의된 규칙 및 입력 데이터 로드
# ---------------------------------------------------------------------------

def load_selected_orders() -> pd.DataFrame:
    df = pd.read_parquet(ARIMA_ORDERS_PATH)
    sel = df[df["development_status"] == "selected"].copy()
    sel["order_pdq"] = list(zip(
        sel["selected_p"].astype(int), sel["selected_d"].astype(int), sel["selected_q"].astype(int)
    ))
    sel["trend"] = np.where(sel["with_intercept"].astype(bool), "c", None)
    print(f"  [Step1] arima_development_orders.parquet: development_status=='selected' "
          f"{len(sel):,}/{len(df):,}행 (A={len(sel[sel.center_id=='A']):,}, "
          f"B={len(sel[sel.center_id=='B']):,})")
    return sel[[CENTER_COL, SKU_COL, "order_pdq", "trend", "with_intercept"]]


def load_dev_frame() -> pd.DataFrame:
    """A: 2021-01-04~2023-12-25 전체. B: 2023-07-03~2023-12-25(Post-regime)만.
    plain 공휴일_W0/W-1/W+1을 그 주 자체 기준으로 새로 계산해 붙인다(원본엔 없음).
    01_build_arima_orders.py(load_source)와 동일하게 2024 누출/윈도우 이탈을 방어적으로
    assert 재검증한다(입력 파일 자체가 이미 2021~2023뿐이라도 안전장치로 재확인)."""
    cols = [CENTER_COL, SKU_COL, WEEK_COL, QTY_COL, TARGET_COL, "ccsi_lag_m1", "cpi_y1_prev",
            "cpi_y2_prev_yoy", "covid_flag"]
    df = pd.read_parquet(DEV_PATH, columns=cols)
    assert df[WEEK_COL].max() <= DEV_MAX_WEEK, "development 입력에 2024 이후 데이터가 섞여 있음"

    b_out_of_window = df[(df[CENTER_COL] == "B") & (df[WEEK_COL] < B_POST_REGIME_START)]
    assert b_out_of_window.empty, (
        f"B센터에 Post-regime({B_POST_REGIME_START.date()}) 이전 데이터가 {len(b_out_of_window):,}행 "
        "존재 — development_2021_2023.parquet은 이미 B=Post-regime만 담고 있는 것으로 전제했는데 "
        "어긋남(01_build_arima_orders.py load_source()와 동일 방어적 검증)"
    )

    is_a = df[CENTER_COL] == "A"
    is_b_post = (df[CENTER_COL] == "B") & (df[WEEK_COL] >= B_POST_REGIME_START)
    out = df[is_a | is_b_post].copy()

    out[HOLIDAY_COLS] = get_holiday_flags(out[WEEK_COL])
    out = out.sort_values([CENTER_COL, SKU_COL, WEEK_COL]).reset_index(drop=True)
    print(f"  [Step1] development 로드: A 전체 + B Post-regime(>={B_POST_REGIME_START.date()}) "
          f"-> {len(out):,}행")
    return out


def load_holdout_frame() -> pd.DataFrame:
    cols = [CENTER_COL, SKU_COL, WEEK_COL, QTY_COL, TARGET_COL, "target_h1", "target_h2", "target_h4",
            "ccsi_lag_m1", "cpi_y1_prev", "cpi_y2_prev_yoy", "covid_flag"]
    df = pd.read_parquet(HOLDOUT_PATH, columns=cols)
    df[HOLIDAY_COLS] = get_holiday_flags(df[WEEK_COL])
    return df.sort_values([CENTER_COL, SKU_COL, WEEK_COL]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2.1: A센터 periodogram 진단 (정보 제공용, 더 이상 m 후보를 걸러내는 게이트가 아님)
# ---------------------------------------------------------------------------

def diagnose_a_periodogram() -> pd.DataFrame:
    """A센터 Center-week(전 SKU 합산) 시계열 periodogram 계산(statistical_preanalysis.py와
    동일 정의). 2026-08-21 Strict Implementation Protocol 확정: SKU 단위 ACF 사전
    필터링으로 m 후보를 걸러내지 않고 {13,26,52} 전체를 AICc 탐색에 개방하므로, 이 결과는
    더 이상 탐색 대상을 제한하지 않는 순수 진단/기록용 산출물이다(a_periodogram_diagnosis.csv)."""
    raw = pd.read_parquet(DATA_DIR / "final_feature_table.parquet", columns=[CENTER_COL, WEEK_COL, QTY_COL])
    series_df = load_center_week_series(raw)
    a_series = series_df[series_df[CENTER_COL] == "A"].sort_values(WEEK_COL)[QTY_COL].to_numpy(dtype=float)
    pgram = compute_periodogram(a_series)
    top3 = pgram.sort_values("power", ascending=False).head(3)
    print(f"  [Step2.1] A센터 periodogram 상위 3개 주기(정보용, 탐색 제한 없음): "
          + ", ".join(f"{r.period_weeks:.1f}주(p={r.power:.2e})" for r in top3.itertuples()))
    return pgram


# ---------------------------------------------------------------------------
# Step 2: SKU별 계절 구조 (P,D,Q,m) 탐색
# ---------------------------------------------------------------------------

def search_seasonal_structure(y: np.ndarray, order: tuple, trend, m_candidates: list[int]) -> dict | None:
    """order(p,d,q)/trend는 고정. 계절 없음(baseline, m 무관하게 1회만) + 각 m 후보에 대해
    D는 nsdiffs(OCSB 검정, max_D=1)로 먼저 1회 자동 결정하고 P/Q만 그리드(AICc)로 탐색해
    최소인 구조를 선택한다. 최적 후보가 미수렴이면 동일 탐색 내 수렴 후보 중 AICc 최소
    모델로 폴백(development_fallback_used 상당 플래그 기록)."""
    rows = []

    base_res, base_conv = fit_sarimax(y, order, (0, 0, 0, 0), trend)
    if base_res is not None:
        rows.append({"seasonal_order": (0, 0, 0, 0), "aicc": aicc(base_res), "converged": base_conv})

    for m in m_candidates:
        # 길이 기반 사전 필터링은 m_candidates_for_history()의 104주 규칙(m=52) 하나로
        # 충분(Strict Implementation Protocol 확정 — 추가 임의 컷오프 금지). m=13/26은
        # 표본이 짧아도 시도하고, 미수렴/실패는 아래 convergence 추적과 fit_sarimax의
        # try/except가 그대로 걸러낸다.
        try:
            D = int(nsdiffs(y, m=m, max_D=SEASONAL_MAX_D, test="ocsb"))
        except Exception:
            D = 0
        for P in SEASONAL_P_RANGE:
            for Q in SEASONAL_Q_RANGE:
                if P == 0 and D == 0 and Q == 0:
                    continue  # baseline과 중복(계절 없음은 m 무관하게 위에서 이미 평가)
                res, conv = fit_sarimax(y, order, (P, D, Q, m), trend)
                if res is None:
                    continue
                rows.append({"seasonal_order": (P, D, Q, m), "aicc": aicc(res), "converged": conv})

    if not rows:
        return None
    grid = pd.DataFrame(rows)
    converged = grid[grid["converged"]]
    pool = converged if len(converged) else grid
    best = pool.loc[pool["aicc"].idxmin()]
    global_best_converged = bool(grid.loc[grid["aicc"].idxmin(), "converged"])
    fallback_used = (not global_best_converged) and len(converged) > 0

    return {
        "seasonal_order": best["seasonal_order"],
        "aicc": float(best["aicc"]),
        "fallback_used": fallback_used,
        "n_candidates": len(grid),
        "n_converged": len(converged),
    }


# ---------------------------------------------------------------------------
# Step 3: Development 최종 1회 Joint Fit (SARIMA + SARIMAX S4)
# ---------------------------------------------------------------------------

def fit_development_models(y: np.ndarray, exog: np.ndarray, order: tuple, seasonal_order: tuple, trend):
    """SARIMA(exog 없음)와 SARIMAX-S4(exog=7종)를 각각 확정 구조로 Development 전체에 1회
    fit. Step2 grid search에서 이미 동일 (order,seasonal_order,trend)로 SARIMA를 적합했더라도,
    "복사하지 않고 공동추정"이 요구사항이므로 계수 재사용 없이 이 함수에서 새로 fit한다."""
    sarima_res, sarima_conv = fit_sarimax(y, order, seasonal_order, trend, exog=None)
    sarimax_res, sarimax_conv = fit_sarimax(y, order, seasonal_order, trend, exog=exog)
    return {
        "sarima": {"res": sarima_res, "converged": sarima_conv},
        "sarimax_s4": {"res": sarimax_res, "converged": sarimax_conv},
    }


# ---------------------------------------------------------------------------
# Step 4: 2024 Final Holdout 롤링 예측(No-Refit)
# ---------------------------------------------------------------------------

RESULT_COLS = [
    CENTER_COL, SKU_COL, "forecast_origin", "target_week", "horizon", "forecast_source",
    "selected_p", "selected_d", "selected_q", "selected_P", "selected_D", "selected_Q", "selected_m",
    "with_intercept", "state_update_success", "prediction_unclipped", "prediction", "actual",
]


def rolling_forecast_sku(center: str, sku: str, dev_sku_df: pd.DataFrame, holdout_sku_df: pd.DataFrame,
                          order: tuple, seasonal_order: tuple, trend, dev_fits: dict) -> list[dict]:
    """확정 구조로 fit된 Development 모델(SARIMA/SARIMAX-S4)에서 시작해, refit 없이
    append(state만 갱신)로 한 주씩 전진하며 h1/h2/h4를 추출한다. 매 append 직후 계수가
    Development fit과 동일한지 assert로 검증(Parameter Refit 절대 금지 원칙)."""
    rows: list[dict] = []
    n_eval = len(holdout_sku_df)
    if n_eval == 0:
        return rows

    for model_name, exog_cols in [("sarima", None), ("sarimax_s4", EXOG_COLS)]:
        entry = dev_fits[model_name]
        res = entry["res"]
        if res is None:
            continue
        base_params = res.params.copy()
        cur_res = res
        current_origin_week = dev_sku_df[WEEK_COL].iloc[-1] if len(dev_sku_df) else pd.NaT
        # ccsi_lag_m1: forecast origin에서 실제 확보된 최근 값을 t+1~t+4에 동일하게 고정
        # 공급 — cpi_y1_prev/cpi_y2_prev_yoy(항상 1~2년 전 발표치 참조라
        # 이미 확정된 값)/covid_flag/공휴일(달력 결정적)와 달리, ccsi_lag_m1은 각 미래
        # 주 자신의 달 기준으로 재계산하면 아직 발표 전인 값을 미리 아는 셈이 되어 금지.
        last_ccsi = float(dev_sku_df["ccsi_lag_m1"].iloc[-1]) if len(dev_sku_df) else np.nan
        ccsi_idx = EXOG_COLS.index("ccsi_lag_m1")

        for i in range(n_eval):
            steps = min(MAX_HORIZON, n_eval - i)

            if exog_cols is not None:
                future_block = holdout_sku_df.iloc[i:i + steps][exog_cols].to_numpy(dtype=float)
                if len(future_block) < steps:
                    break  # 미래 exog 부족(연말 경계) -> 이 시점부터 forecast 불가, 중단
                future_block[:, ccsi_idx] = last_ccsi
                try:
                    fc = cur_res.get_forecast(steps=steps, exog=future_block)
                except Exception:
                    break
            else:
                try:
                    fc = cur_res.get_forecast(steps=steps)
                except Exception:
                    break
            pred_log1p = np.asarray(fc.predicted_mean)

            batch = []
            for h in HORIZONS:
                if h > steps:
                    continue
                target_row = holdout_sku_df.iloc[i + h - 1]
                pred_unclipped = float(np.expm1(pred_log1p[h - 1]))
                prediction = max(pred_unclipped, 0.0)
                # 2026-08-22 버그 수정: target_h{h}는 그 행 "자신" 기준 h주 뒤로 이미
                # shift된 값이라, target_week 행에서 재조회하면 실제로는 target_week보다
                # h주 더 미래의 qty를 actual로 기록하게 됨(02_run_arima_holdout.py와 대조해
                # 858,885건 불일치로 발견/검증). target_row는 이미 target_week 시점 행이므로
                # raw qty를 직접 lookup(ARIMA와 동일 방식)한다 — target_h{h} 경유 폐기.
                actual = float(target_row[QTY_COL])
                batch.append({
                    CENTER_COL: center, SKU_COL: sku,
                    "forecast_origin": current_origin_week,
                    "target_week": target_row[WEEK_COL],
                    "horizon": h,
                    "forecast_source": model_name,
                    "selected_p": order[0], "selected_d": order[1], "selected_q": order[2],
                    "selected_P": seasonal_order[0], "selected_D": seasonal_order[1],
                    "selected_Q": seasonal_order[2], "selected_m": seasonal_order[3],
                    "with_intercept": (trend == "c"),
                    "prediction_unclipped": pred_unclipped,
                    "prediction": prediction,
                    "actual": actual,
                })

            row_i = holdout_sku_df.iloc[i]
            new_y = [float(row_i[TARGET_COL])]
            new_exog = row_i[exog_cols].to_numpy(dtype=float).reshape(1, -1) if exog_cols is not None else None
            state_ok = True
            try:
                cur_res = cur_res.append(new_y, exog=new_exog, refit=False)
                if not np.allclose(np.asarray(cur_res.params), np.asarray(base_params), equal_nan=True):
                    raise AssertionError(
                        f"refit 금지 위반: {center}/{sku}/{model_name} "
                        f"origin={current_origin_week} 계수가 append 후 변경됨"
                    )
            except AssertionError:
                raise
            except Exception as e:
                state_ok = False
                print(f"    [WARN] state 갱신 실패: {center}/{sku}/{model_name} "
                      f"origin={current_origin_week} ({type(e).__name__}: {e})")

            for r in batch:
                r["state_update_success"] = state_ok
            rows.extend(batch)

            current_origin_week = row_i[WEEK_COL]
            if exog_cols is not None:
                last_ccsi = float(row_i["ccsi_lag_m1"])
            if not state_ok:
                break

    return rows


def fallback_forecast_sku(center: str, sku: str, dev_sku_df: pd.DataFrame,
                           holdout_sku_df: pd.DataFrame) -> list[dict]:
    """development_status != 'selected' 이거나 SARIMA/SARIMAX fit 자체가 실패한 SKU.
    관측 이력이 전부 동일 값이면 constant, 아니면 naive_mean(A는 development 전체,
    B는 Post-regime만 이미 dev_sku_df 구성 단계에서 반영됨). Development 이력이 아예
    없는 신규 SKU(2024 신규 입점)는 안전한 기본값 0으로 constant 처리한다."""
    rows: list[dict] = []
    if len(dev_sku_df) == 0:
        source, level = "constant", 0.0
    else:
        y_raw = dev_sku_df[QTY_COL].to_numpy(dtype=float)
        if np.all(y_raw == y_raw[0]):
            source, level = "constant", float(y_raw[0])
        else:
            source, level = "naive_mean", float(y_raw.mean())

    n_eval = len(holdout_sku_df)
    prev_week = dev_sku_df[WEEK_COL].iloc[-1] if len(dev_sku_df) else pd.NaT
    for i in range(n_eval):
        origin_week = prev_week if i == 0 else holdout_sku_df.iloc[i - 1][WEEK_COL]
        for h in HORIZONS:
            if i + h - 1 >= n_eval:
                continue
            target_row = holdout_sku_df.iloc[i + h - 1]
            # 2026-08-22 버그 수정: 위 rolling_forecast_sku()와 동일한 이유로 target_h{h}
            # 대신 target_row의 raw qty를 직접 lookup(ARIMA와 동일 방식).
            actual = float(target_row[QTY_COL])
            rows.append({
                CENTER_COL: center, SKU_COL: sku,
                "forecast_origin": origin_week,
                "target_week": target_row[WEEK_COL],
                "horizon": h,
                "forecast_source": source,
                "selected_p": np.nan, "selected_d": np.nan, "selected_q": np.nan,
                "selected_P": np.nan, "selected_D": np.nan, "selected_Q": np.nan, "selected_m": np.nan,
                "with_intercept": np.nan,
                "state_update_success": True,
                "prediction_unclipped": level,
                "prediction": max(level, 0.0),
                "actual": actual,
            })
    return rows


def process_sku(center: str, sku: str, order: tuple, trend, dev_sku_df: pd.DataFrame,
                 holdout_sku_df: pd.DataFrame) -> tuple[list[dict], dict | None]:
    """SKU 1건에 대한 Step2->Step3->Step4 전체 파이프라인. (결과행 리스트, order_log dict) 반환.
    joblib으로 SKU 단위 병렬화 가능하도록 순수 함수로 분리(외부 상태 공유 없음)."""
    y = dev_sku_df[TARGET_COL].to_numpy(dtype=float)
    exog = dev_sku_df[EXOG_COLS].to_numpy(dtype=float)
    m_candidates = m_candidates_for_history(len(dev_sku_df))

    search = search_seasonal_structure(y, order, trend, m_candidates)
    if search is None:
        return fallback_forecast_sku(center, sku, dev_sku_df, holdout_sku_df), None

    seasonal_order = search["seasonal_order"]
    fits = fit_development_models(y, exog, order, seasonal_order, trend)
    if fits["sarima"]["res"] is None and fits["sarimax_s4"]["res"] is None:
        return fallback_forecast_sku(center, sku, dev_sku_df, holdout_sku_df), None

    rows = rolling_forecast_sku(center, sku, dev_sku_df, holdout_sku_df, order, seasonal_order, trend, fits)
    order_log = {
        CENTER_COL: center, SKU_COL: sku,
        "selected_p": order[0], "selected_d": order[1], "selected_q": order[2],
        "selected_P": seasonal_order[0], "selected_D": seasonal_order[1],
        "selected_Q": seasonal_order[2], "selected_m": seasonal_order[3],
        "with_intercept": (trend == "c"),
        "seasonal_aicc": search["aicc"], "seasonal_fallback_used": search["fallback_used"],
        "n_seasonal_candidates": search["n_candidates"], "n_seasonal_converged": search["n_converged"],
        "sarima_converged": fits["sarima"]["converged"], "sarimax_s4_converged": fits["sarimax_s4"]["converged"],
        "n_dev_obs": len(dev_sku_df),
    }
    return rows, order_log


def process_sku_safe(center: str, sku: str, order: tuple, trend, dev_sku_df: pd.DataFrame,
                      holdout_sku_df: pd.DataFrame) -> tuple[str, str, list[dict], dict | None, float]:
    """joblib.Parallel(delayed(...)) 워커에서 실행되는 래퍼. 예외가 나도 워커 전체가 죽지
    않도록 잡아서 constant/naive_mean 폴백으로 대체한다(에러는 로그로만 남김).
    2026-08-21: SKU 1건 처리마다 명시적 gc.collect()를 호출해 statsmodels/pmdarima가
    fit()마다 만드는 대량의 중간 배열(Kalman filter 상태행렬, 최적화 과정의 임시 객체 등)이
    워커 프로세스 안에 누적되는 것을 완화한다(메모리 누수/워커 반복 재시작 실측 대응)."""
    t0 = time.time()
    try:
        rows, order_log = process_sku(center, sku, order, trend, dev_sku_df, holdout_sku_df)
    except Exception as e:
        print(f"    [ERROR] {center}/{sku} 처리 중 예외 발생, fallback으로 대체: {type(e).__name__}: {e}")
        rows, order_log = fallback_forecast_sku(center, sku, dev_sku_df, holdout_sku_df), None
    gc.collect()
    return center, sku, rows, order_log, time.time() - t0


# ---------------------------------------------------------------------------
# Phase 1: 확정 원칙 사전 점검 체크리스트 (본 실행 전 자체 진단)
# ---------------------------------------------------------------------------

def preflight_checklist() -> None:
    """"SARIMA/SARIMAX Strict Implementation Protocol"(2026-08-21) Phase 1 체크리스트를
    코드 레벨에서 검증하고 로그로 남긴다. 하나라도 위반이면 즉시 AssertionError로 중단한다
    (본 실행을 시작하기 전에 걸러내는 게 목적이라 Step1 데이터 로드 직후, SKU 루프 이전에
    호출). rolling_forecast_sku 등 함수 소스를 inspect로 직접 들여다봐서 "코드가 실제로
    그렇게 되어 있는지"를 확인하는 항목은 문자열 존재 여부로 검증(구조적 보증의 근사치이며
    완전한 정적 분석은 아니지만, 조용히 어긋나는 것보다는 낫다는 원칙으로 채택)."""
    import inspect

    print("=" * 80)
    print("[Phase 1] 확정 원칙 사전 점검 체크리스트")

    rolling_src = inspect.getsource(rolling_forecast_sku)
    fallback_src = inspect.getsource(fallback_forecast_sku)

    # 1) P3/P9/P20 No-Refit & Data Leakage
    ok = "refit=False" in rolling_src and "refit=True" not in rolling_src
    print(f"  [1] No-Refit(refit=False, 2024 전 구간 coefficient 불변) 코드 검증: {'PASS' if ok else 'FAIL'}")
    assert ok, "rolling_forecast_sku가 refit=False를 쓰지 않거나 refit=True 흔적이 있음"
    ok = "np.allclose" in rolling_src and "base_params" in rolling_src
    print(f"  [1] append 후 계수 불변 assert(np.allclose vs base_params) 존재: {'PASS' if ok else 'FAIL'}")
    assert ok, "append 직후 계수 불변 검증 assert가 코드에 없음"
    print("  [1] 2024 신규 관측치/exog는 state 갱신에만 사용, order/exog조합/fallback 규칙 사후변경 없음: "
          "PASS (Step2~3에서 확정한 구조를 그대로 dev_fits로 넘겨받아 Step4에서 재탐색하지 않음)")

    # 2) P5/P11 B센터 Regime 격리
    ok = B_POST_REGIME_START == pd.Timestamp("2023-07-03")
    print(f"  [2] B센터 Post-regime 시작일=2023-07-03 고정: {'PASS' if ok else 'FAIL'}")
    assert ok, "B_POST_REGIME_START 상수가 어긋남"
    ok = m_candidates_for_history(26) == [13, 26] and 52 not in m_candidates_for_history(26)
    print(f"  [2] B센터(이력<=26주) m 후보에서 m=52 배제: {'PASS' if ok else 'FAIL'} "
          f"(m_candidates_for_history(26)={m_candidates_for_history(26)})")
    assert ok, "B센터 이력 길이에서 m=52가 후보에 남아있음"

    # 3) 비계절 차수 고정 및 공동 추정
    load_src = inspect.getsource(load_selected_orders)
    ok = "development_status" in load_src and 'sel["selected_p"]' in load_src and "auto_selected_order" not in load_src
    print(f"  [3] arima_development_orders.parquet: development_status=='selected' 필터 + "
          f"selected_p/d/q(fallback 반영 최종값) 사용, auto_selected_order 미사용: {'PASS' if ok else 'FAIL'}")
    assert ok, "load_selected_orders가 auto_selected_order를 쓰거나 selected_p/d/q를 안 씀"
    fit_dev_src = inspect.getsource(fit_development_models)
    ok = "fit_sarimax(y, order, seasonal_order, trend" in fit_dev_src
    print(f"  [3] Development 전체로 SARIMA/SARIMAX 1회 공동 추정(계수 복사 아님): {'PASS' if ok else 'FAIL'}")
    assert ok, "fit_development_models이 새로 fit하지 않고 계수를 복사해오는 것으로 의심됨"

    # 4) Exog 공급 및 배제
    expected_exog = ["ccsi_lag_m1", "cpi_y1_prev", "cpi_y2_prev_yoy", "covid_flag",
                      "공휴일_W0", "공휴일_W-1", "공휴일_W+1"]
    ok = EXOG_COLS == expected_exog
    print(f"  [4] S4 Operational-Full exog 7종(Econ3+COVID1+Holiday3) 정확 일치: {'PASS' if ok else 'FAIL'}")
    assert ok, f"EXOG_COLS가 S4 스펙과 다름: {EXOG_COLS}"
    weather_terms = ["평균온도", "총강수량", "강수량_호우"]
    ok = not any(w in col for col in EXOG_COLS for w in weather_terms)
    print(f"  [4] 날씨 변수 완전 배제: {'PASS' if ok else 'FAIL'}")
    assert ok, "EXOG_COLS에 날씨 변수가 섞여 있음"
    if HOLIDAY_WEEKLY_PATH.exists():
        print(f"  [4] 공휴일 exog: {HOLIDAY_WEEKLY_PATH} 로드해서 사용: PASS")
    else:
        print(f"  [4] 공휴일 exog: {HOLIDAY_WEEKLY_PATH} 미존재 -> KOREAN_LUNAR_HOLIDAYS 직접계산 폴백 사용 "
              f"[경고 — Specification 16절 지정 파일이 아직 전달되지 않음]")
    ok = "future_block" in rolling_src and "for i in range(n_eval)" in rolling_src
    print(f"  [4] t+1~t+4 4-step 전체 exog 공급(h1/h2/h4만 저장해도 h3용 exog도 생성): {'PASS' if ok else 'FAIL'}")
    assert ok

    # 5) 후처리 및 Fallback 일관성
    ok = "np.expm1" in rolling_src and "max(pred_unclipped, 0.0)" in rolling_src
    print(f"  [5] expm1 복원 + 0 클리핑: {'PASS' if ok else 'FAIL'}")
    assert ok, "rolling_forecast_sku에 expm1/클리핑 로직이 없음"
    ok = "constant" in fallback_src and "naive_mean" in fallback_src and "np.all(y_raw == y_raw[0])" in fallback_src
    print(f"  [5] Fallback 규칙(전부 동일값=constant, 아니면 naive_mean): {'PASS' if ok else 'FAIL'}")
    assert ok, "fallback_forecast_sku의 constant/naive_mean 판정 로직이 없음"

    print("[Phase 1] 전 항목 PASS — 본 실행 진행")


# ---------------------------------------------------------------------------
# Step 5: 산출물 저장 및 무결성 감사
# ---------------------------------------------------------------------------

def audit_results(result_df: pd.DataFrame) -> None:
    print(f"  [Step5 Audit] 총 {len(result_df):,}행")

    key_cols = [CENTER_COL, SKU_COL, "forecast_origin", "horizon", "forecast_source"]
    dup = result_df.duplicated(subset=key_cols).sum()
    print(f"    - (center,sku,origin,horizon,source) 키 유일성: 중복 {dup}건 "
          f"{'OK' if dup == 0 else '!! 위반'}")
    assert dup == 0, "결과 키 유일성 위반"

    for col in ["prediction", "prediction_unclipped"]:
        n_nan = result_df[col].isna().sum()
        n_inf = np.isinf(result_df[col].to_numpy(dtype=float)).sum()
        print(f"    - {col}: NaN {n_nan}건, Inf {n_inf}건")

    n_neg = (result_df["prediction"] < 0).sum()
    print(f"    - prediction 음수(클리핑 후): {n_neg}건 {'OK' if n_neg == 0 else '!! 위반'}")
    assert n_neg == 0, "클리핑 후에도 음수 prediction 존재"

    fail_state = (~result_df["state_update_success"]).sum()
    print(f"    - state_update_success == False: {fail_state}건")

    allowed_sources = {"sarima", "sarimax_s4", "naive_mean", "constant"}
    unknown_sources = set(result_df["forecast_source"].unique()) - allowed_sources
    assert not unknown_sources, f"알 수 없는 forecast_source: {unknown_sources}"

    no_order_mask = result_df["forecast_source"].isin(["naive_mean", "constant"])
    bad_no_order = result_df.loc[no_order_mask, ["selected_p", "selected_d", "selected_q"]].notna().any(axis=1)
    assert not bad_no_order.any(), f"naive_mean/constant인데 selected order 존재: {int(bad_no_order.sum())}건"

    print(f"    - forecast_source 분포:\n{result_df['forecast_source'].value_counts().to_string()}")


# ---------------------------------------------------------------------------
# 체크포인트 (01_build_arima_orders.py와 동일 컨벤션: 순차 루프 + 원자적 저장/재개)
# ---------------------------------------------------------------------------

def load_checkpoint(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def save_checkpoint(df: pd.DataFrame, path: Path) -> None:
    """중단 시 손상된 checkpoint가 남지 않도록 임시 파일에 쓰고 원자적으로 교체한다."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, path)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", type=int, default=0,
                     help="센터별 selected SKU 중 앞 N개만 사용(파일럿 검증용, 0이면 전체 18,371개 본 실행)")
    args = ap.parse_args()

    t0 = time.time()
    print("=" * 80)
    print("[Step1] 사전 합의된 규칙 및 입력 데이터 로드")
    orders = load_selected_orders()
    dev_df = load_dev_frame()
    holdout_df = load_holdout_frame()

    preflight_checklist()

    print("=" * 80)
    print("[Step2.1] A센터 periodogram 진단(정보용)")
    pgram_df = diagnose_a_periodogram()

    if args.pilot > 0:
        orders = (
            orders.groupby(CENTER_COL, group_keys=False)
            .apply(lambda g: g.head(args.pilot))
        )
        print(f"  [PILOT MODE] 센터별 앞 {args.pilot}개 SKU만 사용 -> 총 {len(orders):,}개")

    dev_groups = {k: v for k, v in dev_df.groupby([CENTER_COL, SKU_COL], observed=True)}
    holdout_groups = {k: v for k, v in holdout_df.groupby([CENTER_COL, SKU_COL], observed=True)}

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"_pilot{args.pilot}" if args.pilot > 0 else ""
    result_path = RESULT_DIR / f"forecast_results{suffix}.parquet"
    order_log_path = RESULT_DIR / f"seasonal_order_selection{suffix}.csv"
    pgram_path = RESULT_DIR / "a_periodogram_diagnosis.csv"
    forecast_ckpt_path = RESULT_DIR / f"forecast_results{suffix}.checkpoint.parquet"
    order_log_ckpt_path = RESULT_DIR / f"seasonal_order_selection{suffix}.checkpoint.parquet"

    # 01_build_arima_orders.py와 동일하게 checkpoint를 먼저 로드해 이미 처리된
    # (center_id, sku_id)는 재실행 시 건너뛴다. SKU 단위 처리 자체는 joblib(n_jobs=-1)로
    # CHECKPOINT_EVERY개씩 청크 병렬화(Strict Implementation Protocol 확정)하되, 청크가
    # 끝날 때마다 원자적으로 checkpoint를 남겨 중단/재개가 여전히 안전하게 동작한다.
    ckpt_forecast_df = load_checkpoint(forecast_ckpt_path)
    ckpt_order_log_df = load_checkpoint(order_log_ckpt_path)
    all_rows: list[dict] = ckpt_forecast_df.to_dict("records") if len(ckpt_forecast_df) else []
    order_logs: list[dict] = ckpt_order_log_df.to_dict("records") if len(ckpt_order_log_df) else []
    done_keys = set(zip(ckpt_forecast_df.get(CENTER_COL, []), ckpt_forecast_df.get(SKU_COL, [])))
    if done_keys:
        print(f"  [Checkpoint] 이전 실행분 {len(done_keys):,}개 SKU 재사용, 이어서 진행")

    print("=" * 80)
    print(f"[Step2~4] SKU {len(orders):,}개 joblib(n_jobs={N_JOBS}) 병렬 처리 시작"
          f"(체크포인트 매 {CHECKPOINT_EVERY}개 청크)")
    per_sku_times: list[float] = []

    def flush_checkpoint():
        save_checkpoint(pd.DataFrame(all_rows, columns=RESULT_COLS), forecast_ckpt_path)
        save_checkpoint(pd.DataFrame(order_logs), order_log_ckpt_path)

    pending = []
    for rec in orders.itertuples(index=False):
        center, sku, order, trend = rec.center_id, rec.sku_id, rec.order_pdq, rec.trend
        if (center, sku) in done_keys:
            continue
        dev_sku = dev_groups.get((center, sku))
        holdout_sku = holdout_groups.get((center, sku))
        if dev_sku is None or holdout_sku is None:
            continue
        pending.append((center, sku, order, trend, dev_sku, holdout_sku))

    for chunk_start in range(0, len(pending), CHECKPOINT_EVERY):
        chunk = pending[chunk_start:chunk_start + CHECKPOINT_EVERY]
        results = Parallel(n_jobs=N_JOBS)(delayed(process_sku_safe)(*task) for task in chunk)
        for center, sku, rows, order_log, elapsed in results:
            per_sku_times.append(elapsed)
            all_rows.extend(rows)
            if order_log is not None:
                order_logs.append(order_log)
            done_keys.add((center, sku))

        # 2026-08-21 메모리 누수 대응: 청크 끝날 때마다 loky 워커 풀을 강제 종료하고 다음
        # 청크에서 새로 띄운다. 개별 워커가 여러 청크에 걸쳐 계속 살아남으며 메모리를
        # 누적하는 것(실측: 36분에 여유메모리 ~1.5GB 감소, 워커 CPU시간 편차로 반복
        # 재시작 정황 확인)을 청크 경계에서 확실히 끊어낸다. kill_workers=True로 대기 중인
        # 워커까지 전부 종료(다음 Parallel() 호출 시 자동으로 새 프로세스가 뜸).
        get_reusable_executor(max_workers=N_JOBS if N_JOBS > 0 else None).shutdown(kill_workers=True)
        gc.collect()

        flush_checkpoint()
        n_done = min(chunk_start + CHECKPOINT_EVERY, len(pending))
        elapsed_total = time.time() - t0
        avg = np.mean(per_sku_times) if per_sku_times else 0.0
        wall_rate = n_done / elapsed_total if elapsed_total > 0 else 0.0  # 실측 병렬 처리량(SKU/s)
        remaining = len(pending) - n_done
        eta_str = f"{remaining / wall_rate / 60:.1f}분" if wall_rate > 0 else "N/A"
        print(f"    {n_done}/{len(pending)} 완료 (경과 {elapsed_total:.1f}s, SKU당 CPU평균 {avg:.3f}s, "
              f"실측처리량 {wall_rate:.3f} SKU/s, 잔여 ETA {eta_str})")

    print("=" * 80)
    print("[Step4 Fallback] development_status != 'selected'(short_history/constant/미수렴 등)이거나"
          " arima_development_orders.parquet에 아예 없는 2024 신규 SKU — 새 order 탐색 없이"
          " constant/naive_mean 폴백만 적용(문서 5번: ARIMA와 동일한 fallback 정책)")
    processed_keys = set(zip(orders[CENTER_COL].tolist(), orders[SKU_COL].tolist()))
    fallback_keys = [k for k in holdout_groups.keys() if k not in processed_keys and k not in done_keys]
    if args.pilot > 0:
        fallback_keys = fallback_keys[: args.pilot * 2]
        print(f"  [PILOT MODE] 폴백 대상도 앞 {len(fallback_keys)}개만 처리")
    empty_dev = dev_df.iloc[0:0]
    n_since_ckpt = 0
    for n_fb, (center, sku) in enumerate(fallback_keys, start=1):
        dev_sku = dev_groups.get((center, sku), empty_dev)
        holdout_sku = holdout_groups[(center, sku)]
        all_rows.extend(fallback_forecast_sku(center, sku, dev_sku, holdout_sku))
        done_keys.add((center, sku))
        n_since_ckpt += 1

        if n_since_ckpt >= CHECKPOINT_EVERY:
            flush_checkpoint()
            n_since_ckpt = 0

        if n_fb % max(1, len(fallback_keys) // 5) == 0 or n_fb == len(fallback_keys):
            print(f"    폴백 {n_fb}/{len(fallback_keys)} 완료")

    result_df = pd.DataFrame(all_rows, columns=RESULT_COLS)
    order_log_df = pd.DataFrame(order_logs)

    print("=" * 80)
    print("[Step5] 무결성 감사 및 저장")
    audit_results(result_df)  # 여기서 assert 실패하면 최종 저장/checkpoint 삭제 없이 즉시 중단(원본 run_audit와 동일 원칙)

    result_df.to_parquet(result_path, index=False)
    order_log_df.to_csv(order_log_path, index=False, encoding="utf-8-sig")
    pgram_df.to_csv(pgram_path, index=False, encoding="utf-8-sig")

    for p in [forecast_ckpt_path, order_log_ckpt_path]:
        if p.exists():
            p.unlink()

    total_elapsed = time.time() - t0
    avg_per_sku = np.mean(per_sku_times) if per_sku_times else 0.0
    print("=" * 80)
    print(f"[완료] SKU {len(orders):,}개, 총 {total_elapsed:.1f}s, SKU당 평균 {avg_per_sku:.3f}s")
    print(f"  결과 저장 -> {result_path}")
    print(f"  order log -> {order_log_path}")
    print(f"  periodogram -> {pgram_path}")

    n_selected_total = 18371  # arima_development_orders.parquet development_status=='selected' 총합
    if avg_per_sku > 0:
        est_full_hours = avg_per_sku * n_selected_total / 3600
        print(f"  [전체 스케일 추정] SKU당 평균 {avg_per_sku:.3f}s x {n_selected_total:,}개 "
              f"= 약 {est_full_hours:.1f}시간(단일 프로세스, 원본 컨벤션대로 병렬화 없음)")


if __name__ == "__main__":
    main()
