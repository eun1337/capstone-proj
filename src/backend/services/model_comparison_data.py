"""outputs/model_comparison/*.csv 공용 로더 — 프로세스당 1회만 읽고 캐시한다.
모델링/전처리 산출물은 read-only로만 사용하며 이 파일은 산출물을 생성하지 않는다."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

OUTPUTS_DIR = Path(__file__).resolve().parents[3] / "outputs" / "model_comparison"

MODEL_LABELS = {
    "ARIMA_S0": "ARIMA",
    "ARIMAX_S1": "ARIMAX-S1",
    "ARIMAX_S2": "ARIMAX-S2",
    "ARIMAX_S3": "ARIMAX-S3",
    "ARIMAX_S4": "ARIMAX-S4",
    "SARIMA": "SARIMA",
    "SARIMAX_S4": "SARIMAX",
}
MODEL_ORDER = list(MODEL_LABELS.keys())

EXTREME_MODELS = {"ARIMAX_S1", "ARIMAX_S4"}

HORIZONS = [1, 2, 4]
METRICS = ["WAPE", "Bias", "MAE", "RMSE"]

MODEL_DETAILS = {
    "ARIMA_S0": {
        "input": "SKU x 센터 단위 주간 판매수량",
        "structure": "비계절 (p,d,q), exog 없음",
        "selection": "auto_arima 후보 중 AICc 최소화 (미수렴 시 수렴 후보 중 최저 AICc)",
        "characteristics": "외생변수 없는 순수 시계열 baseline",
    },
    "ARIMAX_S1": {
        "input": "ARIMA_S0 입력 + 경제지표(Economic) exog",
        "structure": "비계절 (p,d,q) + 경제 exog block(S1)",
        "selection": "ARIMA_S0와 동일 (p,d,q) 상속, 경제 exog만 추가 적합",
        "characteristics": "일부 SKU에서 계수 발산으로 WAPE가 극단적으로 커짐(fallback_rate 높음)",
    },
    "ARIMAX_S2": {
        "input": "ARIMA_S0 입력 + COVID 지표 exog",
        "structure": "비계절 (p,d,q) + COVID exog block(S2)",
        "selection": "ARIMA_S0와 동일 (p,d,q) 상속, COVID exog만 추가 적합",
        "characteristics": "ARIMA_S0 대비 근소한 차이, A/B 센터 모두 비교적 안정",
    },
    "ARIMAX_S3": {
        "input": "ARIMA_S0 입력 + 공휴일 exog",
        "structure": "비계절 (p,d,q) + 공휴일 exog block(S3)",
        "selection": "ARIMA_S0와 동일 (p,d,q) 상속, 공휴일 exog만 추가 적합",
        "characteristics": "ARIMA_S0 대비 근소한 차이, A/B 센터 모두 비교적 안정",
    },
    "ARIMAX_S4": {
        "input": "ARIMA_S0 입력 + 경제·COVID·공휴일 exog 전체(Operational-Full)",
        "structure": "비계절 (p,d,q) + 경제+COVID+공휴일 exog block(S4)",
        "selection": "ARIMA_S0와 동일 (p,d,q) 상속, 전체 exog 동시 적합",
        "characteristics": "exog 수가 많아 계수 불안정·발산 빈도가 가장 높음",
    },
    "SARIMA": {
        "input": "SKU x 센터 단위 주간 판매수량",
        "structure": "(p,d,q) + 계절 (P,D,Q,m), m ∈ {13,26,52} 후보, exog 없음",
        "selection": "AICc 최소화(계절주기 m 포함 탐색), 비계절 (p,d,q)는 development_status=selected 상속",
        "characteristics": "exog 미사용, h1~h4 전 구간에서 WAPE가 비교적 좁은 범위로 안정적",
    },
    "SARIMAX_S4": {
        "input": "SARIMA 입력 + 경제·COVID·공휴일 exog 전체(Operational-Full, 7종)",
        "structure": "(p,d,q) + 계절 (P,D,Q,m) + exog block(S4)",
        "selection": "SARIMA와 동일 구조 상속 + S4 exog 7종(Econ3+COVID1+Holiday3) 추가 적합",
        "characteristics": "계절성 + exog 결합, SARIMA 대비 WAPE가 다소 높고 변동폭이 큼",
    },
}

@lru_cache(maxsize=None)
def load_stat_common_metrics() -> pd.DataFrame:
    """model x center x horizon 단위 통계모델 공통 지표(WAPE/Bias/MAE/RMSE 등)."""
    return pd.read_csv(OUTPUTS_DIR / "stat_common_metrics_2024.csv")
