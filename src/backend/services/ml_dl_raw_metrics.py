"""2023 P13 HPO 원본 결과 JSON(outputs/hpo/*, outputs/hurdle_hpo/*)에서
trial_id 기준으로 pooled_mae/pooled_rmse만 조회한다. ml_dl_model_selection_2023.csv에는
mae/rmse 컬럼이 없어서 만든 보조 조회기이며, 절대 값을 계산·추정하지 않고
원본 JSON에 실제로 존재하는 값만 반환한다(trial_id 불일치 또는 pooled_wape 불일치 시 미제공)."""

import json
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

_FAMILY_FILES = {
    "RF":              ("outputs/hpo/rf/p13_rf_h{h}.json", "all_trials"),
    "LightGBM":        ("outputs/hpo/lgbm/p13_lightgbm_h{h}.json", "all_trials"),
    "LSTM":            ("outputs/hpo/lstm/p13_lstm_h{h}.json", "all_trials"),
    "TFT":             ("outputs/hpo/tft/p13_tft_h{h}.json", "all_trials"),
    "Informer":        ("outputs/hpo/informer/p13_informer_h{h}.json", "all_trials"),
    "Hurdle-RF":       ("outputs/hurdle_hpo/rf_hurdle/p13_hurdle_rf_h{h}.json", "all_combinations"),
    "Hurdle-LightGBM": ("outputs/hurdle_hpo/lgbm_hurdle/p13_hurdle_lightgbm_h{h}.json", "all_trials"),
}

_WAPE_TOLERANCE = 0.01


@lru_cache(maxsize=None)
def _load_trial_index(model: str, horizon: int) -> dict[int, dict]:
    """{trial_id: {pooled_wape, pooled_mae, pooled_rmse}} 또는 파일이 없으면 빈 dict."""
    entry = _FAMILY_FILES.get(model)
    if entry is None:
        return {}
    path_tpl, list_key = entry
    path = REPO_ROOT / path_tpl.format(h=horizon)
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    trials = data.get(list_key) or []
    return {
        int(t["trial_id"]): {
            "pooled_wape": t.get("pooled_wape"),
            "pooled_mae": t.get("pooled_mae"),
            "pooled_rmse": t.get("pooled_rmse"),
        }
        for t in trials
        if "trial_id" in t
    }


def get_mae_rmse(model: str, horizon: int, trial_id: int, expected_wape: float):
    """(mae, rmse, status). trial_id가 파일에 존재하고 그 trial의 pooled_wape가
    CSV 대표값(expected_wape)과 오차 0.01 이내로 일치할 때만 ('mae','rmse','ok')를 반환한다.
    그 외에는 전부 (None, None, 'unavailable') — 임의 계산/대체 금지."""
    index = _load_trial_index(model, horizon)
    row = index.get(trial_id)
    if row is None:
        return None, None, "unavailable"
    wape = row.get("pooled_wape")
    if wape is None or abs(float(wape) - float(expected_wape)) >= _WAPE_TOLERANCE:
        return None, None, "unavailable"
    mae, rmse = row.get("pooled_mae"), row.get("pooled_rmse")
    if mae is None or rmse is None:
        return None, None, "unavailable"
    return float(mae), float(rmse), "ok"
