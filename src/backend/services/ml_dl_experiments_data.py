"""Hurdle 구조 도입 전 실제로 수행된 진단/비교 실험 결과를 원본 JSON에서 읽어 제공한다.
모든 수치는 파일에서 직접 로드하며(계산/추정 없음), 파일이 없으면 해당 항목만 조용히 생략한다.

동일 평가조건 확인(2026-09 조사, 5개 파일 전부 horizon=1 · common_eval_key_count=577319 ·
동일 five_family_common_evaluation_keys.parquet 모집단으로 확인됨) → 아래 EXPERIMENT_ENTRIES(5-variant)는
하나의 표/막대로 직접 비교 가능. LSTM raw-scale/B센터/seed 로버스트니스는 모델·HP grid 자체가 달라
NARRATIVE_NOTES로만 제공한다(수치 막대 비교에 섞지 않음)."""

import json
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(rel_path: str) -> dict | None:
    path = REPO_ROOT / rel_path
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _trial(data: dict, list_key: str, trial_id: int) -> dict | None:
    for t in data.get(list_key) or []:
        if int(t.get("trial_id", -1)) == trial_id:
            return t
    return None


@lru_cache(maxsize=None)
def load_experiment_entries() -> list[dict]:
    """LightGBM h1, 동일 평가모집단(577319행) 내에서 target-transform/Hurdle/Tweedie 대안을
    비교한 5개 variant. 전부 outputs/hpo, outputs/diagnostics 원본 값 그대로."""
    official = _load("outputs/hpo/lgbm/p13_lightgbm_h1.json")
    rawscale = _load("outputs/hpo/_diag_rawscale_lightgbm_h1/p13_lightgbm_h1_rawscale.json")
    hurdle_diag = _load("outputs/diagnostics/target_transform_bias/lightgbm_h1_hurdle_diagnostic.json")
    target_scale = _load("outputs/diagnostics/target_transform_bias/lightgbm_h1_hurdle_target_scale_diagnostic.json")
    tweedie = _load("outputs/diagnostics/target_transform_bias/lightgbm_h1_hurdle_tweedie_diagnostic.json")

    variants = []

    official_t3 = _trial(official, "all_trials", 3) if official else None
    if official_t3:
        variants.append({
            "key": "log1p_official", "label": "공식 log1p (trial3)",
            "wape": float(official_t3["pooled_wape"]), "bias": float(official_t3["pooled_bias"]), "adopted": False,
        })

    rawscale_t3 = _trial(rawscale, "all_trials", 3) if rawscale else None
    if rawscale_t3:
        variants.append({
            "key": "raw_scale", "label": "raw-scale (trial3, 동일 HP)",
            "wape": float(rawscale_t3["pooled_wape"]), "bias": float(rawscale_t3["pooled_bias"]), "adopted": False,
        })

    if hurdle_diag:
        pred_hurdle = hurdle_diag.get("pooled_metrics_common", {}).get("pred_hurdle")
        if pred_hurdle:
            variants.append({
                "key": "hurdle_raw_regressor", "label": "Hurdle · 회귀단계 raw",
                "wape": float(pred_hurdle["wape"]), "bias": float(pred_hurdle["bias"]), "adopted": False,
            })

    if target_scale:
        pred_log = target_scale.get("pooled_metrics_common", {}).get("pred_hurdle_log")
        if pred_log:
            variants.append({
                "key": "hurdle_log_regressor", "label": "Hurdle · 회귀단계 log1p (채택)",
                "wape": float(pred_log["wape"]), "bias": float(pred_log["bias"]), "adopted": True,
            })

    if tweedie:
        tw_metrics = tweedie.get("pooled_metrics_common", {})
        tw_powers = tweedie.get("tweedie_variance_powers", [])
        for p in tw_powers:
            key = f"pred_tweedie_{p}"
            m = tw_metrics.get(key)
            if m:
                variants.append({
                    "key": key, "label": f"Tweedie (variance_power={p})",
                    "wape": float(m["wape"]), "bias": float(m["bias"]), "adopted": False,
                })

    if not variants:
        return []

    return [{
        "id": "lightgbm_h1_target_transform",
        "title": "LightGBM h1 — target transform / Hurdle 구조 비교",
        "hypothesis": "log1p→expm1 재변환 편향이 음의 Bias의 원인인지, 어떤 방식으로 해소되는지 검증",
        "changed": "동일 feature·4-fold·평가모집단(577,319행)에서 target 변환 방식과 모델 구조만 변경",
        "eval_note": "5개 variant 전부 horizon=1, common evaluation population 577,319행 동일 확인됨 → 직접 비교 가능",
        "variants": variants,
        "conclusion": (
            "raw-scale은 Bias는 0에 가까워지지만 WAPE가 악화되어 채택하지 않음. "
            "Hurdle(분류+조건부회귀) 구조에 회귀단계 log1p를 적용한 조합이 WAPE·Bias 모두 가장 양호해 "
            "실제 P13 Hurdle-LightGBM h1 selected trial과 동일한 값으로 채택됨. "
            "Tweedie 단일모델은 이 조합보다 WAPE가 전부 높아 제외."
        ),
        "source_paths": [
            "outputs/hpo/lgbm/p13_lightgbm_h1.json",
            "outputs/hpo/_diag_rawscale_lightgbm_h1/p13_lightgbm_h1_rawscale.json",
            "outputs/diagnostics/target_transform_bias/lightgbm_h1_hurdle_diagnostic.json",
            "outputs/diagnostics/target_transform_bias/lightgbm_h1_hurdle_target_scale_diagnostic.json",
            "outputs/diagnostics/target_transform_bias/lightgbm_h1_hurdle_tweedie_diagnostic.json",
        ],
    }]


@lru_cache(maxsize=None)
def load_narrative_notes() -> list[dict]:
    notes = []

    lstm_raw = _load("outputs/hpo/_diag_rawscale_lstm_h1/p13_lstm_h1_rawscale.json")
    if lstm_raw:
        trials = lstm_raw.get("all_trials") or []
        if trials:
            wapes = [float(t["pooled_wape"]) for t in trials]
            biases = [float(t["pooled_bias"]) for t in trials]
            notes.append({
                "id": "lstm_raw_scale",
                "title": "LSTM raw-scale 시도 (h1)",
                "body": (
                    f"LightGBM과 동일한 원리로 LSTM에도 raw-scale(log1p 미적용) 학습을 시도했으나 "
                    f"WAPE {min(wapes):.1f}~{max(wapes):.1f}%로 발산하고 Bias도 "
                    f"+{min(biases):.1f}~+{max(biases):.1f}%p로 급격히 악화되어, "
                    f"raw-scale은 전 모델 공통 해법으로 부적합하다고 판단해 제외함(모델·HP grid가 달라 "
                    f"LightGBM 비교표와는 별도로 기록)."
                ),
                "source_paths": ["outputs/hpo/_diag_rawscale_lstm_h1/p13_lstm_h1_rawscale.json"],
            })

    single_b = _load("outputs/hurdle_hpo/lgbm_hurdle/b_center/single_lightgbm_h1.json")
    hurdle_b = _load("outputs/hurdle_hpo/lgbm_hurdle/b_center/hurdle_lightgbm_h1.json")
    if single_b and hurdle_b:
        sp, hp = single_b["pooled_metrics"], hurdle_b["pooled_metrics"]
        notes.append({
            "id": "b_center_robustness",
            "title": "B센터 walk-forward 로버스트니스 체크 (재튜닝 없음, h1)",
            "body": (
                f"A센터로 고정한 하이퍼파라미터를 재튜닝 없이 B센터 2023 이후 walk-forward에 적용: "
                f"단일 LightGBM WAPE {sp['wape']:.2f}% / Bias {sp['bias']:.2f}%p → "
                f"Hurdle-LightGBM WAPE {hp['wape']:.2f}% / Bias {hp['bias']:.2f}%p. "
                f"WAPE는 소폭 악화되지만 Bias는 크게 개선됨 — 모델 재선정에는 쓰이지 않은 참고 체크."
            ),
            "source_paths": [
                "outputs/hurdle_hpo/lgbm_hurdle/b_center/single_lightgbm_h1.json",
                "outputs/hurdle_hpo/lgbm_hurdle/b_center/hurdle_lightgbm_h1.json",
            ],
        })

    seed = _load("outputs/hurdle_hpo/lgbm_hurdle/robustness/seed_stability.json")
    if seed:
        parts = []
        for h in ("1", "2", "4"):
            block = seed.get(h)
            if not block:
                continue
            seeds = block.get("seeds", {})
            passes = [abs(v["pooled"]["bias"]) <= 20 for v in seeds.values()]
            parts.append(f"h{h}: {sum(passes)}/{len(passes)} seed 통과")
        if parts:
            notes.append({
                "id": "seed_robustness",
                "title": "Hurdle-LightGBM seed 강건성 체크 (재선정 없음)",
                "body": (
                    "공식 선정 하이퍼파라미터를 seed 42/123/456으로 재학습(재탐색·재선정 없음): "
                    + ", ".join(parts)
                    + ". 일부 horizon은 seed에 따라 Bias 가드레일 통과 여부가 달라질 수 있음을 있는 그대로 기록."
                ),
                "source_paths": ["outputs/hurdle_hpo/lgbm_hurdle/robustness/seed_stability.json"],
            })

    return notes
