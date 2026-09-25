from typing import Dict, List, Literal, Optional

import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from services.stat_common_key_data import (
    MODEL_LABELS as STAT_COMMON_MODEL_LABELS,
    MODEL_ORDER as STAT_COMMON_MODEL_ORDER,
    all_family_row as stat_all_family_row,
    coverage_row as stat_coverage_row,
    is_extreme as stat_is_extreme,
    is_extreme_all_family as stat_is_extreme_all_family,
)
from services.model_comparison_data import (
    EXTREME_MODELS,
    HORIZONS,
    MODEL_LABELS,
    MODEL_ORDER,
    load_stat_common_metrics,
)
from services.ml_dl_selection_data import (
    BIAS_BAND as MLDL_BIAS_BAND,
    LABEL_TO_MODEL as MLDL_LABEL_TO_MODEL,
    MODEL_LABELS as MLDL_MODEL_LABELS,
    MODEL_ORDER as MLDL_MODEL_ORDER,
    load_parameter_summary,
    load_trials,
    representative_row,
)
from services.ml_dl_raw_metrics import get_mae_rmse
from services.ml_dl_experiments_data import load_experiment_entries, load_narrative_notes
from services.model_final_compare_data import (
    METRIC_COLUMNS as FINAL_METRIC_COLUMNS,
    load_cross_track as final_load_cross_track,
)
from services.qa_detail_data import (
    STAT_KEY_TO_MODEL_VARIANT as QA_STAT_KEY_TO_MODEL_VARIANT,
    load_stat_model_comparison as qa_load_stat_model_comparison,
)

router = APIRouter()

Center = Literal["A", "B", "ALL"]
Metric = Literal["WAPE", "Bias", "MAE", "RMSE"]
HorizonKey = Literal["h1", "h2", "h4"]

HORIZON_LABELS = {1: "h1", 2: "h2", 4: "h4"}
HORIZON_VALUES = {v: k for k, v in HORIZON_LABELS.items()}

class HeatmapRow(BaseModel):
    model:      str
    label:      str
    is_extreme: bool
    values:     Dict[str, float]

class ColorScale(BaseModel):
    min: float
    max: float

class HeatmapResponse(BaseModel):
    metric:      Metric
    center:      Center
    horizons:    List[str]
    rows:        List[HeatmapRow]
    color_scale: ColorScale

@router.get("/stat/heatmap", response_model=HeatmapResponse)
def get_stat_heatmap(center: Center = Query("ALL"), metric: Metric = Query("WAPE")):
    df = load_stat_common_metrics()
    sub = df[df["center"] == center]

    rows: List[HeatmapRow] = []
    stable_values: List[float] = []
    for m in MODEL_ORDER:
        m_rows = sub[sub["model"] == m]
        values: Dict[str, float] = {}
        for h in HORIZONS:
            hit = m_rows[m_rows["horizon"] == h]
            if hit.empty:
                continue
            v = float(hit.iloc[0][metric])
            values[HORIZON_LABELS[h]] = v
            if m not in EXTREME_MODELS:
                stable_values.append(v)
        rows.append(HeatmapRow(model=m, label=MODEL_LABELS[m], is_extreme=m in EXTREME_MODELS, values=values))

    if not stable_values:
        raise HTTPException(status_code=404, detail="데이터가 없습니다.")

    return HeatmapResponse(
        metric=metric, center=center, horizons=[HORIZON_LABELS[h] for h in HORIZONS],
        rows=rows, color_scale=ColorScale(min=min(stable_values), max=max(stable_values)),
    )

StatCommonMetric = Literal["WAPE", "Bias", "MAE", "RMSE"]


class CommonHeatmapRow(BaseModel):
    model: str
    label: str
    is_extreme: bool
    values: Dict[str, float]
    n_rows: Dict[str, int]

class CommonHeatmapResponse(BaseModel):
    metric: StatCommonMetric
    center: Center
    horizons: List[str]
    n_sku: Optional[int] = None
    rows: List[CommonHeatmapRow]

@router.get("/stat/common-heatmap", response_model=CommonHeatmapResponse)
def get_stat_common_heatmap(center: Center = Query("ALL"), metric: StatCommonMetric = Query("WAPE")):
    """카드2 Heatmap — 7개 모델이 전부 동시에 존재하는 단일 공통 panel(all-family common
    panel)에서 계산한 값만 사용한다(모델마다 다른 population 아님)."""
    metric_key = {"WAPE": "wape", "Bias": "bias", "MAE": "mae", "RMSE": "rmse"}[metric]
    rows: List[CommonHeatmapRow] = []
    n_sku = None
    for model in STAT_COMMON_MODEL_ORDER:
        values, nrows = {}, {}
        for h in HORIZONS:
            r = stat_all_family_row(model, center, h)
            if r is None:
                continue
            hk = HORIZON_LABELS[h]
            values[hk] = r[metric_key]
            nrows[hk] = r["n_rows"]
            n_sku = r["n_sku"]
        rows.append(CommonHeatmapRow(
            model=model, label=STAT_COMMON_MODEL_LABELS[model], is_extreme=stat_is_extreme_all_family(model),
            values=values, n_rows=nrows,
        ))
    return CommonHeatmapResponse(metric=metric, center=center, horizons=[HORIZON_LABELS[h] for h in HORIZONS], n_sku=n_sku, rows=rows)


class CommonBiasPoint(BaseModel):
    model: str
    label: str
    wape: float
    bias: float
    n_rows: int
    n_sku: int

class CommonBiasMapResponse(BaseModel):
    center: Center
    horizon: str
    points: List[CommonBiasPoint]

@router.get("/stat/common-wape-bias", response_model=CommonBiasMapResponse)
def get_stat_common_wape_bias(center: Center = Query("ALL"), horizon: HorizonKey = Query("h1")):
    """카드5 WAPE×Bias — 카드2와 정확히 동일한 all-family common panel 값만 사용한다.
    통계 트랙에는 Bias ±20% guardrail이나 P13 개념이 존재하지 않으므로(ML/DL 전용) 그런
    기준선은 표시하지 않는다."""
    points: List[CommonBiasPoint] = []
    for model in STAT_COMMON_MODEL_ORDER:
        if stat_is_extreme_all_family(model):
            continue
        r = stat_all_family_row(model, center, HORIZON_VALUES[horizon])
        if r is None:
            continue
        points.append(CommonBiasPoint(
            model=model, label=STAT_COMMON_MODEL_LABELS[model], wape=r["wape"], bias=r["bias"],
            n_rows=r["n_rows"], n_sku=r["n_sku"],
        ))
    return CommonBiasMapResponse(center=center, horizon=horizon, points=points)


class CoverageEntry(BaseModel):
    model: str
    label: str
    n_total: int
    n_normal: int
    n_fallback_constant: int
    n_fallback_naive_mean: int
    n_fallback_coldstart: int
    normal_rate: float
    fallback_rate: float
    n_has_observed_history_false: int
    n_override_instability: Optional[int] = None
    is_extreme: bool

class CoverageResponse(BaseModel):
    entries: List[CoverageEntry]

@router.get("/stat/coverage", response_model=CoverageResponse)
def get_stat_coverage():
    """카드4 — 단순 초대형 WAPE 값이 아니라 모델별 정상/실패/fallback/cold-start/override
    건수를 함께 보여준다(own 전체 population 기준, pair로 좁히지 않음)."""
    entries: List[CoverageEntry] = []
    for model in STAT_COMMON_MODEL_ORDER:
        r = stat_coverage_row(model)
        if r is None:
            continue
        entries.append(CoverageEntry(
            model=model, label=STAT_COMMON_MODEL_LABELS[model],
            n_total=int(r["n_total"]), n_normal=int(r["n_normal"]),
            n_fallback_constant=int(r["n_fallback_constant"]), n_fallback_naive_mean=int(r["n_fallback_naive_mean"]),
            n_fallback_coldstart=int(r["n_fallback_coldstart"]), normal_rate=float(r["normal_rate"]),
            fallback_rate=float(r["fallback_rate"]), n_has_observed_history_false=int(r["n_has_observed_history_false"]),
            n_override_instability=int(r["n_override_instability"]) if r.get("n_override_instability") is not None else None,
            is_extreme=stat_is_extreme(model),
        ))
    return CoverageResponse(entries=entries)


MlDlModel = Literal["RF", "LGBM", "LSTM", "TFT", "Informer", "H-RF", "H-LGBM"]

class ParamSummaryRow(BaseModel):
    parameter_type:      str
    parameter:            str
    candidates_or_rule:   str
    selected_or_fixed:    str
    selection_stage:      str
    selection_criterion:  str

class TrialInfo(BaseModel):
    horizon:     HorizonKey
    trial:       int
    wape:        float
    bias:        float
    bias_pass:   bool
    is_selected: bool

class ModelDetailResponse(BaseModel):
    model:   str
    label:   str
    track:   str
    summary: List[ParamSummaryRow]
    trial:   Optional[TrialInfo]

@router.get("/mldl/model-detail", response_model=ModelDetailResponse)
def get_mldl_model_detail(model: MlDlModel = Query("H-LGBM"), horizon: HorizonKey = Query("h1")):
    csv_model = MLDL_LABEL_TO_MODEL[model]

    params = load_parameter_summary()
    rows = params[params["model"] == csv_model]
    if rows.empty:
        raise HTTPException(status_code=404, detail="존재하지 않는 모델입니다.")

    summary = [
        ParamSummaryRow(
            parameter_type=r["parameter_type"], parameter=r["parameter"],
            candidates_or_rule=r["candidates_or_rule"], selected_or_fixed=r["selected_or_fixed"],
            selection_stage=r["selection_stage"], selection_criterion=r["selection_criterion"],
        )
        for _, r in rows.iterrows()
    ]

    trials = load_trials()
    m_rows = trials[(trials["model"] == csv_model) & (trials["horizon"] == HORIZON_VALUES[horizon])]
    trial_info: Optional[TrialInfo] = None
    if not m_rows.empty:
        sel_rows = m_rows[m_rows["selected"]]
        if not sel_rows.empty:
            r = sel_rows.iloc[0]
            trial_info = TrialInfo(horizon=horizon, trial=int(r["trial"]), wape=float(r["pooled_wape"]), bias=float(r["pooled_bias"]), bias_pass=bool(r["bias_pass"]), is_selected=True)
        else:
            r = m_rows.loc[m_rows["pooled_wape"].idxmin()]
            trial_info = TrialInfo(horizon=horizon, trial=int(r["trial"]), wape=float(r["pooled_wape"]), bias=float(r["pooled_bias"]), bias_pass=bool(r["bias_pass"]), is_selected=False)

    return ModelDetailResponse(model=csv_model, label=model, track=str(rows.iloc[0]["track"]), summary=summary, trial=trial_info)

MLDL_MODEL_TRACK: Dict[str, str] = {
    "RF": "ml", "LightGBM": "ml",
    "LSTM": "dl", "TFT": "dl", "Informer": "dl",
    "Hurdle-RF": "hurdle", "Hurdle-LightGBM": "hurdle",
}

class ModelSummaryEntry(BaseModel):
    model:       str
    label:       str
    track:       Literal["ml", "dl", "hurdle"]
    horizon:     HorizonKey
    trial:       int
    wape:        float
    bias:        float
    mae:         Optional[float]
    rmse:        Optional[float]
    mae_status:  Literal["ok", "unavailable"]
    rmse_status: Literal["ok", "unavailable"]
    bias_pass:   bool
    status:      Literal["selected", "reference_only"]

class ModelSummaryResponse(BaseModel):
    bias_band: float
    entries:   List[ModelSummaryEntry]

@router.get("/mldl/model-summary", response_model=ModelSummaryResponse)
def get_mldl_model_summary():
    """7개 모델 × h1/h2/h4의 2023 P13 대표 행. status='selected'는 실제 bias guardrail
    통과 selected trial, status='reference_only'는 통과 trial이 없어 참고용으로만 쓰는
    최저 Pooled WAPE trial이다(둘을 절대 같은 의미로 표시하지 않는다)."""
    df = load_trials()
    entries: List[ModelSummaryEntry] = []
    for m in MLDL_MODEL_ORDER:
        for h in HORIZONS:
            rows = df[(df["model"] == m) & (df["horizon"] == h)]
            if rows.empty:
                continue
            row, status = representative_row(df, m, h)
            trial_id = int(row["trial"])
            wape, bias = float(row["pooled_wape"]), float(row["pooled_bias"])
            mae, rmse, mae_status = get_mae_rmse(m, h, trial_id, wape)
            entries.append(ModelSummaryEntry(
                model=m, label=MLDL_MODEL_LABELS[m], track=MLDL_MODEL_TRACK[m],
                horizon=HORIZON_LABELS[h], trial=trial_id, wape=wape, bias=bias,
                mae=mae, rmse=rmse, mae_status=mae_status, rmse_status=mae_status,
                bias_pass=bool(row["bias_pass"]), status=status,
            ))
    return ModelSummaryResponse(bias_band=MLDL_BIAS_BAND, entries=entries)

class ExperimentVariant(BaseModel):
    key:     str
    label:   str
    wape:    float
    bias:    float
    adopted: bool

class ExperimentEntry(BaseModel):
    id:         str
    title:      str
    hypothesis: str
    changed:    str
    eval_note:  str
    variants:   List[ExperimentVariant]
    conclusion: str
    source_paths: List[str]

class NarrativeNote(BaseModel):
    id:   str
    title: str
    body: str
    source_paths: List[str]

class ExperimentsResponse(BaseModel):
    entries: List[ExperimentEntry]
    narrative_notes: List[NarrativeNote]

@router.get("/mldl/improvement-experiments", response_model=ExperimentsResponse)
def get_mldl_improvement_experiments():
    """Hurdle 구조 도입 전후에 실제로 수행된 진단 실험(outputs/hpo, outputs/diagnostics 원본 JSON
    직접 로드). 동일 평가조건인 실험끼리만 entries(수치 비교), 조건이 다른 실험은 narrative_notes(서술)."""
    return ExperimentsResponse(
        entries=[ExperimentEntry(**e) for e in load_experiment_entries()],
        narrative_notes=[NarrativeNote(**n) for n in load_narrative_notes()],
    )

FinalCenter = Literal["ALL", "A", "B"]
FinalHorizon = Literal["ALL", "h1", "h2", "h4"]
FinalScope = Literal["full_common", "model_fit", "fallback"]
FinalMetric = Literal["WAPE", "Bias", "MAE", "RMSE"]
FinalQuartile = Literal["Q1", "Q2", "Q3", "Q4"]

def _final_horizons(horizon: FinalHorizon) -> List[int]:
    return [1, 2, 4] if horizon == "ALL" else [HORIZON_VALUES[horizon]]

class FinalKpiEntry(BaseModel):
    horizon:    str
    stat_value: float
    ml_value:   float
    diff:       float

class FinalKpiResponse(BaseModel):
    metric:  FinalMetric
    entries: List[FinalKpiEntry]

@router.get("/final/kpi", response_model=FinalKpiResponse)
def get_final_kpi(
    center:  FinalCenter = Query("ALL"),
    horizon: FinalHorizon = Query("ALL"),
    metric:  FinalMetric  = Query("WAPE"),
    scope:   FinalScope   = Query("full_common"),
):
    stat_col, ml_col = FINAL_METRIC_COLUMNS[metric]
    df = final_load_cross_track()
    sub = df[(df["center"] == center) & (df["comparison_scope"] == scope)]

    entries: List[FinalKpiEntry] = []
    for h in _final_horizons(horizon):
        hit = sub[sub["horizon"] == h]
        if hit.empty:
            continue
        row = hit.iloc[0]
        sv, mv = float(row[stat_col]), float(row[ml_col])
        entries.append(FinalKpiEntry(horizon=HORIZON_LABELS[h], stat_value=sv, ml_value=mv, diff=sv - mv))

    return FinalKpiResponse(metric=metric, entries=entries)

class QaStatVariableEntry(BaseModel):
    key:        str
    label:      str
    is_extreme: bool
    wape:       float
    bias:       float
    mae:        float
    rmse:       float

class QaStatVariableResponse(BaseModel):
    center:  Center
    horizon: HorizonKey
    entries: List[QaStatVariableEntry]

@router.get("/qa/stat-variable-effect", response_model=QaStatVariableResponse)
def get_qa_stat_variable_effect(center: Center = Query("ALL"), horizon: HorizonKey = Query("h1")):
    df = qa_load_stat_model_comparison()
    sub = df[(df["center"] == center) & (df["horizon"] == HORIZON_VALUES[horizon])]

    entries: List[QaStatVariableEntry] = []
    for key in MODEL_ORDER:
        m, v = QA_STAT_KEY_TO_MODEL_VARIANT[key]
        hit = sub[(sub["model"] == m) & (sub["variant"] == v)]
        if hit.empty:
            continue
        row = hit.iloc[0]
        entries.append(QaStatVariableEntry(
            key=key, label=MODEL_LABELS[key], is_extreme=key in EXTREME_MODELS,
            wape=float(row["WAPE"]), bias=float(row["Bias"]), mae=float(row["MAE"]), rmse=float(row["RMSE"]),
        ))
    return QaStatVariableResponse(center=center, horizon=horizon, entries=entries)
