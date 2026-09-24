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
    MODEL_DETAILS,
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
    ML_LABEL as FINAL_ML_LABEL,
    ML_MODEL as FINAL_ML_MODEL,
    METRIC_COLUMNS as FINAL_METRIC_COLUMNS,
    QUARTILE_LABELS as FINAL_QUARTILE_LABELS,
    QUARTILE_ORDER as FINAL_QUARTILE_ORDER,
    STAT_LABEL as FINAL_STAT_LABEL,
    STAT_MODEL_SKU as FINAL_STAT_MODEL_SKU,
    assign_quartile as final_assign_quartile,
    filter_sku_compare as final_filter_sku_compare,
    load_cross_track as final_load_cross_track,
)
from services.qa_detail_data import (
    COVERAGE_SCOPE_LABELS as QA_SCOPE_LABELS,
    ML_MODEL as QA_ML_MODEL,
    ML_VARIANT as QA_ML_VARIANT,
    STAT_KEY_TO_MODEL_VARIANT as QA_STAT_KEY_TO_MODEL_VARIANT,
    STAT_MODEL as QA_STAT_MODEL,
    STAT_VARIANT as QA_STAT_VARIANT,
    TS_MODELS as QA_TS_MODELS,
    TS_MODEL_LABELS as QA_TS_MODEL_LABELS,
    compute_sku_summary as qa_compute_sku_summary,
    get_product_info as qa_get_product_info,
    load_coverage_summary as qa_load_coverage_summary,
    load_sku_product_lookup as qa_load_sku_product_lookup,
    load_stat_model_comparison as qa_load_stat_model_comparison,
    load_weekly_error as qa_load_weekly_error,
    query_sku_predictions as qa_query_sku_predictions,
    query_sku_predictions_multi as qa_query_sku_predictions_multi,
    representative_skus as qa_representative_skus,
    search_products as qa_search_products,
    weighted_weekly_combine as qa_weighted_weekly_combine,
)

router = APIRouter()

Center = Literal["A", "B", "ALL"]
Metric = Literal["WAPE", "Bias", "MAE", "RMSE"]
HorizonKey = Literal["h1", "h2", "h4"]

HORIZON_LABELS = {1: "h1", 2: "h2", 4: "h4"}
HORIZON_VALUES = {v: k for k, v in HORIZON_LABELS.items()}

class ModelInfo(BaseModel):
    model: str
    label: str
    is_extreme: bool
    detail: Dict[str, str]

@router.get("/stat/models", response_model=List[ModelInfo])
def get_stat_models():
    return [
        ModelInfo(model=m, label=MODEL_LABELS[m], is_extreme=m in EXTREME_MODELS, detail=MODEL_DETAILS[m])
        for m in MODEL_ORDER
    ]

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

class ProfileSeries(BaseModel):
    model:      str
    label:      str
    is_extreme: bool
    values:     List[Optional[float]]

class ProfileResponse(BaseModel):
    metric:   Metric
    center:   Center
    horizons: List[str]
    series:   List[ProfileSeries]

@router.get("/stat/horizon-profile", response_model=ProfileResponse)
def get_stat_horizon_profile(
    center:          Center = Query("ALL"),
    metric:          Metric = Query("WAPE"),
    include_extreme: bool   = Query(False),
):
    df = load_stat_common_metrics()
    sub = df[df["center"] == center]

    models = MODEL_ORDER if include_extreme else [m for m in MODEL_ORDER if m not in EXTREME_MODELS]
    series: List[ProfileSeries] = []
    for m in models:
        m_rows = sub[sub["model"] == m]
        values: List[Optional[float]] = []
        for h in HORIZONS:
            hit = m_rows[m_rows["horizon"] == h]
            values.append(float(hit.iloc[0][metric]) if not hit.empty else None)
        series.append(ProfileSeries(model=m, label=MODEL_LABELS[m], is_extreme=m in EXTREME_MODELS, values=values))

    return ProfileResponse(
        metric=metric, center=center, horizons=[HORIZON_LABELS[h] for h in HORIZONS], series=series,
    )

class BiasPoint(BaseModel):
    model:      str
    label:      str
    is_extreme: bool
    wape:       float
    bias:       float

class BiasMapResponse(BaseModel):
    center:  Center
    horizon: str
    points:  List[BiasPoint]

@router.get("/stat/wape-bias", response_model=BiasMapResponse)
def get_stat_wape_bias(
    center:          Center     = Query("ALL"),
    horizon:         HorizonKey = Query("h1"),
    include_extreme: bool       = Query(False),
):
    df = load_stat_common_metrics()
    sub = df[(df["center"] == center) & (df["horizon"] == HORIZON_VALUES[horizon])]

    models = MODEL_ORDER if include_extreme else [m for m in MODEL_ORDER if m not in EXTREME_MODELS]
    points: List[BiasPoint] = []
    for m in models:
        hit = sub[sub["model"] == m]
        if hit.empty:
            continue
        row = hit.iloc[0]
        points.append(BiasPoint(
            model=m, label=MODEL_LABELS[m], is_extreme=m in EXTREME_MODELS,
            wape=float(row["WAPE"]), bias=float(row["Bias"]),
        ))

    return BiasMapResponse(center=center, horizon=horizon, points=points)

class CenterSeries(BaseModel):
    center: str
    values: List[float]

class CenterCompareResponse(BaseModel):
    model:    str
    label:    str
    horizons: List[str]
    series:   List[CenterSeries]

@router.get("/stat/center-compare", response_model=CenterCompareResponse)
def get_stat_center_compare(model: str = Query("SARIMA")):
    if model not in MODEL_ORDER:
        raise HTTPException(status_code=404, detail="존재하지 않는 모델입니다.")

    df = load_stat_common_metrics()
    sub = df[df["model"] == model]

    series: List[CenterSeries] = []
    for c in ["A", "B", "ALL"]:
        c_rows = sub[sub["center"] == c]
        values = []
        for h in HORIZONS:
            hit = c_rows[c_rows["horizon"] == h]
            values.append(float(hit.iloc[0]["WAPE"]) if not hit.empty else 0.0)
        series.append(CenterSeries(center=c, values=values))

    return CenterCompareResponse(
        model=model, label=MODEL_LABELS[model], horizons=[HORIZON_LABELS[h] for h in HORIZONS], series=series,
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


MlDlModelFilter = Literal["ALL", "RF", "LGBM", "LSTM", "TFT", "Informer", "H-RF", "H-LGBM"]
MlDlModel = Literal["RF", "LGBM", "LSTM", "TFT", "Informer", "H-RF", "H-LGBM"]
TrialFilter = Literal["ALL", "PASS", "SELECTED"]

class TrialPoint(BaseModel):
    model:      str
    label:      str
    trial:      int
    wape:       float
    bias:       float
    bias_pass:  bool
    selected:   bool
    hp:         Dict[str, str]

class TrialScatterResponse(BaseModel):
    horizon:      HorizonKey
    model:        MlDlModelFilter
    trial_filter: TrialFilter
    bias_band:    float
    points:       List[TrialPoint]

@router.get("/mldl/trials", response_model=TrialScatterResponse)
def get_mldl_trials(
    horizon:      HorizonKey        = Query("h1"),
    model:        MlDlModelFilter   = Query("ALL"),
    trial_filter: TrialFilter       = Query("ALL"),
):
    df = load_trials()
    sub = df[df["horizon"] == HORIZON_VALUES[horizon]]
    if model != "ALL":
        sub = sub[sub["model"] == MLDL_LABEL_TO_MODEL[model]]
    if trial_filter == "PASS":
        sub = sub[sub["bias_pass"]]
    elif trial_filter == "SELECTED":
        sub = sub[sub["selected"]]

    points = [
        TrialPoint(
            model=row["model"], label=MLDL_MODEL_LABELS[row["model"]], trial=int(row["trial"]),
            wape=float(row["pooled_wape"]), bias=float(row["pooled_bias"]),
            bias_pass=bool(row["bias_pass"]), selected=bool(row["selected"]),
            hp={k: str(v) for k, v in row["hp"].items()},
        )
        for _, row in sub.iterrows()
    ]
    return TrialScatterResponse(horizon=horizon, model=model, trial_filter=trial_filter, bias_band=MLDL_BIAS_BAND, points=points)

class BoxplotEntry(BaseModel):
    model: str
    label: str
    box:   List[float]  

class BoxplotResponse(BaseModel):
    horizon:   HorizonKey
    bias_band: float
    entries:   List[BoxplotEntry]

@router.get("/mldl/bias-boxplot", response_model=BoxplotResponse)
def get_mldl_bias_boxplot(horizon: HorizonKey = Query("h1")):
    df = load_trials()
    sub = df[df["horizon"] == HORIZON_VALUES[horizon]]

    entries: List[BoxplotEntry] = []
    for m in MLDL_MODEL_ORDER:
        vals = pd.Series(sub.loc[sub["model"] == m, "pooled_bias"].to_numpy())
        if vals.empty:
            continue
        q1, med, q3 = float(vals.quantile(0.25)), float(vals.quantile(0.5)), float(vals.quantile(0.75))
        entries.append(BoxplotEntry(model=m, label=MLDL_MODEL_LABELS[m], box=[float(vals.min()), q1, med, q3, float(vals.max())]))

    return BoxplotResponse(horizon=horizon, bias_band=MLDL_BIAS_BAND, entries=entries)

class PassRateEntry(BaseModel):
    model:       str
    label:       str
    pass_count:  int
    total_count: int
    pass_rate:   float

class PassRateResponse(BaseModel):
    horizon: HorizonKey
    entries: List[PassRateEntry]

@router.get("/mldl/pass-rate", response_model=PassRateResponse)
def get_mldl_pass_rate(horizon: HorizonKey = Query("h1")):
    df = load_trials()
    sub = df[df["horizon"] == HORIZON_VALUES[horizon]]

    entries: List[PassRateEntry] = []
    for m in MLDL_MODEL_ORDER:
        m_rows = sub[sub["model"] == m]
        total = len(m_rows)
        if total == 0:
            continue
        passed = int(m_rows["bias_pass"].sum())
        entries.append(PassRateEntry(model=m, label=MLDL_MODEL_LABELS[m], pass_count=passed, total_count=total, pass_rate=passed / total * 100))

    return PassRateResponse(horizon=horizon, entries=entries)

class ImprovementPoint(BaseModel):
    horizon: str
    trial:   int
    wape:    float
    bias:    float

class ImprovementSeries(BaseModel):
    model:  str
    label:  str
    points: List[ImprovementPoint]

class ImprovementResponse(BaseModel):
    horizons: List[str]
    old:      ImprovementSeries
    new:      ImprovementSeries

@router.get("/mldl/improvement", response_model=ImprovementResponse)
def get_mldl_improvement():
    """LightGBM(WAPE 최저 trial, bias guardrail 통과 trial이 없어 참고용) → Hurdle-LightGBM(selected trial)."""
    df = load_trials()

    old_points: List[ImprovementPoint] = []
    new_points: List[ImprovementPoint] = []
    for h in HORIZONS:
        label = HORIZON_LABELS[h]

        lgbm_rows = df[(df["model"] == "LightGBM") & (df["horizon"] == h)]
        if not lgbm_rows.empty:
            best = lgbm_rows.loc[lgbm_rows["pooled_wape"].idxmin()]
            old_points.append(ImprovementPoint(horizon=label, trial=int(best["trial"]), wape=float(best["pooled_wape"]), bias=float(best["pooled_bias"])))

        hlgbm_rows = df[(df["model"] == "Hurdle-LightGBM") & (df["horizon"] == h) & (df["selected"])]
        if not hlgbm_rows.empty:
            sel = hlgbm_rows.iloc[0]
            new_points.append(ImprovementPoint(horizon=label, trial=int(sel["trial"]), wape=float(sel["pooled_wape"]), bias=float(sel["pooled_bias"])))

    return ImprovementResponse(
        horizons=[HORIZON_LABELS[h] for h in HORIZONS],
        old=ImprovementSeries(model="LightGBM", label="LGBM", points=old_points),
        new=ImprovementSeries(model="Hurdle-LightGBM", label="H-LGBM", points=new_points),
    )

class HurdleCompareResponse(BaseModel):
    horizons: List[str]
    series:   List[ImprovementSeries]

@router.get("/mldl/hurdle-compare", response_model=HurdleCompareResponse)
def get_mldl_hurdle_compare():
    """Bias guardrail을 통과한 두 Hurdle 후보(H-RF/H-LGBM)의 selected trial을 horizon별로 비교."""
    df = load_trials()

    series: List[ImprovementSeries] = []
    for m in ["Hurdle-RF", "Hurdle-LightGBM"]:
        points: List[ImprovementPoint] = []
        for h in HORIZONS:
            rows = df[(df["model"] == m) & (df["horizon"] == h) & (df["selected"])]
            if rows.empty:
                continue
            sel = rows.iloc[0]
            points.append(ImprovementPoint(horizon=HORIZON_LABELS[h], trial=int(sel["trial"]), wape=float(sel["pooled_wape"]), bias=float(sel["pooled_bias"])))
        series.append(ImprovementSeries(model=m, label=MLDL_MODEL_LABELS[m], points=points))

    return HurdleCompareResponse(horizons=[HORIZON_LABELS[h] for h in HORIZONS], series=series)

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

class FinalBarPoint(BaseModel):
    horizon:    str
    stat_value: float
    ml_value:   float

class FinalBarResponse(BaseModel):
    metric: FinalMetric
    scope:  FinalScope
    center: FinalCenter
    points: List[FinalBarPoint]

@router.get("/final/bar", response_model=FinalBarResponse)
def get_final_bar(
    center:  FinalCenter = Query("ALL"),
    horizon: FinalHorizon = Query("ALL"),
    metric:  FinalMetric  = Query("WAPE"),
    scope:   FinalScope   = Query("full_common"),
):
    stat_col, ml_col = FINAL_METRIC_COLUMNS[metric]
    df = final_load_cross_track()
    sub = df[(df["center"] == center) & (df["comparison_scope"] == scope)]

    points: List[FinalBarPoint] = []
    for h in _final_horizons(horizon):
        hit = sub[sub["horizon"] == h]
        if hit.empty:
            continue
        row = hit.iloc[0]
        points.append(FinalBarPoint(horizon=HORIZON_LABELS[h], stat_value=float(row[stat_col]), ml_value=float(row[ml_col])))

    return FinalBarResponse(metric=metric, scope=scope, center=center, points=points)

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

class FinalHeatmapCell(BaseModel):
    center:     str
    horizon:    str
    stat_value: float
    ml_value:   float
    diff:       float

class FinalHeatmapResponse(BaseModel):
    metric:      FinalMetric
    scope:       FinalScope
    centers:     List[str]
    horizons:    List[str]
    cells:       List[FinalHeatmapCell]
    color_scale: ColorScale

@router.get("/final/heatmap", response_model=FinalHeatmapResponse)
def get_final_heatmap(metric: FinalMetric = Query("WAPE"), scope: FinalScope = Query("full_common")):
    stat_col, ml_col = FINAL_METRIC_COLUMNS[metric]
    df = final_load_cross_track()
    sub = df[df["comparison_scope"] == scope]

    cells: List[FinalHeatmapCell] = []
    diffs: List[float] = []
    for c in ["A", "B", "ALL"]:
        for h in [1, 2, 4]:
            hit = sub[(sub["center"] == c) & (sub["horizon"] == h)]
            if hit.empty:
                continue
            row = hit.iloc[0]
            sv, mv = float(row[stat_col]), float(row[ml_col])
            d = sv - mv
            diffs.append(d)
            cells.append(FinalHeatmapCell(center=c, horizon=HORIZON_LABELS[h], stat_value=sv, ml_value=mv, diff=d))

    if not diffs:
        raise HTTPException(status_code=404, detail="데이터가 없습니다.")

    return FinalHeatmapResponse(
        metric=metric, scope=scope, centers=["A", "B", "ALL"], horizons=[HORIZON_LABELS[h] for h in [1, 2, 4]],
        cells=cells, color_scale=ColorScale(min=min(diffs), max=max(diffs)),
    )

class FinalWinnerShareEntry(BaseModel):
    quartile:     str
    label:        str
    sarima_share: float
    hlgbm_share:  float
    n_skus:       int
    demand_min:   float
    demand_max:   float

class FinalWinnerShareResponse(BaseModel):
    center:  FinalCenter
    horizon: FinalHorizon
    entries: List[FinalWinnerShareEntry]

@router.get("/final/winner-share", response_model=FinalWinnerShareResponse)
def get_final_winner_share(center: FinalCenter = Query("ALL"), horizon: FinalHorizon = Query("ALL")):
    df = final_filter_sku_compare(center, horizon)
    df = df.assign(demand_quartile=final_assign_quartile(df["actual_sum"]))

    entries: List[FinalWinnerShareEntry] = []
    for q in FINAL_QUARTILE_ORDER:
        sub = df[df["demand_quartile"] == q]
        n = len(sub)
        if n == 0:
            continue
        counts = sub["winner"].value_counts()
        sarima_n, hlgbm_n = int(counts.get(FINAL_STAT_MODEL_SKU, 0)), int(counts.get(FINAL_ML_MODEL, 0))
        entries.append(FinalWinnerShareEntry(
            quartile=q, label=FINAL_QUARTILE_LABELS[q],
            sarima_share=sarima_n / n * 100, hlgbm_share=hlgbm_n / n * 100,
            n_skus=n, demand_min=float(sub["actual_sum"].min()), demand_max=float(sub["actual_sum"].max()),
        ))

    return FinalWinnerShareResponse(center=center, horizon=horizon, entries=entries)

class FinalWapeProfileEntry(BaseModel):
    quartile:  str
    label:     str
    stat_wape: float
    ml_wape:   float
    n_skus:    int

class FinalWapeProfileResponse(BaseModel):
    center:  FinalCenter
    horizon: FinalHorizon
    entries: List[FinalWapeProfileEntry]

@router.get("/final/wape-profile", response_model=FinalWapeProfileResponse)
def get_final_wape_profile(center: FinalCenter = Query("ALL"), horizon: FinalHorizon = Query("ALL")):
    df = final_filter_sku_compare(center, horizon)
    df = df.assign(demand_quartile=final_assign_quartile(df["actual_sum"]))

    entries: List[FinalWapeProfileEntry] = []
    for q in FINAL_QUARTILE_ORDER:
        sub = df[df["demand_quartile"] == q]
        n = len(sub)
        if n == 0:
            continue
        w = sub["actual_sum"]
        stat_wape = float((sub["stat_WAPE"] * w).sum() / w.sum())
        ml_wape = float((sub["ml_WAPE"] * w).sum() / w.sum())
        entries.append(FinalWapeProfileEntry(quartile=q, label=FINAL_QUARTILE_LABELS[q], stat_wape=stat_wape, ml_wape=ml_wape, n_skus=n))

    return FinalWapeProfileResponse(center=center, horizon=horizon, entries=entries)

class FinalShareBasis(BaseModel):
    sarima_pct: float
    hlgbm_pct:  float

class FinalDemandShareResponse(BaseModel):
    center:       FinalCenter
    horizon:      FinalHorizon
    n_skus:       int
    total_demand: float
    sku_basis:    FinalShareBasis
    demand_basis: FinalShareBasis

@router.get("/final/demand-share", response_model=FinalDemandShareResponse)
def get_final_demand_share(center: FinalCenter = Query("ALL"), horizon: FinalHorizon = Query("ALL")):
    df = final_filter_sku_compare(center, horizon)
    n = len(df)
    if n == 0:
        raise HTTPException(status_code=404, detail="데이터가 없습니다.")

    counts = df["winner"].value_counts()
    sarima_n, hlgbm_n = int(counts.get(FINAL_STAT_MODEL_SKU, 0)), int(counts.get(FINAL_ML_MODEL, 0))
    demand = df.groupby("winner")["actual_sum"].sum()
    total_demand = float(df["actual_sum"].sum())
    sarima_d, hlgbm_d = float(demand.get(FINAL_STAT_MODEL_SKU, 0.0)), float(demand.get(FINAL_ML_MODEL, 0.0))

    return FinalDemandShareResponse(
        center=center, horizon=horizon, n_skus=n, total_demand=total_demand,
        sku_basis=FinalShareBasis(sarima_pct=sarima_n / n * 100, hlgbm_pct=hlgbm_n / n * 100),
        demand_basis=FinalShareBasis(sarima_pct=sarima_d / total_demand * 100, hlgbm_pct=hlgbm_d / total_demand * 100),
    )

class FinalDemandDetailResponse(BaseModel):
    quartile:       FinalQuartile
    label:          str
    center:         FinalCenter
    horizon:        FinalHorizon
    n_skus:         int
    demand_min:     float
    demand_max:     float
    winner:         str
    winner_label:   str
    sku_share:      FinalShareBasis
    demand_share:   FinalShareBasis
    interpretation: str

@router.get("/final/demand-detail", response_model=FinalDemandDetailResponse)
def get_final_demand_detail(
    center:   FinalCenter  = Query("ALL"),
    horizon:  FinalHorizon = Query("ALL"),
    quartile: FinalQuartile = Query("Q4"),
):
    df = final_filter_sku_compare(center, horizon)
    df = df.assign(demand_quartile=final_assign_quartile(df["actual_sum"]))
    sub = df[df["demand_quartile"] == quartile]
    n = len(sub)
    if n == 0:
        raise HTTPException(status_code=404, detail="데이터가 없습니다.")

    counts = sub["winner"].value_counts()
    sarima_n, hlgbm_n = int(counts.get(FINAL_STAT_MODEL_SKU, 0)), int(counts.get(FINAL_ML_MODEL, 0))
    demand = sub.groupby("winner")["actual_sum"].sum()
    total_demand = float(sub["actual_sum"].sum())
    sarima_d, hlgbm_d = float(demand.get(FINAL_STAT_MODEL_SKU, 0.0)), float(demand.get(FINAL_ML_MODEL, 0.0))

    sku_share = FinalShareBasis(sarima_pct=sarima_n / n * 100, hlgbm_pct=hlgbm_n / n * 100)
    demand_share = FinalShareBasis(sarima_pct=sarima_d / total_demand * 100, hlgbm_pct=hlgbm_d / total_demand * 100)

    is_hlgbm_winner = hlgbm_n >= sarima_n
    winner = FINAL_ML_MODEL if is_hlgbm_winner else FINAL_STAT_MODEL_SKU
    winner_label = FINAL_ML_LABEL if is_hlgbm_winner else FINAL_STAT_LABEL
    winner_sku_pct = sku_share.hlgbm_pct if is_hlgbm_winner else sku_share.sarima_pct
    winner_demand_pct = demand_share.hlgbm_pct if is_hlgbm_winner else demand_share.sarima_pct
    interpretation = f"{FINAL_QUARTILE_LABELS[quartile]}에서 {winner_label} 우세 (SKU 기준 {winner_sku_pct:.0f}%, 실제 수요량 기준 {winner_demand_pct:.0f}%)"

    return FinalDemandDetailResponse(
        quartile=quartile, label=FINAL_QUARTILE_LABELS[quartile], center=center, horizon=horizon,
        n_skus=n, demand_min=float(sub["actual_sum"].min()), demand_max=float(sub["actual_sum"].max()),
        winner=winner, winner_label=winner_label, sku_share=sku_share, demand_share=demand_share,
        interpretation=interpretation,
    )

QaSkuCenter = Literal["A", "B"]

class QaProductSearchItem(BaseModel):
    sku_id:            str
    center:            str
    product_name:      Optional[str] = None
    barcode:           Optional[str] = None
    option_code:       Optional[str] = None
    centers_available: List[str]

class QaProductSearchResponse(BaseModel):
    query: str
    items: List[QaProductSearchItem]

@router.get("/qa/products/search", response_model=QaProductSearchResponse)
def get_qa_products_search(q: str = Query(..., min_length=1), limit: int = Query(8, ge=1, le=30)):
    df = qa_search_products(q, limit)
    items = [
        QaProductSearchItem(
            sku_id=r.sku_id, center=r.center_id,
            product_name=r.상품명 if pd.notna(r.상품명) else None,
            barcode=str(r.barcode) if pd.notna(r.barcode) else None,
            option_code=r.option_code if pd.notna(r.option_code) else None,
            centers_available=r.centers_available,
        )
        for r in df.itertuples()
    ]
    return QaProductSearchResponse(query=q, items=items)

class QaProductInfoResponse(BaseModel):
    sku_id:            str
    center:            str
    product_name:      Optional[str]
    barcode:           Optional[str]
    option_code:       Optional[str]
    category_large:    Optional[str]
    category_middle:   Optional[str]
    category_small:    Optional[str]
    centers_available: List[str]

@router.get("/qa/product-info", response_model=QaProductInfoResponse)
def get_qa_product_info(sku_id: str = Query(...), center: QaSkuCenter = Query(...)):
    info = qa_get_product_info(sku_id, center)
    if not info:
        raise HTTPException(status_code=404, detail="해당 SKU/센터 조합의 상품 정보를 찾을 수 없습니다.")
    return QaProductInfoResponse(**info)

QaWeeksWindow = Literal["12", "24", "52", "all"]

class QaTsWeekPoint(BaseModel):
    target_date: str
    actual:      float
    predictions: Dict[str, Optional[float]]

class QaTsModelSummary(BaseModel):
    model:              str
    label:              str
    is_default:         bool
    n_weeks_used:       int
    wape:               Optional[float]
    bias:               Optional[float]
    mae:                Optional[float]
    rmse:               Optional[float]
    actual_total:       float
    prediction_total:   float
    fallback_ratio:     float
    cold_start_ratio:   float
    model_fit_ratio:    float
    development_n_obs:  Optional[float]
    model_applied_rows:  int
    comparison_scope_counts: Dict[str, int]
    forecast_source_counts: Dict[str, int]

class QaTsSummary(BaseModel):
    weeks_available:    int
    weeks_used:         int
    actual_consistent:  bool
    inconsistent_dates: List[str]
    avg_actual:         Optional[float]
    zero_demand_ratio:  Optional[float]
    zero_demand_count:  Optional[int]
    actual_total:       Optional[float]
    winner_model:       Optional[str]
    models:             List[QaTsModelSummary]
    diagnostics:        Dict[str, object]

class QaSkuTimeseriesResponse(BaseModel):
    sku_id:          str
    center:          QaSkuCenter
    horizon:         HorizonKey
    weeks_window:    QaWeeksWindow
    available_models: List[str]
    weeks:           List[QaTsWeekPoint]
    summary:         QaTsSummary
    evaluation_design: str
    evaluation_start: Optional[str]
    evaluation_end: Optional[str]

@router.get("/qa/sku-timeseries", response_model=QaSkuTimeseriesResponse)
def get_qa_sku_timeseries(
    sku_id:  str        = Query(...),
    center:  QaSkuCenter = Query(...),
    horizon: HorizonKey  = Query("h1"),
    weeks:   QaWeeksWindow = Query("24"),
):
    models = [(m, v) for m, v, _label, _default in QA_TS_MODELS]
    df = qa_query_sku_predictions_multi(sku_id, center, HORIZON_VALUES[horizon], models)
    if df.empty:
        raise HTTPException(status_code=404, detail="해당 SKU/센터/예측시점에 저장된 SKU-level 예측 데이터를 찾을 수 없습니다.")

    summary = qa_compute_sku_summary(df, weeks)
    window_dates = summary["window_dates"]
    available_models = sorted({m["model"] for m in summary["models"]})

    week_points: List[QaTsWeekPoint] = []
    if window_dates:
        actual_by_date = df.drop_duplicates(subset="target_date").set_index("target_date")["actual"]
        pred_lookup = {
            (r.model, r.target_date): r.prediction
            for r in df.itertuples() if r.target_date in set(window_dates)
        }
        for d in window_dates:
            preds = {m: pred_lookup.get((m, d)) for m in available_models}
            week_points.append(QaTsWeekPoint(
                target_date=pd.Timestamp(d).strftime("%Y-%m-%d"),
                actual=float(actual_by_date.loc[d]),
                predictions={k: (float(v) if v is not None else None) for k, v in preds.items()},
            ))

    summary_out = QaTsSummary(
        weeks_available=summary["weeks_available"], weeks_used=summary["weeks_used"],
        actual_consistent=summary["actual_consistent"], inconsistent_dates=summary["inconsistent_dates"],
        avg_actual=summary["avg_actual"], zero_demand_ratio=summary["zero_demand_ratio"],
        zero_demand_count=summary["zero_demand_count"], actual_total=summary["actual_total"],
        winner_model=summary["winner_model"],
        models=[QaTsModelSummary(**m) for m in summary["models"]],
        diagnostics=summary["diagnostics"],
    )
    return QaSkuTimeseriesResponse(
        sku_id=sku_id, center=center, horizon=horizon, weeks_window=weeks,
        available_models=available_models, weeks=week_points, summary=summary_out,
        evaluation_design="2024 Holdout",
        evaluation_start=week_points[0].target_date if week_points else None,
        evaluation_end=week_points[-1].target_date if week_points else None,
    )

class QaRepresentativeCase(BaseModel):
    key: str
    sku_id: str
    center: str
    actual_sum: float
    n_rows: int
    stat_wape: float
    ml_wape: float
    stat_bias: float
    ml_bias: float
    zero_ratio: float
    fallback_rows: int

@router.get("/qa/representative-cases", response_model=List[QaRepresentativeCase])
def get_qa_representative_cases(horizon: HorizonKey = Query("h1")):
    return [QaRepresentativeCase(**row) for row in qa_representative_skus(HORIZON_VALUES[horizon])]

@router.get("/qa/weeks")
def get_qa_weeks():
    df = qa_load_weekly_error()
    return {"weeks": sorted(df["target_week"].dropna().unique().tolist())}

class QaSkuPoint(BaseModel):
    sku_id:         str
    center:         str
    product_name:   Optional[str] = None
    option_code:    Optional[str] = None
    actual_sum:     float
    stat_wape:      float
    ml_wape:        float
    diff:           float
    winner:         str
    quartile:       str
    quartile_label: str

class QaSkuScatterResponse(BaseModel):
    center:   Center
    horizon:  HorizonKey
    total:    int
    points:   List[QaSkuPoint]

@router.get("/qa/sku-scatter", response_model=QaSkuScatterResponse)
def get_qa_sku_scatter(
    center:     Center     = Query("ALL"),
    horizon:    HorizonKey = Query("h1"),
    sku_search: Optional[str] = Query(None),
):
    df = final_filter_sku_compare(center, horizon)
    df = df.merge(qa_load_sku_product_lookup(), on=["center", "sku_id"], how="left")
    if sku_search:
        mask = (
            df["sku_id"].str.contains(sku_search, case=False, na=False, regex=False)
            | df["상품명"].str.contains(sku_search, case=False, na=False, regex=False)
        )
        df = df[mask]
    if df.empty:
        return QaSkuScatterResponse(center=center, horizon=horizon, total=0, points=[])

    df = df.assign(demand_quartile=final_assign_quartile(df["actual_sum"]))
    points = [
        QaSkuPoint(
            sku_id=r.sku_id, center=r.center,
            product_name=r.상품명 if pd.notna(r.상품명) else None,
            option_code=r.option_code if pd.notna(r.option_code) else None,
            actual_sum=float(r.actual_sum),
            stat_wape=float(r.stat_WAPE), ml_wape=float(r.ml_WAPE), diff=float(r.stat_WAPE - r.ml_WAPE),
            winner=r.winner, quartile=str(r.demand_quartile),
            quartile_label=FINAL_QUARTILE_LABELS.get(str(r.demand_quartile), str(r.demand_quartile)),
        )
        for r in df.itertuples()
    ]
    return QaSkuScatterResponse(center=center, horizon=horizon, total=len(points), points=points)

class QaSkuDetailResponse(BaseModel):
    sku_id:         str
    center:         str
    horizon:        HorizonKey
    product_name:   Optional[str] = None
    option_code:    Optional[str] = None
    actual_sum:     float
    stat_wape:      float
    ml_wape:        float
    stat_mae:       float
    ml_mae:         float
    winner:         str
    quartile:       str
    quartile_label: str

@router.get("/qa/sku-detail", response_model=QaSkuDetailResponse)
def get_qa_sku_detail(sku_id: str = Query(...), center: QaSkuCenter = Query(...), horizon: HorizonKey = Query("h1")):
    df = final_filter_sku_compare(center, horizon)
    hit = df[df["sku_id"] == sku_id]
    if hit.empty:
        raise HTTPException(status_code=404, detail="해당 조건에서 유효한 SKU 데이터를 찾을 수 없습니다.")
    row = hit.iloc[0]
    q = str(final_assign_quartile(pd.Series([row["actual_sum"]])).iloc[0])
    lookup = qa_load_sku_product_lookup()
    match = lookup[(lookup["center"] == center) & (lookup["sku_id"] == sku_id)]
    product_name = str(match["상품명"].iloc[0]) if not match.empty else None
    option_code = str(match["option_code"].iloc[0]) if not match.empty else None
    return QaSkuDetailResponse(
        sku_id=sku_id, center=center, horizon=horizon,
        product_name=product_name, option_code=option_code,
        actual_sum=float(row["actual_sum"]),
        stat_wape=float(row["stat_WAPE"]), ml_wape=float(row["ml_WAPE"]),
        stat_mae=float(row["stat_MAE"]), ml_mae=float(row["ml_MAE"]), winner=row["winner"],
        quartile=q, quartile_label=FINAL_QUARTILE_LABELS.get(q, q),
    )

class QaWeeklyPoint(BaseModel):
    week_index:      int
    target_week:     str
    actual:          float
    stat_prediction: Optional[float]
    ml_prediction:   Optional[float]
    stat_scope:      Optional[str]

class QaSkuWeeklyResponse(BaseModel):
    sku_id:  str
    center:  QaSkuCenter
    horizon: HorizonKey
    weeks:   List[QaWeeklyPoint]

@router.get("/qa/sku-weekly", response_model=QaSkuWeeklyResponse)
def get_qa_sku_weekly(sku_id: str = Query(...), center: QaSkuCenter = Query(...), horizon: HorizonKey = Query("h1")):
    df = qa_query_sku_predictions(sku_id, center, HORIZON_VALUES[horizon])
    if df.empty:
        raise HTTPException(status_code=404, detail="해당 SKU의 예측 데이터를 찾을 수 없습니다.")

    dates = sorted(df["target_date"].dropna().unique())
    week_index = {d: i + 1 for i, d in enumerate(dates)}
    stat_rows = df[df["model"] == QA_STAT_MODEL].set_index("target_date")
    ml_rows = df[df["model"] == QA_ML_MODEL].set_index("target_date")

    weeks: List[QaWeeklyPoint] = []
    for d in dates:
        if d in stat_rows.index:
            actual = float(stat_rows.loc[d, "actual"])
        elif d in ml_rows.index:
            actual = float(ml_rows.loc[d, "actual"])
        else:
            continue
        weeks.append(QaWeeklyPoint(
            week_index=week_index[d], target_week=pd.Timestamp(d).strftime("%Y-%m-%d"), actual=actual,
            stat_prediction=float(stat_rows.loc[d, "prediction"]) if d in stat_rows.index else None,
            ml_prediction=float(ml_rows.loc[d, "prediction"]) if d in ml_rows.index else None,
            stat_scope=str(stat_rows.loc[d, "comparison_scope"]) if d in stat_rows.index else None,
        ))

    return QaSkuWeeklyResponse(sku_id=sku_id, center=center, horizon=horizon, weeks=weeks)

class QaWeeklyErrorPoint(BaseModel):
    week_index:  int
    target_week: str
    stat_wape:   float
    ml_wape:     float
    stat_bias:   float
    ml_bias:     float

class QaWeeklyErrorResponse(BaseModel):
    center:          Center
    horizon:         HorizonKey
    points:          List[QaWeeklyErrorPoint]
    top_error_weeks: List[str]

@router.get("/qa/weekly-error", response_model=QaWeeklyErrorResponse)
def get_qa_weekly_error(center: Center = Query("ALL"), horizon: HorizonKey = Query("h1")):
    df = qa_load_weekly_error()
    df = df[
        ((df["model"] == QA_STAT_MODEL) & (df["variant"] == QA_STAT_VARIANT))
        | ((df["model"] == QA_ML_MODEL) & (df["variant"] == QA_ML_VARIANT))
    ]
    df = df[df["horizon"] == HORIZON_VALUES[horizon]]
    if center != "ALL":
        df = df[df["center"] == center]
    if df.empty:
        raise HTTPException(status_code=404, detail="데이터가 없습니다.")

    combined = qa_weighted_weekly_combine(df, ["model", "target_week"])
    weeks = sorted(combined["target_week"].unique())
    week_index = {w: i + 1 for i, w in enumerate(weeks)}
    stat = combined[combined["model"] == QA_STAT_MODEL].set_index("target_week")
    ml = combined[combined["model"] == QA_ML_MODEL].set_index("target_week")

    points: List[QaWeeklyErrorPoint] = []
    combined_score: Dict[str, float] = {}
    for w in weeks:
        if w not in stat.index or w not in ml.index:
            continue
        sw, mw = float(stat.loc[w, "WAPE"]), float(ml.loc[w, "WAPE"])
        points.append(QaWeeklyErrorPoint(
            week_index=week_index[w], target_week=w, stat_wape=sw, ml_wape=mw,
            stat_bias=float(stat.loc[w, "Bias"]), ml_bias=float(ml.loc[w, "Bias"]),
        ))
        combined_score[w] = (sw + mw) / 2

    top_error_weeks = sorted(combined_score, key=combined_score.get, reverse=True)[:5]
    return QaWeeklyErrorResponse(center=center, horizon=horizon, points=points, top_error_weeks=top_error_weeks)

class QaCoverageEntry(BaseModel):
    scope: str
    label: str
    count: int
    ratio: float

class QaCoverageResponse(BaseModel):
    source: Literal["sku", "aggregate"]
    model:  str
    total:  int
    entries: List[QaCoverageEntry]
    note:   str

@router.get("/qa/coverage", response_model=QaCoverageResponse)
def get_qa_coverage(
    center:     Center     = Query("ALL"),
    horizon:    HorizonKey = Query("h1"),
    sku_id:     Optional[str] = Query(None),
    sku_center: Optional[QaSkuCenter] = Query(None),
):
    if sku_id:
        if not sku_center:
            raise HTTPException(status_code=400, detail="sku_id 조회에는 sku_center가 필요합니다.")
        df = qa_query_sku_predictions(sku_id, sku_center, HORIZON_VALUES[horizon])
        stat_rows = df[df["model"] == QA_STAT_MODEL]
        if stat_rows.empty:
            raise HTTPException(status_code=404, detail="해당 SKU의 SARIMA 적용 범위 데이터를 찾을 수 없습니다.")
        counts = stat_rows["comparison_scope"].value_counts()
        total = int(counts.sum())
        entries = [
            QaCoverageEntry(scope=s, label=QA_SCOPE_LABELS.get(s, s), count=int(c), ratio=float(c) / total * 100)
            for s, c in counts.items()
        ]
        return QaCoverageResponse(
            source="sku", model=QA_STAT_MODEL, total=total, entries=entries,
            note=f"선택 SKU({sku_id})의 2024년 {horizon} 주차 중 SARIMA가 실제 적용된 범위입니다. "
                 "H-LGBM은 별도 Fallback/Cold-start 구분 없이 항상 Model-fit로 적용됩니다.",
        )

    df = qa_load_coverage_summary()
    sub = df[(df["model"] == QA_STAT_MODEL) & (df["variant"] == QA_STAT_VARIANT) & (df["horizon"] == HORIZON_VALUES[horizon])]
    if center != "ALL":
        sub = sub[sub["center"] == center]
    if sub.empty:
        raise HTTPException(status_code=404, detail="데이터가 없습니다.")

    g = sub.groupby("comparison_scope", as_index=False)["n_rows"].sum()
    total = int(g["n_rows"].sum())
    entries = [
        QaCoverageEntry(
            scope=r["comparison_scope"], label=QA_SCOPE_LABELS.get(r["comparison_scope"], r["comparison_scope"]),
            count=int(r["n_rows"]), ratio=float(r["n_rows"]) / total * 100,
        )
        for _, r in g.iterrows()
    ]
    return QaCoverageResponse(
        source="aggregate", model=QA_STAT_MODEL, total=total, entries=entries,
        note="SARIMA가 2024년 전체 예측 시점 중 실제 적용된 범위(Center/Horizon 집계)입니다. "
             "H-LGBM은 별도 Fallback/Cold-start 구분 없이 항상 Model-fit로 적용됩니다.",
    )

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

class QaParamRow(BaseModel):
    parameter_type:      str
    parameter:            str
    candidates_or_rule:   str
    selected_or_fixed:    str
    selection_stage:      str
    selection_criterion:  str

class QaParamModel(BaseModel):
    model:  str
    label:  str
    track:  str
    detail: Optional[Dict[str, str]]
    rows:   List[QaParamRow]

class QaParameterSummaryResponse(BaseModel):
    models: List[QaParamModel]

QA_PARAM_MODELS = ["ARIMA", "SARIMA", "ARIMAX", "SARIMAX", "RF", "LightGBM", "LSTM", "TFT", "Informer", "Hurdle-RF", "Hurdle-LightGBM"]
QA_PARAM_LABELS = {"ARIMA": "ARIMA", "SARIMA": "SARIMA", "ARIMAX": "ARIMAX", "SARIMAX": "SARIMAX", "RF": "RF", "LightGBM": "LGBM", "LSTM": "LSTM", "TFT": "TFT", "Informer": "Informer", "Hurdle-RF": "H-RF", "Hurdle-LightGBM": "H-LGBM"}
QA_PARAM_DETAIL_KEY = {"ARIMA": "ARIMA_S0", "ARIMAX": "ARIMAX_S4", "SARIMAX": "SARIMAX_S4", "SARIMA": "SARIMA"}

@router.get("/qa/parameter-summary", response_model=QaParameterSummaryResponse)
def get_qa_parameter_summary(models: Optional[str] = Query(None)):
    requested = [m.strip() for m in models.split(",")] if models else QA_PARAM_MODELS
    unknown = [m for m in requested if m not in QA_PARAM_MODELS]
    if unknown:
        raise HTTPException(status_code=404, detail=f"지원하지 않는 모델입니다: {unknown}")

    df = load_parameter_summary()
    out: List[QaParamModel] = []
    for m in requested:
        rows = df[df["model"] == m]
        if rows.empty:
            continue
        out.append(QaParamModel(
            model=m, label=QA_PARAM_LABELS.get(m, m), track=str(rows.iloc[0]["track"]),
            detail=MODEL_DETAILS.get(QA_PARAM_DETAIL_KEY.get(m, m)),
            rows=[
                QaParamRow(
                    parameter_type=r["parameter_type"], parameter=r["parameter"],
                    candidates_or_rule=r["candidates_or_rule"], selected_or_fixed=r["selected_or_fixed"],
                    selection_stage=r["selection_stage"], selection_criterion=r["selection_criterion"],
                )
                for _, r in rows.iterrows()
            ],
        ))
    return QaParameterSummaryResponse(models=out)
