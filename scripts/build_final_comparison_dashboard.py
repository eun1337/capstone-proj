"""Rebuild the 04 dashboard from stored predictions; never fit or change a model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'data/dashboard/final_model_comparison'
PRED = ROOT / 'outputs/model_comparison/cross_track_predictions_2024.parquet'
DEV = ROOT / 'data/development_2021_2023.parquet'
HOLD = ROOT / 'data/holdout_2024.parquet'
SKU_SOURCE = ROOT / 'outputs/model_comparison/sku_model_comparison_2024.csv'
SELECTION = ROOT / 'outputs/model_comparison/ml_dl_model_selection_2023.csv'
ZERO = ROOT / 'outputs/diagnostics/target_transform_bias/lightgbm_h1_zero_demand_diagnostic.json'
METRICS = ['WAPE', 'Bias', 'MAE', 'RMSE', 'MASE']
KEY = ['center_id', 'sku_id', 'forecast_origin', 'target_date', 'horizon']
UNIT = ['center_id', 'sku_id']
Q_LABELS = ['Q1', 'Q2', 'Q3', 'Q4']
SCOPES = ['model_fit', 'fallback', 'constant', 'full_coverage']
CONSTANT_COUNTS = ['constant_no_development_rows', 'constant_with_development_rows', 'constant_zero_rows', 'constant_positive_rows']
SUMS = ['n', 'actual_sum', 'stat_prediction_sum', 'ml_prediction_sum', 'stat_abs', 'ml_abs', 'stat_sq', 'ml_sq', 'stat_scaled', 'ml_scaled', 'mase_n'] + CONSTANT_COUNTS


def clean(value):
    if isinstance(value, dict): return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    if isinstance(value, np.generic): return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value): return None
    return value


def measure(g):
    sums = g[SUMS].sum()
    n, actual = float(sums['n']), float(sums['actual_sum'])
    units = g.groupby(UNIT, observed=True)['mase_n'].sum()
    valid = int((units > 0).sum())
    result = dict(n_rows=int(n), n_skus=len(units), actual_sum=actual, mase_valid_rows=int(sums['mase_n']), mase_valid_skus=valid, mase_excluded_skus=len(units)-valid, mase_excluded_pct=(len(units)-valid)/len(units)*100 if len(units) else None)
    result.update({key: int(sums[key]) for key in CONSTANT_COUNTS})
    for model in ['stat', 'ml']:
        result.update({f'{model}_WAPE': sums[f'{model}_abs']/actual*100 if actual > 0 else None,
                       f'{model}_Bias': (sums[f'{model}_prediction_sum']-actual)/actual*100 if actual > 0 else None,
                       f'{model}_MAE': sums[f'{model}_abs']/n if n else None,
                       f'{model}_RMSE': np.sqrt(sums[f'{model}_sq']/n) if n else None,
                       f'{model}_MASE': sums[f'{model}_scaled']/sums['mase_n'] if sums['mase_n'] else None})
    return clean(result)


def winner_rows(g, common):
    units = g.groupby(UNIT, as_index=False, observed=True)[SUMS].sum()
    results = []
    for metric in METRICS:
        a, b = [], []
        for model, dest in [('stat', a), ('ml', b)]:
            if metric == 'WAPE': value = units[f'{model}_abs']/units.actual_sum.replace(0, np.nan)*100
            elif metric == 'Bias': value = ((units[f'{model}_prediction_sum']-units.actual_sum)/units.actual_sum.replace(0, np.nan)*100).abs()
            elif metric == 'MAE': value = units[f'{model}_abs']/units.n
            elif metric == 'RMSE': value = np.sqrt(units[f'{model}_sq']/units.n)
            else: value = units[f'{model}_scaled']/units.mase_n.replace(0, np.nan)
            dest.extend(value)
        a, b = np.array(a), np.array(b)
        valid = np.isfinite(a) & np.isfinite(b)
        tie = valid & np.equal(a, b)
        masks = {'stat': valid & ~tie & (a < b), 'ml': valid & ~tie & (b < a), 'tie': tie}
        total, demand = int(valid.sum()), float(units.loc[valid, 'actual_sum'].sum())
        row = {**common, 'metric': metric, 'valid_skus': total, 'excluded_skus': len(units)-total, 'valid_actual_sum': demand}
        for model, mask in masks.items():
            row[f'{model}_sku_count'] = int(mask.sum())
            row[f'{model}_share'] = mask.sum()/total*100 if total else None
            model_demand = float(units.loc[mask, 'actual_sum'].sum())
            row[f'{model}_demand_sum'] = model_demand
            row[f'{model}_demand_share'] = model_demand/demand*100 if demand else None
        results.append(clean(row))
    return results


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    history = pd.read_parquet(DEV, columns=UNIT+['week_st', 'qty'])
    history = history[(history.week_st >= pd.Timestamp('2021-01-01')) & (history.week_st <= pd.Timestamp('2023-12-25')) & ((history.center_id == 'A') | ((history.center_id == 'B') & (history.week_st >= pd.Timestamp('2023-07-03'))))].sort_values(UNIT+['week_st'])
    assert not history.duplicated(UNIT+['week_st']).any(), 'Duplicate history weeks'
    history['diff'] = history.groupby(UNIT, observed=True).qty.diff().abs()
    scale = history.groupby(UNIT, as_index=False, observed=True).agg(development_n_obs=('qty', 'size'), mase_scale=('diff', 'mean'))
    first = pd.concat([history[UNIT+['week_st']], pd.read_parquet(HOLD, columns=UNIT+['week_st'])]).groupby(UNIT, as_index=False, observed=True).week_st.min().rename(columns={'week_st': 'first_week'})
    sku_source = pd.read_csv(SKU_SOURCE)
    _, edges = pd.qcut(sku_source.loc[sku_source.actual_sum > 0, 'actual_sum'], 4, retbins=True, duplicates='drop')
    assert len(edges) == 5, 'Existing Q definition is not four bins'
    dataset = ds.dataset(PRED)
    stats, audit, scope_sources = [], [], {}
    for center in ['A', 'B']:
        for horizon in [1, 2, 4]:
            base = (ds.field('center_id') == center) & (ds.field('horizon') == horizon)
            stat = dataset.to_table(filter=base & (ds.field('model') == 'SARIMA') & (ds.field('variant') == 'none'), columns=KEY+['actual','prediction','forecast_source']).to_pandas()
            ml = dataset.to_table(filter=base & (ds.field('model') == 'Hurdle-LightGBM') & (ds.field('variant') == 'operational_final'), columns=KEY+['actual','prediction']).to_pandas()
            assert not stat.duplicated(KEY).any() and not ml.duplicated(KEY).any(), 'Duplicate prediction key'
            common = stat.merge(ml, on=KEY, how='inner', suffixes=('_stat','_ml'), validate='one_to_one')
            assert np.allclose(common.actual_stat, common.actual_ml, rtol=0, atol=1e-9, equal_nan=True), 'Actual values disagree'
            assert np.isfinite(common[['actual_stat','prediction_stat','prediction_ml']]).all().all(), 'Non-finite prediction/actual'
            common = common.merge(scale, on=UNIT, how='left', validate='many_to_one').merge(first, on=UNIT, how='left', validate='many_to_one')
            assert common.first_week.notna().all(), 'Missing first observed week'
            origin = common.forecast_origin.fillna(common.target_date-pd.to_timedelta(common.horizon*7, unit='D'))
            cold = origin < common.first_week
            normal = common.forecast_source.eq('sarima')
            constant = common.forecast_source.eq('constant')
            common['scope'] = np.select([normal, constant], ['model_fit','constant'], default='fallback')
            no_development = common.development_n_obs.fillna(0).eq(0)
            common['constant_no_development_rows'] = (constant & no_development).astype(int)
            common['constant_with_development_rows'] = (constant & ~no_development).astype(int)
            common['constant_zero_rows'] = (constant & common.prediction_stat.eq(0)).astype(int)
            common['constant_positive_rows'] = (constant & common.prediction_stat.gt(0)).astype(int)
            assert (common.loc[constant, 'prediction_stat'] >= 0).all(), 'Negative constant forecast'
            assert not cold.any(), 'Origin-zero common rows require separate interpretation'
            for scope, g in common.groupby('scope'):
                scope_sources.setdefault(scope, set()).update(g.forecast_source.unique())
            common['actual_sum'] = common.actual_stat
            common['n'] = 1
            valid = np.isfinite(common.mase_scale) & (common.mase_scale > 0) & (common.development_n_obs >= 2)
            common['mase_n'] = valid.astype(int)
            for model in ['stat','ml']:
                error = common[f'prediction_{model}'] - common.actual_stat
                common[f'{model}_prediction_sum'] = common[f'prediction_{model}']
                common[f'{model}_abs'] = error.abs()
                common[f'{model}_sq'] = error**2
                common[f'{model}_scaled'] = (error.abs()/common.mase_scale).where(valid, 0)
            grouped = common.groupby(UNIT+['horizon','scope'], as_index=False, observed=True)[SUMS].sum()
            stats.append(grouped)
            audit.append(dict(center=center,horizon=horizon,stat_rows=len(stat),ml_rows=len(ml),common_rows=len(common),stat_unmatched=len(stat)-len(common),ml_unmatched=len(ml)-len(common),null_origins=int(common.forecast_origin.isna().sum()),origin_zero_common_rows=int(cold.sum())))
            print(f'{center} h{horizon}: {len(common):,} common rows', flush=True)
    stats = pd.concat(stats, ignore_index=True)
    all_units = stats[UNIT].drop_duplicates().merge(scale, on=UNIT, how='left')
    all_units['valid'] = np.isfinite(all_units.mase_scale) & (all_units.mase_scale > 0) & (all_units.development_n_obs >= 2)
    all_units['excluded_reason'] = np.select([all_units.development_n_obs.isna(), all_units.development_n_obs.lt(2), all_units.mase_scale.eq(0)], ['no_development_history','fewer_than_two_observations','zero_scale'], default='')
    all_units.to_csv(OUT/'mase_scale_2024.csv', index=False)
    full = stats.groupby(UNIT+['horizon'], as_index=False, observed=True)[SUMS].sum()
    previous = full.merge(sku_source, left_on=['center_id','sku_id','horizon'], right_on=['center','sku_id','horizon'], suffixes=('', '_previous'), validate='one_to_one')
    assert len(previous) == len(full) == len(sku_source), 'SKU comparison key mismatch'
    assert np.allclose(previous.actual_sum, previous.actual_sum_previous), 'Existing demand sums changed'
    assert (previous.n == previous.n_rows).all(), 'Existing common row counts changed'
    valid_actual = previous.actual_sum > 0
    stat_wape = previous.stat_abs / previous.actual_sum * 100
    ml_wape = previous.ml_abs / previous.actual_sum * 100
    assert np.allclose(stat_wape[valid_actual], previous.loc[valid_actual, 'stat_WAPE']), 'Existing stat WAPE mismatch'
    assert np.allclose(ml_wape[valid_actual], previous.loc[valid_actual, 'ml_WAPE']), 'Existing ML WAPE mismatch'
    expected_winner = np.where(stat_wape < ml_wape, 'SARIMA', np.where(ml_wape < stat_wape, 'Hurdle-LightGBM', 'tie'))
    assert (expected_winner[valid_actual] == previous.loc[valid_actual, 'winner']).all(), 'Existing winner rule mismatch'
    full['quartile'] = pd.cut(full.actual_sum.where(full.actual_sum > 0), edges, labels=Q_LABELS, include_lowest=True)
    assert full.loc[full.actual_sum > 0, 'quartile'].notna().all(), 'Positive-demand SKU outside fixed Q edges'
    overall, scopes, demand, winners = [], [], [], []
    for center in ['ALL','A','B']:
        for horizon in ['ALL',1,2,4]:
            g = full if center == 'ALL' else full[full.center_id == center]
            sg = stats if center == 'ALL' else stats[stats.center_id == center]
            if horizon != 'ALL': g, sg = g[g.horizon == horizon], sg[sg.horizon == horizon]
            common = {'center':center,'horizon':str(horizon)}
            overall.append({**common, **measure(g)})
            for scope in SCOPES:
                chunk = g if scope == 'full_coverage' else sg[sg.scope == scope]
                scopes.append({**common,'scope':scope,**measure(chunk)})
            for i, q in enumerate(Q_LABELS):
                chunk = g[g.quartile == q]
                meta = {**common,'quartile':q,'range_low':float(edges[i]),'range_high':float(edges[i+1]),**measure(chunk)}
                demand.append(meta)
                winners.extend(winner_rows(chunk, meta))
    trials = pd.read_csv(SELECTION).drop_duplicates(['model','horizon','trial'])
    p13 = {}
    for model in ['LightGBM','Hurdle-LightGBM']:
        g = trials[(trials.model == model) & (trials.horizon == 1)]
        selected = g[g.selected.astype(str).str.lower() == 'true']
        row = (selected if len(selected) else g.sort_values('pooled_wape')).iloc[0]
        p13[model] = {k: clean(row[k]) for k in ['trial','pooled_wape','pooled_bias','bias_pass','selected']}
    hurdle_final_comparison = []
    for horizon in [1, 2, 4]:
        item = {'horizon': horizon}
        for model in ['Hurdle-RF', 'Hurdle-LightGBM']:
            g = trials[(trials.model == model) & (trials.horizon == horizon)]
            selected = g[g.selected.astype(str).str.lower() == 'true']
            row = (selected if len(selected) else g.sort_values('pooled_wape')).iloc[0]
            item[model] = {k: clean(row[k]) for k in ['trial', 'pooled_wape', 'pooled_bias', 'bias_pass', 'selected']}
        hurdle_final_comparison.append(item)
    zero = json.loads(ZERO.read_text())
    zero_rates = {q: zero['by_demand_quartile'][q]['zero_rate']*100 for q in ['Q1','Q2']}
    tables = {'overall_metrics_2024':overall,'demand_type_winner_share_2024':winners,'demand_type_metrics_2024':demand,'coverage_scope_metrics_2024':scopes}
    for name, records in tables.items(): pd.DataFrame(records).to_csv(OUT/f'{name}.csv', index=False)
    payload = clean({**tables, 'metadata': {'models':['SARIMA','H-LGBM'],'q_edges':edges.tolist(),'join_keys':KEY,'audit':audit,'scope_sources':{k:sorted(v) for k,v in scope_sources.items()},'q_zero_excluded_sku_horizons':int((full.actual_sum == 0).sum()),'p13':p13,'hurdle_final_comparison':hurdle_final_comparison,'zero_rates':zero_rates,'constant_note':'Development 기준 상수예측: 이력이 없으면 0, 수요가 일정하면 해당 상수값. forecast-origin 기준 신규 SKU와 다름','cold_start_note':'forecast-origin 기준 신규 SKU는 두 모델의 동일 평가 row가 없어 직접 비교하지 않음','scope_note':'forecast_source=sarima만 정상 적용; naive_mean_override_row는 실제 대체 예측으로 분류','sources':[str(p.relative_to(ROOT)) for p in [PRED,DEV,HOLD,SKU_SOURCE,SELECTION,ZERO]]}})
    (OUT/'final_model_summary.json').write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(',',':'))+'\n')
    (OUT/'README.md').write_text(README, encoding='utf-8')
    print('Generated dashboard aggregates:',OUT,flush=True)

README = """# 04 최종 모델 비교 — 파생 집계

## 재생성 / canonical
저장소 루트에서 `.venv/bin/python scripts/build_final_comparison_dashboard.py` 실행.
원본 예측을 복사하거나 재학습하지 않는다. 이 폴더의 CSV가 04의 canonical 집계이며
`final_model_summary.json`은 같은 실행에서 CSV와 동일 레코드 및 메타데이터를 묶은 UI 번들이다.
04 페이지는 이 JSON을 Vite import로 읽으므로 재생성 후 frontend build가 필요하다.
기존 outputs/model_comparison 집계는 다른 페이지가 사용하므로 변경하지 않는다.

## 원본
- outputs/model_comparison/cross_track_predictions_2024.parquet: SARIMA/none 및 Hurdle-LightGBM/operational_final의 actual, prediction, forecast_source와 예측 키.
- data/development_2021_2023.parquet: MASE 척도 및 최초 관측 주.
- data/holdout_2024.parquet: 최초 관측 주를 찾는 데만 사용. MASE 척도/선정에 사용하지 않는다.
- outputs/model_comparison/sku_model_comparison_2024.csv: 기존 Q 경계 및 결과 일치 확인.
- outputs/model_comparison/ml_dl_model_selection_2023.csv: P13 h1 선정 trial/참고 후보.
- outputs/diagnostics/target_transform_bias/lightgbm_h1_zero_demand_diagnostic.json: P13 진단 Q1/Q2 zero_rate. 이 진단의 Q 정의는 Holdout Q와 다르다.

## 비교 키와 scope
키는 center_id + sku_id + forecast_origin + target_date + horizon 전체이며 one-to-one inner join.
동일 키 actual의 일치, finite 예측 및 키 중복을 검사한다. NULL origin은 키의 일부로 그대로
비교하고 이력 판정에서만 target_date - horizon*7일로 복원한다(현 산출물 NULL origin 0건).
각 원본의 unmatched row 및 common row 수는 JSON metadata.audit에 기록한다.
ALL은 A/B row를 합쳐 계산하며 센터 지표의 단순 평균이 아니다.

- model_fit: forecast_source='sarima'인 실제 SARIMA row. development_fallback_used는 분류에 사용하지 않는다.
- fallback: sarima/constant가 아닌 실제 대체 예측. naive_mean, naive_mean_override_row, naive_mean_override_sku 포함.
- constant: forecast_source='constant'인 동일 평가 row의 상수 예측 적용.
- full_coverage: 정상 SARIMA 적용 + 대체 예측 적용 + 상수 예측 적용의 두 모델 교집합 전체. 원본 단독 coverage 전체가 아니다.

원본 SARIMA 구현(08_sku_sarima_sarimax.py)의 constant는 Development 기준 처리다.
Development 이력이 없으면 0, Development 수요가 일정하면 해당 상수값을 사용한다.
constant(0)을 신규 SKU 또는 Cold-start로 해석하지 않는다.
Development 이력 유무는 원본 Development의 센터별 적용 기간에서 center×SKU 관측 수로 확인한다.
정규화 예측 파일의 비어 있는 development_n_obs를 0으로 간주하지 않는다.
forecast-origin 기준 신규 SKU(origin < 최초 관측 week_st)는 두 모델의 동일 평가 row가 없어
직접 비교하지 않으며 메인 scope에서 제외한다. metadata.audit.origin_zero_common_rows에 검증 건수를 기록한다.
원본 forecast_source별 분류는 metadata.scope_sources 참조.

## Q1~Q4
기존 서비스 model_final_compare_data.demand_quartile_edges와 같은 방식:
기존 SKU 비교 CSV의 actual_sum>0 전체 center×SKU×horizon에 pd.qcut(4)를 적용한 고정 경계
[1, 9, 33, 128, 38278]을 pd.cut(include_lowest=True)로 적용한다.
Q1 [1,9], Q2 (9,33], Q3 (33,128], Q4 (128,38278]. 단위는 SKU-horizon의 2024 실제수요 합.
0수요는 Q 분석에서만 제외하고 overall/scope의 WAPE/Bias/MAE/RMSE에는 포함한다.
분기/간헐성 분류가 아니며 Q1을 자동으로 간헐수요라고 부르지 않는다.
센터·시점 필터를 바꾸어도 경계를 다시 만들지 않는다.

## 계산식
오차 e = prediction - actual. WAPE = Σ|e| / Σactual ×100.
Bias = Σe / Σactual ×100. MAE = Σ|e| / n. RMSE = sqrt(Σe²/n).
Σactual=0이면 WAPE/Bias는 null. 표의 감소폭은 stat−ml, Bias는 |stat|−|ml|.
WAPE/Bias 감소폭 단위는 %p, MAE/RMSE는 판매수량, MASE는 무차원이다.
Q 구간 지표도 해당 구간 전체 row의 합에서 계산한다(SKU 지표 평균 금지).
우세 비율은 SKU별 metric을 재계산하여 비교하며 Bias는 절댓값으로 비교한다.
정확한 동률은 Tie로 별도 집계한다. WAPE의 기존 winner 라벨과 전 SKU 결과 일치를 검증한다.
metric 계산 불가 SKU는 winner denominator에서 제외하며 valid_skus/excluded_skus로 보고한다.
수요 비중은 해당 우세 SKU의 실제 수요 합 / metric 유효 SKU의 실제 수요 합.
Q별 horizon=ALL은 해당 Q에 배정된 SKU-horizon row를 합쳐 center×SKU 단위로 재평가한다.

## MASE
Development qty를 시간순 정렬하여 SKU별 mean(|qty_t−qty_(t−1)|)을 scale로 사용.
A: 2021-01-01~2023-12-25, B: 2023-07-03~2023-12-25.
scale<=0, non-finite, 또는 관측 2개 미만이면 MASE에서만 제외한다.
집계 MASE = Σ(valid row의 |e|/SKU scale) / valid row 수 (row 가중 방식).
유효/전체 SKU, 제외 비율, 유효 row를 보고한다. 스케일 제외가 다른 metric의 row를 제거하지 않는다.

## 파일 / 컬럼
- overall_metrics_2024.csv: center, horizon(1/2/4/ALL), 공통 평가 블록.
- demand_type_metrics_2024.csv: center, horizon, quartile, range_low/high, 공통 평가 블록.
- demand_type_winner_share_2024.csv: 위 컬럼 + metric, valid_skus/excluded_skus,
  stat/ml/tie_sku_count, stat/ml/tie_share(%), stat/ml/tie_demand_share(%).
- coverage_scope_metrics_2024.csv: center, horizon, scope, 공통 평가 블록.
- mase_scale_2024.csv: center_id, sku_id, development_n_obs, mase_scale, valid, excluded_reason.
- final_model_summary.json: 위 4개 집계 배열과 metadata(원본 경로, 키 검증, Q 경계, P13 결과, scope 실제 source).
공통 평가 블록: n_rows, n_skus(center×SKU unique), actual_sum, mase_valid_rows,
mase_valid_skus, mase_excluded_skus, mase_excluded_pct 및 stat/ml_{WAPE,Bias,MAE,RMSE,MASE}.
상수예측 row 분해(공통 평가 블록에 포함): constant_no_development_rows / constant_with_development_rows는
Development 이력 없음/있음, constant_zero_rows / constant_positive_rows는 저장 예측값 0/양수인 row 수다.
각 쌍은 해당 집계의 constant row 전체를 분할한다. 이력 구분과 값 구분은 서로 다른 분류이므로 네 값을 합산하지 않는다.
CSV 빈 셀/JSON null은 데이터 없음이며 0점이 아니다. MASE 제외 SKU의 n_skus는 전체 교집합 기준.

## 해석 한계
P13은 ML/DL 최종모델 선정, 2024는 최종 비교·검증이다. Holdout을 근거로 운영모델을
소급 변경하지 않는다. 예외 구간의 통계측은 SARIMA 자체가 아닌 실제 저장 fallback이며
정확한 source를 밝힌다. 수치만으로 통계적 유의성이나 일반적인 안정성을 단정하지 않는다.
"""

if __name__ == '__main__': main()
