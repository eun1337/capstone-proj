// 02 통계모델 분석 — "선택 모델 상세" 팝업 전용 정적 config.
// 여기 있는 값은 전부 실제 코드/조사 결과를 근거로 작성했으며, API로 받을 수 있는 값(2024 Holdout
// WAPE/Bias/MAE/RMSE 등)은 여기 두지 않고 StatModelAnalysis.jsx에서 api.getQaStatVariableEffect로 받는다.

const QTY = { col: 'qty_log1p', meaning: '센터 × SKU 주간 판매수량의 log1p 변환값', purpose: '과거 판매 추세·자기상관 패턴을 모델에 반영' };
const ECON = [
  { col: 'ccsi_lag_m1', meaning: '소비자심리지수(CCSI), 1개월 시차', purpose: '소비 심리 변화가 판매에 미치는 영향 반영' },
  { col: 'cpi_y1_prev', meaning: '소비자물가지수(CPI), 전년 기준값', purpose: '물가 수준 변화가 판매에 미치는 영향 반영' },
  { col: 'cpi_y2_prev_yoy', meaning: '소비자물가지수(CPI) 전전년 대비 YoY 변화율', purpose: '물가 변화율 추세를 반영' },
];
const COVID = [{ col: 'covid_flag', meaning: '코로나19 관련 기간 여부 플래그', purpose: '코로나 시기의 이례적 수요 변화를 반영' }];
const HOLIDAY = [
  { col: '공휴일_W0', meaning: '해당 주 자체에 공휴일(설·추석)이 포함', purpose: '명절이 있는 주의 판매 변화 반영' },
  { col: '공휴일_W-1', meaning: '해당 주의 1주 전이 공휴일 주', purpose: '명절 다음 주의 판매 변화 반영' },
  { col: '공휴일_W+1', meaning: '해당 주의 1주 후가 공휴일 주', purpose: '명절 이전 주의 판매 변화 반영' },
];

const ARIMA_OFFICIAL_DOCS_ORDER = {
  source: 'pmdarima auto_arima 공식 구현',
  reason: '프로젝트가 실제 사용하는 ARIMA 구조 탐색 구현체의 파라미터 정의·기본 탐색 상한을 확인하기 위함',
  rows: [
    { param: 'p (max_p)', purpose: '과거 판매값이 현재에 미치는 영향 범위(AR)', range: '공식 기본 상한 5' },
    { param: 'd (max_d)', purpose: '비정상성 제거를 위한 차분 횟수', range: '공식 기본 상한 2 · KPSS 검정으로 자동 결정' },
    { param: 'q (max_q)', purpose: '과거 예측오차가 현재에 미치는 영향 범위(MA)', range: '공식 기본 상한 5' },
    { param: 'test', purpose: '차분 필요성 판단 검정', range: 'KPSS' },
    { param: 'stepwise', purpose: '전수탐색 대신 단계적 구조 탐색', range: 'True' },
  ],
};
const ARIMA_ORDER_PRIOR = {
  source: '특정 응용논문 범위를 직접 전이하지 않음',
  reason: 'SKU마다 자기상관·추세·차분 필요성이 다르므로 단일 고정값보다 pmdarima auto_arima의 자동 차수 탐색 절차 자체를 기준으로 사용',
  rows: [],
};
const ARIMA_ACTUAL_PARAMS = [
  { param: 'start_p / max_p', purpose: 'AR 차수 탐색범위', mode: 'search', value: '0 ~ 5', reason: '공식문서 기본 상한 그대로 사용 (코드 근거 없음)' },
  { param: 'd / max_d', purpose: '차분 횟수', mode: 'auto', value: 'None(자동) · 상한 2 · KPSS 검정', reason: 'auto_arima가 KPSS 검정으로 SKU별 자동 결정' },
  { param: 'start_q / max_q', purpose: 'MA 차수 탐색범위', mode: 'search', value: '0 ~ 5', reason: '공식문서 기본 상한 그대로 사용 (코드 근거 없음)' },
  { param: 'with_intercept', purpose: '절편 포함 여부', mode: 'auto', value: 'auto_arima 기본값("auto")', reason: '별도 고정 없이 SKU별 자동 결정 결과를 그대로 저장' },
  { param: 'information_criterion', purpose: '구조 선택 기준', mode: 'fixed', value: 'AICc', reason: 'pmdarima 공식 aicc() API로 탐색 기준과 동일한 값 사용' },
  { param: 'stepwise', purpose: '탐색 방식', mode: 'fixed', value: 'True (전수탐색 아님)', reason: '다수 SKU 반복 탐색의 연산비용 절감' },
];
const ARIMA_PREPROCESSING = [
  { target: 'target(qty)', method: 'log1p 변환 후 학습', reason: '수요 분포 왜도 감소, 음수 예측 방지' },
  { target: '예측값', method: 'expm1 역변환 + 0 클리핑(max(pred,0))', reason: 'log-scale 예측을 원 수량으로 복원, 음수 방지' },
  { target: '미수렴 / 이력부족 / 상수시계열', method: 'fallback 계층: 동일 탐색 내 수렴 후보 중 AICc 최소 → cold-start(KAN 소→중→대→센터 평균) → naive_mean(이력 평균)', reason: 'auto_arima 적합 실패 시에도 항상 예측값 생성' },
];
const AB_CENTER_NOTE = 'A센터는 2021-01-04~2023-12-25 전체 이력 사용, B센터는 2023-07-03(체계 변경 이후)~만 사용. existed_before_regime 플래그는 B에만 존재.';

const ARIMAX_EXOG_DOCS = {
  source: 'pmdarima ARIMA exogenous regression 구조',
  reason: 'ARIMA order는 유지하면서 외생변수를 추가하는 실제 구현 방식을 확인하기 위함',
  rows: [],
};
const ARIMAX_ORDER_INHERIT_PARAM = { param: '(p,d,q) + with_intercept', purpose: '시계열 구조', mode: 'fixed', value: 'ARIMA(S0) 단계에서 확정한 값 그대로 사용, 재탐색 없음', reason: '외생변수 효과와 구조 변경 효과가 섞이지 않도록 구조를 고정하고 exog coefficient만 새로 fit' };

function arimaxFlow(exogLabel) {
  return ['판매이력 log1p', 'ARIMA(S0) 확정 (p,d,q) 상속', `exog 추가 적합(${exogLabel})`, '매 주 상태갱신(refit=False)', '4-step 예측 → expm1'];
}

export const STAT_MODEL_DETAIL = {
  ARIMA_S0: {
    label: 'ARIMA',
    typeLabel: '통계모델 · 비계절 · 외생변수 없음',
    structure: {
      library: 'pmdarima.arima.auto_arima(구조 탐색, development 1회) + pmdarima.ARIMA(적합·rolling 예측)',
      unit: 'SKU × 센터 단위 개별 모델 (수천 개)',
      flow: ['판매이력 log1p', 'auto_arima 구조탐색(AICc)', 'development 1회 적합', '매 주 상태갱신(refit=False)', '4-step 예측 → expm1 역변환'],
      purpose: '외생변수·계절성이 없는 가장 단순한 baseline. SARIMA/ARIMAX 도입 시 개선폭을 비교하는 기준점',
    },
    variableGroups: [{ group: '수요 시계열', items: [QTY] }],
    preprocessing: ARIMA_PREPROCESSING,
    paramRationale: { priorResearch: ARIMA_ORDER_PRIOR, officialDocs: ARIMA_OFFICIAL_DOCS_ORDER },
    actualParams: ARIMA_ACTUAL_PARAMS,
    finalParams: { mode: 'per_sku', note: '(p,d,q)+with_intercept는 SKU × 센터별로 개별 선정됨(공통 단일값 없음). A센터 최빈값은 (0,0,0)이지만 SKU마다 다름.' },
    selectionCriteria: [
      'SKU별 구조 선정 기준: auto_arima 탐색 후보 중 AICc 최소 구조 선택 — family(ARIMA vs SARIMA 등) 간 비교에는 사용하지 않음',
      '최소 AICc 후보가 수렴하지 않으면, 같은 탐색에서 나온 수렴 후보 중 AICc 최소 구조로 대체',
      'family 선정 기준(AICc 아님): 7개 통계모델이 전부 동시에 존재하는 단일 공통 panel(2,970,634행)의 2024 Holdout WAPE — A센터만 보면 ARIMA가 h1/h2/h4 전부 SARIMA보다 낮음(예: h1 76.81% vs 109.34%). 하지만 B센터 h1에서 ARIMA가 327.52%까지 발산(계절성 미반영)해 ALL(pooled)·B센터 운영 안정성 기준으로는 SARIMA가 대표모델로 선정됨(src/forecasting/statistical/13_all_family_common_panel.py, outputs/model_comparison/stat_all_family_common_panel_metrics_2024.csv)',
    ],
  },

  SARIMA: {
    label: 'SARIMA',
    typeLabel: '통계모델 · 계절성 · 외생변수 없음 · 최종 선정',
    structure: {
      library: 'statsmodels.tsa.statespace.sarimax.SARIMAX (비계절 order는 ARIMA 단계 상속, 계절 구조만 이 단계에서 탐색)',
      unit: 'SKU × 센터 단위 개별 모델',
      flow: ['ARIMA 단계 확정 (p,d,q) 상속', '계절주기 m 후보별 (P,D,Q) 그리드 적합', 'AICc 최소 계절구조 선택', '매 주 상태갱신(refit=False)', '4-step 예측 → expm1'],
      purpose: '판매 데이터의 반복 계절 패턴(명절 등)을 반영했을 때 baseline 대비 개선되는지 확인 — 통계 대표모델로 최종 선정',
    },
    variableGroups: [{ group: '수요 시계열', items: [QTY] }],
    preprocessing: [
      { target: 'target(qty)', method: 'log1p 변환(qty_log1p 컬럼 사용)', reason: 'ARIMA와 동일' },
      { target: 'SARIMAX 적합 옵션', method: 'simple_differencing=True, enforce_stationarity=False, enforce_invertibility=False', reason: '150주 내외 짧은 시계열에서 근단위근 계수를 강제 배제하지 않기 위함' },
      { target: '계절 차분 D', method: 'nsdiffs(m별 OCSB 검정, max_D=1)로 자동 결정', reason: '계절 차분 여부를 그리드서치 대신 검정으로 결정(연산 절감)' },
      { target: '예측값', method: 'expm1 역변환 + 0 클리핑 + 수치불안정 override(development 관측 최댓값의 100배 초과 또는 Inf 시 별도 처리)', reason: 'near-unit-root 계수가 반복 append로 발산하는 현상 방지' },
    ],
    paramRationale: {
      priorResearch: {
        source: '특정 논문의 13/26/52 값을 직접 전이하지 않음',
        reason: '주단위 수요에서 분기·반기·연간 수준의 계절주기 가능성을 검증하기 위해 task-specific 후보를 정의',
        rows: [{ param: 'm 후보', purpose: '계절주기 길이', range: '{13,26,52} (주단위 기준 약 분기·반기·연간 — 코드에 명문화된 근거는 없음)' }],
      },
      officialDocs: {
        source: 'statsmodels SARIMAX 공식 구현',
        reason: '계절 (P,D,Q,m) 구조의 정의·입력방식을 확인하기 위함',
        rows: [
          { param: 'P', purpose: '계절 AR 차수', range: '의미만 정의 · 특정 탐색범위 권장 없음' },
          { param: 'D', purpose: '계절 차분 횟수', range: '의미만 정의 · 특정 탐색범위 권장 없음' },
          { param: 'Q', purpose: '계절 MA 차수', range: '의미만 정의 · 특정 탐색범위 권장 없음' },
          { param: 'm', purpose: '계절주기 길이', range: '의미만 정의 · 특정 탐색범위 권장 없음' },
        ],
      },
    },
    actualParams: [
      { param: 'P, Q', purpose: '계절 AR/MA 차수', mode: 'search', value: '{0,1} 각각', reason: '2026-08-21 "Optimized Seasonal Grid" 확정 — 기존 {0,1,2}에서 축소(150주 시계열 과적합 방지 + 연산 가속, 팀 합의). 코드 주석에 사유가 남아있는 드문 항목' },
      { param: 'D', purpose: '계절 차분 횟수', mode: 'auto', value: '최대 1, OCSB 검정', reason: 'm 후보마다 1회 자동 결정, 그리드서치 대상 아님' },
      { param: 'm', purpose: '계절주기', mode: 'search', value: '{13,26,52}, 이력 104주 미만이면 52 제외', reason: 'm=52는 두 주기(104주) 이상 이력 필요 — B센터는 post-regime 최대 26주라 항상 {13,26}만 후보' },
      { param: '(p,d,q)', purpose: '비계절 구조', mode: 'fixed', value: 'ARIMA 단계 확정값 상속', reason: '재탐색 비용 회피 + 구조는 계절성과 무관하다는 설계 전제' },
    ],
    finalParams: { mode: 'per_sku', note: '계절구조 (P,D,Q,m)는 SKU × 센터별 개별 선정됨(공통 단일값 없음). A센터는 m=52 선택 비중이 가장 높고, B센터는 m=52 후보 자체가 없음.' },
    selectionCriteria: [
      'SKU별 구조 선정 기준: 후보 계절구조 중 AICc 최소 선택(비계절 (p,d,q)는 ARIMA 단계 값 상속) — 이 AICc는 family 간 비교에는 전혀 사용하지 않음',
      '최적 후보 미수렴 시 수렴 후보 중 AICc 최소로 대체',
      AB_CENTER_NOTE,
      'family 선정 기준(AICc 아님): ARIMA/ARIMAX-S1~S4/SARIMA/SARIMAX-S4 7개 모델이 전부 동시에 존재하는 단일 공통 panel(2,970,634행, src/forecasting/statistical/13_all_family_common_panel.py)의 2024 Holdout WAPE/Bias.',
      'SARIMA가 "모든 구간에서 최고"는 아님 — ALL(pooled) 기준 h2는 ARIMAX-S3(89.03%)가 SARIMA(95.73%)보다 낮고, h4는 ARIMA(85.32%)·ARIMAX-S2(82.94%)·ARIMAX-S3(88.36%) 전부 SARIMA(101.04%)보다 낮음. A센터만 보면 ARIMA/ARIMAX-S2/S3가 h1/h2/h4 전 구간에서 SARIMA보다 WAPE가 낮음(예: A h1 ARIMA 76.81% vs SARIMA 109.34%).',
      'SARIMA를 대표모델로 선정한 실질적 근거는 "항상 최저 WAPE"가 아니라 B센터 h1에서의 운영 안정성 — 계절성을 반영하지 않는 ARIMA/ARIMAX-S1~S4는 B h1에서 WAPE 282%~10^76까지 발산하는 반면(계절성 없이 명절 등 반복 수요를 못 잡음), SARIMA만 95.06%로 안정적. h2도 SARIMA 86.94% vs ARIMA 계열 127~1.6×10^17로 SARIMA가 유일하게 정상 범위. 이 발산 회피가 SARIMA 선정의 핵심 근거이며, h4나 A센터 단독 비교에서의 절대 우위를 의미하지 않음(outputs/model_comparison/stat_all_family_common_panel_metrics_2024.csv).',
    ],
  },

  ARIMAX_S1: {
    label: 'ARIMAX-S1 Economic',
    typeLabel: '통계모델 · 비계절 · 외생변수(경제지표 3종)',
    structure: {
      library: 'pmdarima.ARIMA (ARIMA order 유지, X=exog로 경제지표 3종 추가)',
      unit: 'SKU × 센터 단위 개별 모델',
      flow: arimaxFlow('경제지표 3종'),
      purpose: '소비자심리지수·물가지수 같은 경제지표가 판매량 예측에 도움이 되는지 확인하기 위한 실험 모델',
    },
    variableGroups: [{ group: '수요 시계열', items: [QTY] }, { group: '외생변수 · 경제', items: ECON }],
    preprocessing: [...ARIMA_PREPROCESSING, { target: '미래 경제지표', method: 'ccsi_lag_m1은 미래 스텝에서 origin 시점 마지막 관측값으로 고정 공급', reason: '발표 전 정보가 미래 시점에 유입되는 leakage 방지' }],
    paramRationale: {
      priorResearch: { source: '별도의 ARIMAX 차수 범위 연구를 적용하지 않음', reason: 'ARIMA 구조를 고정한 상태에서 경제지표 추가 효과만 분리해서 보기 위함 — (p,d,q)까지 다시 탐색하면 구조 변경 효과와 섞이므로 상속 방식이 적합', rows: [] },
      officialDocs: ARIMAX_EXOG_DOCS,
    },
    actualParams: [ARIMAX_ORDER_INHERIT_PARAM, { param: 'exog block', purpose: '경제지표 3종 포함 여부', mode: 'fixed', value: 'S1 = ccsi_lag_m1, cpi_y1_prev, cpi_y2_prev_yoy', reason: '5개 block(S0~S4) 모두 사전 정의된 고정 조합 — 성능 기반 자동탐색 대상 아님' }],
    finalParams: { mode: 'none', note: 'block 자체가 고정 설계이며, order는 ARIMA(S0)의 SKU별 값을 그대로 사용(개별 선정)' },
    selectionCriteria: ['별도 구조 탐색 없음 — S0~S4 5개 block을 2024 Holdout에서 병렬 평가', '일부 SKU에서 exog 계수 발산으로 예측이 극단적으로 커져 2024 Holdout 정상 비교에서 제외됨(카드4 참고)'],
  },
  ARIMAX_S2: {
    label: 'ARIMAX-S2 COVID',
    typeLabel: '통계모델 · 비계절 · 외생변수(COVID 1종)',
    structure: {
      library: 'pmdarima.ARIMA (ARIMA order 유지, X=exog로 covid_flag 추가)',
      unit: 'SKU × 센터 단위 개별 모델',
      flow: arimaxFlow('COVID 지표'),
      purpose: '코로나19 시기의 이례적 수요 변화를 외생변수로 반영했을 때 예측이 개선되는지 확인하기 위한 실험 모델',
    },
    variableGroups: [{ group: '수요 시계열', items: [QTY] }, { group: '외생변수 · COVID', items: COVID }],
    preprocessing: ARIMA_PREPROCESSING,
    paramRationale: {
      priorResearch: { source: '별도의 ARIMAX 차수 범위 연구를 적용하지 않음', reason: 'ARIMA 구조를 고정한 상태에서 COVID 기간 정보 하나가 미치는 추가 효과를 독립적으로 확인하기 위함', rows: [] },
      officialDocs: ARIMAX_EXOG_DOCS,
    },
    actualParams: [ARIMAX_ORDER_INHERIT_PARAM, { param: 'exog block', purpose: 'COVID 지표 포함 여부', mode: 'fixed', value: 'S2 = covid_flag', reason: '5개 block(S0~S4) 모두 사전 정의된 고정 조합' }],
    finalParams: { mode: 'none', note: 'block 자체가 고정 설계이며, order는 ARIMA(S0)의 SKU별 값을 그대로 사용(개별 선정)' },
    selectionCriteria: ['별도 구조 탐색 없음 — S0~S4 5개 block을 2024 Holdout에서 병렬 평가', 'ARIMA baseline 대비 근소한 차이 — A/B 센터 모두 비교적 안정'],
  },
  ARIMAX_S3: {
    label: 'ARIMAX-S3 Holiday',
    typeLabel: '통계모델 · 비계절 · 외생변수(공휴일 3종)',
    structure: {
      library: 'pmdarima.ARIMA (ARIMA order 유지, X=exog로 공휴일 3종 추가)',
      unit: 'SKU × 센터 단위 개별 모델',
      flow: arimaxFlow('공휴일 지표 3종'),
      purpose: '설·추석 등 공휴일 전후의 판매 변화를 외생변수로 반영했을 때 예측이 개선되는지 확인하기 위한 실험 모델',
    },
    variableGroups: [{ group: '수요 시계열', items: [QTY] }, { group: '외생변수 · 공휴일', items: HOLIDAY }],
    preprocessing: [...ARIMA_PREPROCESSING, { target: '미래 공휴일 exog', method: '4-step forecast 각 시점의 실측 캘린더 값을 그대로 사용', reason: '공휴일은 미래라도 미리 확정된 값이므로 origin 고정 없이 실제 날짜로 조회' }],
    paramRationale: {
      priorResearch: { source: '별도의 ARIMAX 차수 탐색 연구를 적용하지 않음', reason: '공휴일 전·당·후주의 수요 변화 효과를 ARIMA 구조 변경 없이 비교하기 위함', rows: [] },
      officialDocs: ARIMAX_EXOG_DOCS,
    },
    actualParams: [ARIMAX_ORDER_INHERIT_PARAM, { param: 'exog block', purpose: '공휴일 지표 포함 여부', mode: 'fixed', value: 'S3 = 공휴일_W0, 공휴일_W-1, 공휴일_W+1', reason: '5개 block(S0~S4) 모두 사전 정의된 고정 조합' }],
    finalParams: { mode: 'none', note: 'block 자체가 고정 설계이며, order는 ARIMA(S0)의 SKU별 값을 그대로 사용(개별 선정)' },
    selectionCriteria: ['별도 구조 탐색 없음 — S0~S4 5개 block을 2024 Holdout에서 병렬 평가', 'ARIMA baseline 대비 근소한 차이 — A/B 센터 모두 비교적 안정'],
  },
  ARIMAX_S4: {
    label: 'ARIMAX-S4 Operational-Full',
    typeLabel: '통계모델 · 비계절 · 외생변수(경제+COVID+공휴일 7종)',
    structure: {
      library: 'pmdarima.ARIMA (ARIMA order 유지, X=exog로 경제·COVID·공휴일 전체 7종 추가)',
      unit: 'SKU × 센터 단위 개별 모델',
      flow: arimaxFlow('경제+COVID+공휴일 전체 7종'),
      purpose: 'S1(경제) + S2(COVID) + S3(공휴일)를 모두 결합했을 때의 효과를 확인하기 위한 실험 모델',
    },
    variableGroups: [{ group: '수요 시계열', items: [QTY] }, { group: '외생변수 · 경제', items: ECON }, { group: '외생변수 · COVID', items: COVID }, { group: '외생변수 · 공휴일', items: HOLIDAY }],
    preprocessing: [...ARIMA_PREPROCESSING, { target: '미래 경제/공휴일 exog', method: 'ccsi_lag_m1은 origin 고정값, 공휴일은 실측 캘린더 값 사용', reason: 'leakage 방지(경제) + 공휴일은 미리 확정된 값이라 실제 값 사용' }],
    paramRationale: {
      priorResearch: { source: '별도의 ARIMAX parameter range 연구를 적용한 것이 아니라 S1~S3의 운영 가능한 외생변수를 결합한 사전 정의 모델', reason: '경제·COVID·공휴일 정보를 함께 사용했을 때의 효과를 확인하기 위함', rows: [] },
      officialDocs: ARIMAX_EXOG_DOCS,
    },
    actualParams: [ARIMAX_ORDER_INHERIT_PARAM, { param: 'exog block', purpose: '전체 exog 포함 여부', mode: 'fixed', value: 'S4 = S1(경제 3종)+S2(COVID 1종)+S3(공휴일 3종) = 7종 전체', reason: '5개 block(S0~S4) 모두 사전 정의된 고정 조합' }],
    finalParams: { mode: 'none', note: 'block 자체가 고정 설계이며, order는 ARIMA(S0)의 SKU별 값을 그대로 사용(개별 선정)' },
    selectionCriteria: ['별도 구조 탐색 없음 — S0~S4 5개 block을 2024 Holdout에서 병렬 평가', 'exog 수가 가장 많아 계수 불안정·발산 빈도가 가장 높음 — 2024 Holdout 정상 비교에서 제외됨(카드4 참고)'],
  },

  SARIMAX_S4: {
    label: 'SARIMAX',
    typeLabel: '통계모델 · 계절성 + 외생변수(전체 7종)',
    structure: {
      library: 'statsmodels SARIMAX (SARIMA와 동일 후보 그리드를 공유하되, exog를 포함한 상태에서 독립적으로 적합)',
      unit: 'SKU × 센터 단위 개별 모델',
      flow: ['SARIMA와 동일한 (p,d,q),(P,D,Q,m) 후보 그리드 사용', '경제+COVID+공휴일 exog 7종 포함해 독립 적합', 'AICc 기준으로 SARIMA와 별도로 수렴 여부 판정', '매 주 상태갱신 → 4-step 예측 → expm1'],
      purpose: '계절성과 모든 외생변수를 함께 결합했을 때 SARIMA보다 더 나은지 확인하기 위한 실험 모델',
    },
    variableGroups: [{ group: '수요 시계열', items: [QTY] }, { group: '외생변수 · 경제', items: ECON }, { group: '외생변수 · COVID', items: COVID }, { group: '외생변수 · 공휴일', items: HOLIDAY }],
    preprocessing: [
      { target: 'target(qty)', method: 'log1p 변환(SARIMA와 동일)', reason: 'ARIMA/SARIMA와 동일' },
      { target: 'SARIMAX 적합 옵션', method: 'simple_differencing=True, enforce_stationarity=False, enforce_invertibility=False', reason: 'SARIMA와 동일' },
      { target: '미래 경제/공휴일 exog', method: 'ccsi_lag_m1은 origin 고정값, 공휴일은 실측 캘린더 값 사용', reason: 'leakage 방지 + 공휴일은 미리 확정된 값' },
      { target: '예측값', method: 'expm1 역변환 + 0 클리핑 + 수치불안정 override(SARIMA와 독립 판정)', reason: '근단위근 계수 발산 방지' },
    ],
    paramRationale: {
      priorResearch: { source: '별도의 SARIMAX 구조 범위를 새로 정의한 연구를 적용하지 않음', reason: 'SARIMA에서 확정한 계절구조 후보를 유지한 상태에서 운영 가능한 외생정보의 추가 효과만 비교하기 위함', rows: [] },
      officialDocs: { source: 'statsmodels SARIMAX 공식 문서', reason: 'SARIMA 구조와 exogenous regressor를 함께 사용하는 구현 방식을 확인하기 위함', rows: [] },
    },
    actualParams: [
      { param: '(p,d,q), (P,D,Q,m)', purpose: '시계열 + 계절 구조', mode: 'fixed', value: 'SARIMA와 동일한 후보 그리드 사용, 재탐색 없음', reason: 'SARIMA 최종 order를 단순 상속하는 것이 아니라, SARIMA와 동일한 후보 범위를 사용해 외생변수를 포함한 상태에서 독립 적합함(수렴 여부가 SARIMA와 다를 수 있음)' },
      { param: 'exog block', purpose: '전체 exog 포함 여부', mode: 'fixed', value: 'S4 = 경제 3종 + COVID 1종 + 공휴일 3종 = 7종 전체 (S1~S3 단독 변형은 SARIMAX에서 미구현)', reason: 'SARIMAX는 S4만 구현 — S1~S3 단독 조합을 만들지 않은 설계 배경은 코드 근거 없음' },
    ],
    finalParams: { mode: 'per_sku', note: '계절구조는 SKU × 센터별 개별 선정(SARIMA와 동일 후보 공유, 값은 SKU별로 다름)' },
    selectionCriteria: ['SARIMA와 동일한 후보 범위에서 exog 포함 상태로 독립 적합 후 수렴 여부 판정', '계절성 + 외생변수를 결합했지만 SARIMA 대비 WAPE가 더 높고 변동폭이 커서 최종 대표모델로는 미선정'],
  },
};
