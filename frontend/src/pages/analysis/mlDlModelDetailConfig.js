// 03 ML/DL 분석 — "선택 모델 상세" 팝업 전용 정적 config.
// 실제 값(HPO 탐색범위·고정값·h1/h2/h4 선정결과·2023/2024 성능)은 전부 기존 API
// (getMlDlModelDetail / getMlDlModelSummary / getFinalKpi)에서 그대로 받아 쓰고, 여기서는
// API에 없는 서술(구조 설명, 변수 의미/활용목적, 전처리 근거, 선행연구/공식문서 근거)만 관리한다.

const COLUMN_INFO = {
  '입수': '포장당 수량 — 상품 단위 특성 반영',
  'KAN_대분류': '상품 대분류 — 상품군 차이 반영', 'KAN_중분류': '상품 중분류 — 상품군 차이 반영', 'KAN_소분류': '상품 소분류 — 상품군 차이 반영',
  'ISO_주차': '연중 주차 — 주 단위 계절성 반영', '월': '달 — 월별 패턴 반영', '분기': '분기 — 분기별 패턴 반영',
  '평균온도': '주간 평균 기온 — 날씨와 수요 관계 반영', '총강수량': '주간 누적 강수량 — 강수와 수요 관계 반영',
  '강수량_호우_count': '호우 발생 횟수 — 집중 강수 영향 반영',
  'qty_log1p': '당해 주 실측 판매량의 log1p 값 — 현재 수준의 판매 이력 반영(target보다 과거 시점)',
  'qty_lag1_filled_log1p': '직전 1주 판매량의 결측 처리·log1p 값 — 최근 수요 반영',
  'qty_rollmean_4_filled_log1p': '4주 이동평균의 결측 처리·log1p 값 — 최근 수요 수준 반영',
  'qty_rollstd_4_filled_log1p': '4주 이동표준편차의 결측 처리·log1p 값 — 수요 변동성 반영',
  'adi_expanding_filled': '누적 평균 수요 발생 간격(ADI) — 수요 간헐성 반영',
  'cv2_expanding_filled': '누적 수요 변동계수 제곱(CV²) — 수요 변동성 반영',
  'weeks_since_last_active_filled': '마지막 판매 이후 경과 주수 — 판매 공백 반영',
  'existed_before_regime': '체계 변경 이전 존재 여부 — 상품 이력 특성 반영(B센터 판별에 사용)',
  'is_warmup': '이력 축적 초기 구간 여부 — 통계 특성이 아직 안정되지 않은 구간 표시',
  'coldstart_flag': '신규/이력 짧음 여부 — cold-start 상품 구분',
  'center_is_B': 'B센터 여부 — 센터 차이 반영',
  'covid_flag': '코로나19 관련 기간 여부 — 시기별 수요 차이 반영',
  'ccsi_lag_m1': '전월 소비자심리지수(CCSI) — 소비 심리 반영',
  'cpi_y1_prev': '전년 기준 소비자물가지수(CPI) — 물가 수준 반영',
  'cpi_y2_prev_yoy': '전전년 대비 물가 변화율(YoY) — 물가 변화 추세 반영',
  'temp_x_precip': '기온 × 강수량 교호항 — 복합 날씨 영향 반영',
  'center_temp_inter': '센터 × 기온 교호항 — 센터별 기온 영향 차이 반영',
};
// COLUMN_INFO 값은 "컬럼 설명 — 활용 목적" 형태라 표의 두 열로 나눠 쓴다.
function col(name) {
  const [meaning, purpose] = (COLUMN_INFO[name] || '설명 데이터 없음 — 확인된 설명 없음').split(' — ');
  return { col: name, meaning, purpose };
}
function cols(names) { return names.map(col); }

export function getMlDlColumnDescription(name) {
  if (COLUMN_INFO[name]) return COLUMN_INFO[name].split(' — ')[0];
  if (name.includes('공휴일_W0')) return '각 예측시점의 대상 주 공휴일 포함 여부';
  if (name.includes('공휴일_W-1')) return '각 예측시점 대상 주의 1주 전 공휴일 여부';
  if (name.includes('공휴일_W+1')) return '각 예측시점 대상 주의 1주 후 공휴일 여부';
  if (name.includes('→ target_log1p')) return '예측시점별 목표 판매수량을 log1p로 변환한 학습값';
  const targetMatch = name.match(/^target_h([124])$/);
  if (targetMatch) return `${targetMatch[1]}주 후 목표 판매수량`;
  return '모델 학습에 적용되는 입력 컬럼';
}

function holidayCols() {
  return [
    { col: 'target_h1/h2/h4_공휴일_W0', meaning: '각 예측시점의 대상 주에 공휴일(설·추석) 포함', purpose: '명절이 있는 주의 수요 변화 반영' },
    { col: 'target_h1/h2/h4_공휴일_W-1', meaning: '각 예측시점 대상 주의 1주 전이 공휴일 주', purpose: '명절 다음 주의 수요 변화 반영' },
    { col: 'target_h1/h2/h4_공휴일_W+1', meaning: '각 예측시점 대상 주의 1주 후가 공휴일 주', purpose: '명절 이전 주의 수요 변화 반영' },
  ];
}

// ML/Hurdle 공용 — get_model_feature_cols(horizon) 실제 반환 30개 컬럼을 5개 변수군으로 재분류(호라이즌 무관 동일 30개).
function mlVariableGroups() {
  return [
    { group: '상품 · 분류', items: cols(['KAN_대분류', 'KAN_중분류', 'KAN_소분류', '입수']) },
    { group: '수요 이력', items: cols(['qty_log1p', 'qty_lag1_filled_log1p', 'qty_rollmean_4_filled_log1p', 'qty_rollstd_4_filled_log1p', 'adi_expanding_filled', 'cv2_expanding_filled', 'weeks_since_last_active_filled']) },
    { group: '시간 · 공휴일', items: [...cols(['ISO_주차', '월', '분기']), ...holidayCols()] },
    { group: '외부환경', items: cols(['평균온도', '총강수량', '강수량_호우_count', 'covid_flag', 'ccsi_lag_m1', 'cpi_y1_prev', 'cpi_y2_prev_yoy']) },
    { group: '운영 · 구조', items: cols(['existed_before_regime', 'is_warmup', 'coldstart_flag', 'center_is_B', 'temp_x_precip', 'center_temp_inter']) },
  ];
}

// DL 공용 — 실제 사용 변수 27개를 업무 의미 기준으로 분류.
function dlVariableGroups() {
  return [
    { group: '상품 · 분류', items: cols(['KAN_대분류', 'KAN_중분류', 'KAN_소분류', '입수']) },
    { group: '수요 이력', items: cols(['qty_log1p', 'adi_expanding_filled', 'cv2_expanding_filled', 'weeks_since_last_active_filled']) },
    { group: '시간 · 공휴일', items: [...cols(['ISO_주차', '월', '분기']), ...holidayCols()] },
    { group: '외부환경', items: cols(['평균온도', '총강수량', '강수량_호우_count', 'covid_flag', 'ccsi_lag_m1', 'cpi_y1_prev', 'cpi_y2_prev_yoy']) },
    { group: '운영 · 구조', items: cols(['existed_before_regime', 'is_warmup', 'coldstart_flag', 'center_is_B', 'temp_x_precip', 'center_temp_inter']) },
  ];
}

const ML_TREE_PREPROCESSING_COMMON = [
  { target: '학습 목표 변환', columns: 'target_h1 · target_h2 · target_h4', method: '선택한 예측시점의 목표 판매수량에 1을 더해 로그 변환하고, 예측 후 expm1으로 원래 단위로 복원한 뒤 음수는 0으로 보정', reason: '큰 판매량의 영향을 완화하고 음수 예측을 방지' },
];

const DL_TIME_VARYING_STANDARDIZED_COLUMNS = [
  'ISO_주차', '월', '분기', 'covid_flag',
  'target_h1/h2/h4_공휴일_W0', 'target_h1/h2/h4_공휴일_W-1', 'target_h1/h2/h4_공휴일_W+1',
  '평균온도', '총강수량', 'qty_log1p', 'is_warmup', 'coldstart_flag',
  'adi_expanding_filled', 'cv2_expanding_filled', '강수량_호우_count', 'temp_x_precip',
  'center_temp_inter', 'weeks_since_last_active_filled', 'ccsi_lag_m1', 'cpi_y1_prev', 'cpi_y2_prev_yoy',
];
const DL_STATIC_STANDARDIZED_COLUMNS = ['입수', 'existed_before_regime', 'center_is_B'];

const RF_PARAMETER_ROWS = [
  { param: 'max_features', meaning: '분할 시 고려할 변수 비율', evidence: '선행연구 후보에 0.1, 1/3 포함', search: '{0.1, 1/3}' },
  { param: 'min_samples_leaf', meaning: 'leaf의 최소 표본 수', evidence: '선행연구 후보에 100, 500 포함', search: '{100, 500}' },
];
const LGBM_PARAMETER_ROWS = [
  { param: 'num_leaves', meaning: '트리 복잡도', evidence: '선행연구 8~1024 · 공식 기본값 31', search: '{8, 31}' },
  { param: 'min_child_samples', meaning: 'leaf의 최소 표본 수', evidence: '선행연구 5~5000', search: '{100, 1000}' },
];
const PARAMETER_UI = {
  RF: {
    references: [{ type: '참고 근거', author: 'Spiliotis et al. (2022)', title: 'Comparison of statistical and machine learning methods for daily SKU demand forecasting', note: '다수 상품의 간헐·불규칙 수요와 RF cross-learning 구조가 유사해 후보 설정에 참고' }],
    rows: RF_PARAMETER_ROWS,
    groups: [{ label: '회귀 모델', axes: [{ name: 'max_features', values: ['0.1', '1/3'] }, { name: 'min_samples_leaf', values: ['100', '500'] }] }],
  },
  LightGBM: {
    references: [{ type: '참고 근거', author: 'Sprangers et al. (2024)', title: 'Hierarchical Forecasting at Scale', note: '주단위 대규모 상품수요를 global LightGBM과 rolling validation으로 예측한 구조를 참고' }, { type: '공식문서', author: 'LightGBM', title: 'LGBMRegressor 공식 문서', note: 'num_leaves 공식 기본값 31 확인' }],
    rows: LGBM_PARAMETER_ROWS,
    groups: [{ label: '회귀 모델', axes: [{ name: 'num_leaves', values: ['8', '31'] }, { name: 'min_child_samples', values: ['100', '1000'] }] }],
  },
  LSTM: {
    references: [{ type: '모델 적용 근거', author: 'Bandara et al. (2019)', title: 'Sales Demand Forecast in E-commerce Using a Long Short-Term Memory Neural Network Methodology', note: '다상품 e-commerce 시계열을 global LSTM으로 학습한 구조가 본 프로젝트와 유사' }, { type: '파라미터 참고', author: 'Zhou et al. (2021)', title: 'Informer 논문의 LSTM baseline', note: 'hidden dimension 후보 {32, 64, 128, 256}에 실제 후보 포함' }],
    rows: [{ param: 'hidden_size', meaning: '은닉 상태 크기', evidence: '시계열 LSTM baseline 후보 {32,64,128,256} 중 32,64 사용', search: '{32, 64}' }],
    groups: [{ label: 'LSTM', axes: [{ name: 'hidden_size', values: ['32', '64'] }] }],
  },
  TFT: {
    references: [{ type: '모델 적용 근거', author: 'Lim et al. (2021)', title: 'Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting', note: '정적·미래 known·과거 observed 변수를 함께 쓰는 multi-horizon 구조를 참고' }, { type: '파라미터 참고', author: 'PyTorch Forecasting', title: 'TemporalFusionTransformer 공식 구현', note: '실제 구현체의 hidden_size 설정 범위 확인' }],
    rows: [{ param: 'hidden_size', meaning: '모델 내부 표현 차원', evidence: '공식 튜토리얼 참고값 8 · 공식 구현 범위 하한 16', search: '{8, 16}' }],
    groups: [{ label: 'TFT', axes: [{ name: 'hidden_size', values: ['8', '16'] }] }],
  },
  Informer: {
    references: [{ type: '참고 근거', author: 'Zhou et al. (2021)', title: 'Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting', note: '원논문 architecture 실험의 n_heads 후보 {8, 16}을 동일하게 적용' }],
    rows: [{ param: 'n_heads', meaning: 'Attention head 수', evidence: '원논문 후보 {8, 16}과 동일', search: '{8, 16}' }],
    groups: [{ label: 'Informer', axes: [{ name: 'n_heads', values: ['8', '16'] }] }],
  },
  'Hurdle-RF': {
    references: [{ type: '참고 근거', author: 'Spiliotis et al. (2022)', title: 'Comparison of statistical and machine learning methods for daily SKU demand forecasting', note: 'Hurdle 자체가 아닌 분류기·회귀기의 RF complexity 후보를 동일하게 설정하기 위해 참고' }],
    rows: [
      ...RF_PARAMETER_ROWS.map((row) => ({ ...row, param: `분류기 ${row.param}` })),
      ...RF_PARAMETER_ROWS.map((row) => ({ ...row, param: `회귀기 ${row.param}` })),
      { param: '회귀기 target_transform', meaning: '판매량 학습 단위', evidence: '', search: '{raw, log1p}' },
    ],
    groups: [
      { label: '분류기', axes: [{ name: 'max_features', values: ['0.1', '1/3'] }, { name: 'min_samples_leaf', values: ['100', '500'] }] },
      { label: '회귀기', axes: [{ name: 'max_features', values: ['0.1', '1/3'] }, { name: 'min_samples_leaf', values: ['100', '500'] }, { name: 'target_transform', values: ['raw', 'log1p'] }] },
    ],
  },
  'Hurdle-LightGBM': {
    references: [{ type: '참고 근거', author: 'Sprangers et al. (2024)', title: 'Hierarchical Forecasting at Scale', note: 'Hurdle 자체가 아닌 분류기·회귀기의 LightGBM complexity 후보를 동일하게 설정하기 위해 참고' }, { type: '공식문서', author: 'LightGBM', title: 'LGBMClassifier · LGBMRegressor 공식 문서', note: 'num_leaves 공식 기본값 31 확인' }],
    rows: [
      ...LGBM_PARAMETER_ROWS.map((row) => ({ ...row, param: `분류기 ${row.param}` })),
      ...LGBM_PARAMETER_ROWS.map((row) => ({ ...row, param: `회귀기 ${row.param}` })),
    ],
    groups: [
      { label: '분류기', axes: [{ name: 'num_leaves', values: ['8', '31'] }, { name: 'min_child_samples', values: ['100', '1000'] }] },
      { label: '회귀기', axes: [{ name: 'num_leaves', values: ['8', '31'] }, { name: 'min_child_samples', values: ['100', '1000'] }] },
    ],
  },
};

export const ML_DL_MODEL_DETAIL = {
  RF: {
    label: 'RF', track: 'ml',
    parameterUi: PARAMETER_UI.RF,
    mainCard: {
      principle: '여러 결정트리의 예측을 종합해 판매량을 예측',
      reason: '비선형 관계를 학습하는 기본 머신러닝 후보로 비교',
    },
    structure: {
      implementation: 'Random Forest 회귀 모델',
      trainingUnit: '여러 상품을 통합해 학습하는 Global 모델',
      predictionMethod: '상품·수요이력·시간·외부환경 변수를 이용해 판매량 예측',
      library: 'sklearn.ensemble.RandomForestRegressor',
      unit: '센터 × SKU × 주 row 단위 — A센터 데이터로 학습',
      flow: ['입력 30개 feature', '여러 결정 트리 학습(bagging)', '트리 예측 평균', 'log1p → expm1 역변환'],
      purpose: '판매 이력과 입력 변수의 비선형 관계를 트리 앙상블로 학습',
    },
    variableGroups: mlVariableGroups,
    preprocessing: [
      ...ML_TREE_PREPROCESSING_COMMON,
      { target: '결측값 보완', columns: 'adi_expanding_filled · cv2_expanding_filled · qty_lag1_filled_log1p · qty_rollmean_4_filled_log1p · qty_rollstd_4_filled_log1p', method: '학습 데이터의 KAN 소분류 → 중분류 → 대분류 → 전체 중앙값 순서로 보완', reason: 'RF가 빈 값을 직접 처리할 수 없어 상품군 특성을 최대한 유지해 보완' },
      { target: '범주형 변환', columns: 'KAN_대분류 · KAN_중분류 · KAN_소분류', method: '학습 데이터에서 범주별 숫자 열로 변환하고, 처음 보는 범주는 모두 0으로 처리', reason: '상품분류를 트리의 분기 조건으로 사용' },
    ],
    paramRationale: {
      priorResearch: {
        source: 'Spiliotis et al. (2022), "Comparison of statistical and machine learning methods for daily SKU demand forecasting", Operational Research',
        reason: '다수 SKU + intermittent/lumpy demand를 함께 다루는 cross-learning 연구로 본 프로젝트의 다수 SKU·희소수요 문제와 유사도가 높음',
        rows: [
          { param: 'ntree (n_estimators)', purpose: 'forest 크기', range: '{100, 250, 500, 1000}' },
          { param: 'mtry (max_features)', purpose: '분기마다 후보 변수 수', range: '{p/2, p/3, p/5, p/10}' },
          { param: 'nodesize (min_samples_leaf)', purpose: 'terminal node 최소 크기', range: '{5, 10, 100, 500}' },
        ],
      },
      officialDocs: {
        source: 'scikit-learn RandomForestRegressor 공식 문서',
        reason: '실제 사용 구현체의 파라미터 정의·기본값 확인 (R randomForest 파라미터를 sklearn에 대응)',
        rows: [
          { param: 'n_estimators', purpose: 'tree 수', range: '공식 기본값 100' },
          { param: 'max_features', purpose: '분기마다 사용할 feature 비율/수', range: '공식 기본값 1.0' },
          { param: 'min_samples_leaf', purpose: 'leaf 최소 표본 수', range: '공식 기본값 1' },
        ],
      },
    },
    hpoNarrative: '논문 범위 중 mtry({p/2,p/3,p/5,p/10} 중 1/3에 해당하는 1/3만), nodesize({5,10,100,500} 중 100·500만) 그리드가 극히 일부만 채택되었고, 그 구체적 숫자(0.1, 1/3, 100, 500)를 고른 이유는 코드에 없음(frozen spec으로만 명시).',
    fixedNarrative: 'n_estimators=100 고정은 "P12 profiling guard 호환을 위해 singleton 유지"라는 운영상 이유만 코드 주석에 있고, 성능적 근거는 없음.',
    lookbackNote: null,
    extraNotes: ['P13 HPO는 A센터 2023 4-fold(2023 Q1~Q4 expanding) 데이터로만 평가함 — B센터는 P13 단계에서 별도 평가하지 않음.'],
  },

  LightGBM: {
    label: 'LGBM', track: 'ml',
    parameterUi: PARAMETER_UI.LightGBM,
    mainCard: {
      principle: '이전 트리의 오차를 순차적으로 보완하며 판매량을 예측',
      reason: '다양한 상품·수요·외부 변수를 함께 활용하는 부스팅 모델로 비교',
    },
    structure: {
      implementation: 'LightGBM 회귀 모델',
      trainingUnit: '여러 상품을 통합해 학습하는 Global 모델',
      predictionMethod: '다양한 입력 변수와 판매량 사이의 비선형 관계를 학습해 수요 예측',
      library: 'lightgbm.LGBMRegressor',
      unit: '센터 × SKU × 주 row 단위 — A센터 데이터로 학습',
      flow: ['입력 30개 feature', 'gradient boosting 트리 순차 학습', 'log1p → expm1 역변환'],
      purpose: 'RF와 동일 입력으로 gradient boosting 구조의 성능 비교',
    },
    variableGroups: mlVariableGroups,
    preprocessing: [
      ...ML_TREE_PREPROCESSING_COMMON,
      { target: '범주형 처리', columns: 'KAN_대분류 · KAN_중분류 · KAN_소분류', method: '범주형 정보로 유지해 LightGBM이 분류값을 직접 구분하도록 처리', reason: '상품분류에 맞는 전용 분기 방식을 사용' },
    ],
    paramRationale: {
      priorResearch: {
        source: 'Sprangers et al. (2024), "Hierarchical Forecasting at Scale", International Journal of Forecasting',
        reason: '대규모 e-commerce 주단위 수요를 global/pooled 방식으로 학습하고 lag·calendar 등 tabular feature를 사용하는 구조가 본 프로젝트와 유사',
        rows: [
          { param: 'num_leaves', purpose: 'tree leaf 수(모델 복잡도)', range: '8 ~ 1024' },
          { param: 'min_child_samples', purpose: 'leaf 최소 표본 수', range: '5 ~ 5000' },
          { param: 'feature_fraction', purpose: 'feature subsampling', range: '0.4 ~ 1.0' },
          { param: 'bagging_fraction', purpose: 'row subsampling', range: '0.4 ~ 1.0' },
          { param: 'bagging_freq', purpose: 'bagging 수행 주기', range: '1 ~ 7' },
          { param: 'lambda_l1 / lambda_l2', purpose: 'L1/L2 규제', range: '1e-8 ~ 10' },
        ],
      },
      officialDocs: {
        source: 'LightGBM LGBMRegressor 공식 문서 · Optuna LightGBMTuner 공식 문서',
        reason: '구현체 파라미터 기본값 확인 + 공식 자동 튜닝 workflow가 실제로 무엇을 tuning 대상으로 보는지 확인',
        rows: [
          { param: 'num_leaves', purpose: '최대 leaf 수', range: '공식 기본값 31' },
          { param: 'min_child_samples', purpose: 'leaf 최소 data 수', range: '공식 기본값 20' },
          { param: 'learning_rate / n_estimators / max_depth', purpose: '학습 속도/트리 수/깊이 제한', range: '공식 기본값 0.1 / 100 / -1' },
          { param: 'LightGBMTuner 공식 tuning 대상', purpose: '공식 자동 튜닝이 다루는 파라미터', range: 'lambda_l1, lambda_l2, num_leaves, feature_fraction, bagging_fraction, bagging_freq, min_child_samples' },
        ],
      },
    },
    hpoNarrative: 'Optuna는 쓰지만 공식 LightGBMTuner(자동 튜닝 workflow)가 아니라 num_leaves×min_child_samples 2개만 GridSampler로 수동 탐색. LightGBMTuner 공식 대상 7개 중 나머지 5개(lambda_l1/l2, feature_fraction, bagging_fraction, bagging_freq)는 전부 고정값이며 하한값(feature/bagging_fraction=0.4)만 채택되고 논문 범위 탐색은 하지 않음.',
    fixedNarrative: 'feature_fraction/bagging_fraction=0.4는 논문 권장범위(0.4~1.0)의 하한만, lambda_l1/l2=0.0은 논문 범위(1e-8~10) 밖(규제 없음)으로 고정 — 값 선정 이유는 코드에 없음.',
    lookbackNote: null,
    extraNotes: ['P13 HPO는 A센터 2023 4-fold로만 평가.', 'use_best_iteration=False — early stopping 없이 고정 100 iteration까지 전부 학습.'],
  },

  LSTM: {
    label: 'LSTM', track: 'dl',
    parameterUi: PARAMETER_UI.LSTM,
    mainCard: {
      principle: '최근 수요 흐름을 순차적으로 학습해 미래 판매량을 예측',
      reason: '시계열 패턴 학습 효과를 확인하기 위한 딥러닝 후보',
    },
    structure: {
      implementation: 'PyTorch LSTM 시계열 모델',
      trainingUnit: '여러 상품을 통합해 학습하는 Global 모델',
      predictionMethod: '최근 13주 수요 흐름과 상품 정보를 학습해 미래 판매량 예측',
      library: 'PyTorch nn.LSTM 직접 구현(라이브러리 래퍼 아님)',
      unit: 'A센터 전체 SKU를 풀링한 global 모델 1개 (SKU별 개별 모델 아님)',
      flow: ['13주 lookback 시퀀스', 'LSTM으로 시계열 패턴 학습', '마지막 hidden state + static feature 결합', 'Linear로 h값 직접 예측(Direct)'],
      purpose: '과거 수요 시퀀스 자체의 패턴을 학습해 미래 판매량을 예측',
    },
    variableGroups: dlVariableGroups,
    preprocessing: [
      { target: '결측값 보완', columns: 'adi_expanding_filled · cv2_expanding_filled', method: '학습 데이터의 KAN 소분류 → 중분류 → 대분류 → 전체 중앙값 순서로 보완', reason: '연속된 입력 구간을 만들기 전에 빈 값을 제거' },
      { target: '시간가변 수치 변수 표준화', columns: DL_TIME_VARYING_STANDARDIZED_COLUMNS, method: '학습 데이터 기준 평균·표준편차로 표준화', reason: '변수 간 스케일 차이를 줄여 신경망 학습 안정화' },
      { target: '정적 수치 변수 표준화', columns: DL_STATIC_STANDARDIZED_COLUMNS, method: '학습 데이터 기준 평균·표준편차로 표준화', reason: '변수 간 스케일 차이를 줄여 신경망 학습 안정화' },
      { target: '범주형 변환', columns: 'KAN_대분류 · KAN_중분류 · KAN_소분류', method: '학습 데이터에서 정수 번호를 부여하고 처음 보는 범주는 미등록 값으로 처리', reason: '상품분류를 임베딩 입력으로 사용' },
      { target: '학습 목표 변환', columns: 'target_h1 · target_h2 · target_h4', method: '학습 오차를 계산할 때 목표 판매수량에 1을 더해 로그 변환', reason: '큰 판매량의 영향을 완화' },
    ],
    paramRationale: {
      priorResearch: {
        source: 'Bandara et al. (2019), "Sales Demand Forecast in E-commerce Using a Long Short-Term Memory Neural Network Methodology", ICONIP/Springer LNCS',
        reason: '실제 e-commerce 판매수요(sparse/intermittent 포함)를 여러 상품을 하나의 global LSTM으로 학습한 구조가 본 프로젝트의 pooled 방식과 유사',
        rows: [
          { param: 'LSTM cell dimension', purpose: 'hidden representation 크기', range: '50 ~ 100 (추가 검토: {32,64,128,256})' },
          { param: 'learning rate', purpose: 'gradient update 크기', range: '1e-6 ~ 1e-3' },
          { param: 'batch size', purpose: '배치 크기', range: '60 ~ 1500' },
          { param: 'L2 regularization', purpose: 'weight 규제', range: '1e-4 ~ 8e-4' },
        ],
      },
      officialDocs: {
        source: 'PyTorch torch.nn.LSTM 공식 문서',
        reason: '실제 사용 LSTM 구현체의 파라미터 의미·기본 구조 확인',
        rows: [
          { param: 'hidden_size', purpose: 'hidden state 차원', range: '필수 지정값(기본값 없음)' },
          { param: 'num_layers', purpose: 'stacked LSTM layer 수', range: '공식 기본값 1' },
          { param: 'dropout', purpose: 'LSTM layer 사이 dropout', range: '공식 기본값 0' },
        ],
      },
    },
    hpoNarrative: 'hidden_size만 {32,64} 2개 trial로 탐색(논문 범위 50~100·추가검토 {32,64,128,256} 중 32,64만 최종 채택). learning_rate=1e-3(논문 범위 상한), weight_decay=0.0(논문 L2 범위 1e-4~8e-4 밖), batch_size=1024는 논문 범위(60~1500) 안이지만 탐색 없이 고정.',
    fixedNarrative: 'num_layers=1, dropout=0.0은 PyTorch 공식 기본값과 동일. lookback=13은 별도 근거(아래 참고).',
    lookbackNote: 'P12 profiling 단계에서는 lookback 13과 26을 모두 실행했으나(cheap=13, high_cost=26), B센터 validation sequence 확보 가능 여부를 감사한 결과 lookback=26에서는 B센터 전 horizon·전 fold에서 validation sequence가 0이 되어, P13 HPO/최종학습에서는 lookback을 13으로 고정(HPO 탐색축 자체에 lookback이 없음). 성능이 아니라 B센터 검증 가능 여부가 근거.',
    extraNotes: ['EarlyStopping 없음 — max_epochs(20)를 전부 학습한 마지막 epoch 모델 사용.', 'P13 HPO는 A센터 2023 4-fold로만 평가.'],
  },

  TFT: {
    label: 'TFT', track: 'dl',
    parameterUi: PARAMETER_UI.TFT,
    mainCard: {
      principle: '상품·시간·수요 정보를 함께 학습하고 중요한 변수와 시점에 집중',
      reason: '다양한 시계열 변수의 활용 효과를 확인하기 위한 모델',
    },
    structure: {
      implementation: 'Temporal Fusion Transformer',
      trainingUnit: '여러 상품의 시계열을 함께 학습하는 Global 모델',
      predictionMethod: '과거 수요와 상품·시간·외부 변수를 함께 학습해 미래 수요 예측',
      library: 'pytorch_forecasting.TemporalFusionTransformer.from_dataset() (서드파티 라이브러리 그대로 사용)',
      unit: 'A센터 전체를 하나의 TimeSeriesDataSet으로 학습하는 global 모델',
      flow: ['13주 encoder + 1-step decoder 구성', 'static/known/observed 역할별 입력', 'Attention 기반 시계열 학습', 'h값 직접 예측(Direct)'],
      purpose: '변수 역할(static/known/observed)을 명시적으로 구분해 해석 가능한 attention 기반 구조로 예측',
    },
    variableGroups: dlVariableGroups,
    preprocessing: [
      { target: '결측값 보완', columns: 'adi_expanding_filled · cv2_expanding_filled', method: 'LSTM과 같은 상품분류 단계별 중앙값으로 보완', reason: '연속된 입력 구간을 만들기 전에 빈 값을 제거' },
      { target: '시간가변 수치 변수 표준화', columns: DL_TIME_VARYING_STANDARDIZED_COLUMNS, method: '학습 데이터 기준 평균·표준편차로 표준화', reason: '변수 간 스케일 차이를 줄여 신경망 학습 안정화' },
      { target: '정적 수치 변수 표준화', columns: DL_STATIC_STANDARDIZED_COLUMNS, method: '학습 데이터 기준 평균·표준편차로 표준화', reason: '변수 간 스케일 차이를 줄여 신경망 학습 안정화' },
      { target: '학습 목표 변환', columns: 'target_h1 · target_h2 · target_h4 → target_log1p', method: '목표 판매수량에 1을 더해 로그 변환하고 별도의 target 정규화는 적용하지 않음', reason: '큰 판매량의 영향을 완화하고 변환된 단위를 그대로 학습' },
      { target: '범주형 변환', columns: 'KAN_대분류 · KAN_중분류 · KAN_소분류', method: '학습 데이터에서 범주별 번호를 만들고 처음 보는 범주는 결측 범주로 처리', reason: '라이브러리의 범주형 인코더를 사용' },
    ],
    paramRationale: {
      priorResearch: {
        source: 'Lim et al. (2021), "Temporal Fusion Transformers for Interpretable Multi-Horizon Time Series Forecasting", International Journal of Forecasting',
        reason: 'TFT 원 논문 — static/known/observed covariate를 함께 처리하는 다수 entity multi-horizon 구조, retail 데이터셋 포함으로 적합도 높음',
        rows: [
          { param: 'hidden_size', purpose: '모델 내부 표현 차원', range: '{10, 20, 40, 80, 160, 240, 320}' },
          { param: 'attention_head_size', purpose: 'attention 병렬 head 수', range: '{1, 4}' },
          { param: 'dropout', purpose: 'regularization', range: '{0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9}' },
          { param: 'learning_rate', purpose: '학습 step 크기', range: '{1e-4, 1e-3, 1e-2}' },
        ],
      },
      officialDocs: {
        source: 'PyTorch Forecasting TemporalFusionTransformer 공식 문서 · 공식 demand forecasting tutorial',
        reason: '실제 사용 구현체의 reference architecture 확인 + 공식 튜토리얼의 실용적 소형 reference 값 확인',
        rows: [
          { param: 'hidden_size', purpose: '핵심 model capacity', range: '공식 tuning helper 참고범위 16 ~ 265' },
          { param: 'hidden_continuous_size', purpose: '연속변수 representation 크기', range: '참고범위 8 ~ 64' },
          { param: 'attention_head_size', purpose: 'attention head 수', range: '참고범위 1 ~ 4' },
          { param: '공식 튜토리얼 소형 reference', purpose: '실용 예제 설정', range: 'hidden_size=8, hidden_continuous_size=8, attention_head_size=1, dropout=0.1, gradient_clip_val=0.1' },
        ],
      },
    },
    hpoNarrative: 'hidden_size만 {8,16} 2개 trial로 탐색 — 원 논문 그리드({10,20,40,80,160,240,320})와 거의 겹치지 않고, 오히려 공식 튜토리얼의 소형 reference(hidden_size=8)와 정확히 일치.',
    fixedNarrative: 'hidden_continuous_size=8, attention_head_size=1, dropout=0.1, gradient_clip_val=0.1 전부 공식 튜토리얼 reference 값과 정확히 동일 — 원 논문 그리드보다 공식 튜토리얼 값을 그대로 채택한 것으로 보임. 3개 DL 모델 중 TFT만 gradient clipping(0.1)을 사용.',
    lookbackNote: 'LSTM과 동일한 근거로 P12에서 13/26 모두 시도했으나 B센터 validation feasibility 문제로 P13은 13 고정.',
    extraNotes: ['EarlyStopping 없음 — max_epochs(2)를 전부 학습.', 'P13 HPO는 A센터 2023 4-fold로만 평가.'],
  },

  Informer: {
    label: 'Informer', track: 'dl',
    parameterUi: PARAMETER_UI.Informer,
    mainCard: {
      principle: '과거 시계열의 중요한 시점에 집중해 미래 수요를 예측',
      reason: 'Attention 기반 시계열 학습 효과를 비교하기 위한 모델',
    },
    structure: {
      implementation: 'Informer 기반 시계열 Transformer',
      trainingUnit: '여러 상품의 시계열을 함께 학습하는 Global 모델',
      predictionMethod: 'Attention으로 과거 시계열 패턴을 학습해 미래 수요 예측',
      library: '커스텀 PyTorch 구현(ProbSparse self-attention encoder + encoder distilling + full-attention decoder, 원논문 구조 직접 재구현)',
      unit: 'A센터 전체를 풀링한 global encoder-decoder 모델',
      flow: ['13주 encoder 입력(known+observed)', 'label_len(=lookback/2) 과거 구간 + 1-step decoder', 'ProbSparse attention + distilling', 'h값 직접 예측(Direct)'],
      purpose: 'Long-sequence 효율화를 위한 ProbSparse attention 구조가 짧은 주간 시퀀스에서도 유효한지 확인',
    },
    variableGroups: dlVariableGroups,
    preprocessing: [
      { target: '결측값 보완', columns: 'adi_expanding_filled · cv2_expanding_filled', method: 'LSTM·TFT와 같은 상품분류 단계별 중앙값으로 보완', reason: '연속된 입력 구간을 만들기 전에 빈 값을 제거' },
      { target: '시간가변 수치 변수 표준화', columns: DL_TIME_VARYING_STANDARDIZED_COLUMNS, method: '학습 데이터 기준 평균·표준편차로 표준화', reason: '변수 간 스케일 차이를 줄여 신경망 학습 안정화' },
      { target: '정적 수치 변수 표준화', columns: DL_STATIC_STANDARDIZED_COLUMNS, method: '학습 데이터 기준 평균·표준편차로 표준화', reason: '변수 간 스케일 차이를 줄여 신경망 학습 안정화' },
      { target: '학습 목표 변환', columns: 'target_h1 · target_h2 · target_h4', method: '학습 오차를 계산할 때 목표 판매수량에 1을 더해 로그 변환', reason: '큰 판매량의 영향을 완화' },
    ],
    paramRationale: {
      priorResearch: {
        source: 'Zhou et al. (2021), "Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting", AAAI 2021',
        reason: 'Informer 원 논문 — ProbSparse Self-Attention·distilling·generative decoder의 직접적 구조 근거(단, 원 연구는 long-sequence 중심이라 task 차이 존재)',
        rows: [
          { param: 'e_layers', purpose: 'encoder 깊이', range: '{2, 3, 4, 6}' },
          { param: 'n_heads', purpose: 'multi-head attention 수', range: '{8, 16}' },
        ],
      },
      officialDocs: {
        source: 'Informer2020 원 저자 공식 GitHub 구현',
        reason: '프로젝트 구현과 비교할 reference architecture·training recipe 확인',
        rows: [
          { param: 'd_model / d_ff', purpose: 'attention embedding / feed-forward 차원', range: 'reference 512 / 2048' },
          { param: 'e_layers / d_layers', purpose: 'encoder/decoder layer 수', range: 'reference 2 / 1' },
          { param: 'n_heads', purpose: 'attention head 수', range: 'reference 8' },
          { param: 'factor', purpose: 'ProbSparse sampling factor', range: 'reference 5' },
          { param: 'dropout / learning rate / batch size', purpose: '정규화/학습률/배치', range: 'reference 0.05 / 1e-4 / 32' },
        ],
      },
    },
    hpoNarrative: 'n_heads만 {8,16} 2개 trial로 탐색 — 원 논문 그리드와 정확히 일치.',
    fixedNarrative: 'd_model/d_ff/e_layers/factor/dropout/learning_rate/batch_size 전부 공식 GitHub reference 값과 정확히 동일 — 3개 DL 모델 중 원 논문·공식 구현 reference와 가장 폭넓게 일치. e_layers=2는 논문 그리드({2,3,4,6}) 중 최솟값만 고정 채택(탐색 대상 아님).',
    lookbackNote: 'LSTM/TFT와 동일한 근거로 P12에서 13/26 모두 시도했으나 B센터 validation feasibility 문제로 P13은 13 고정.',
    extraNotes: ['EarlyStopping 없음 — max_epochs(2)를 전부 학습.', 'P13 HPO는 A센터 2023 4-fold로만 평가.'],
  },

  'Hurdle-RF': {
    label: 'H-RF', track: 'hurdle',
    parameterUi: PARAMETER_UI['Hurdle-RF'],
    mainCard: {
      principle: '판매 발생 여부와 발생 시 판매량을 분리해 예측',
      reason: '0수요가 많은 데이터의 과소예측 문제를 완화하기 위해 검토',
    },
    hurdle: {
      classifierRole: '판매 발생 여부(y>0) 분류 — RandomForestClassifier',
      regressorRole: '양수 수요 행만으로 판매량 회귀 — RandomForestRegressor',
      combination: 'P(y>0) × 조건부 예측값(regressor) = 최종 예측 (soft 곱셈, threshold 없음)',
    },
    structure: {
      implementation: 'Random Forest 분류기 + 회귀기 Hurdle 모델',
      trainingUnit: '여러 상품을 통합해 분류·회귀 모델 학습',
      predictionMethod: '판매 발생 확률과 발생 시 판매량을 각각 예측해 최종 수요 산출',
      library: 'sklearn RandomForestClassifier + RandomForestRegressor 2단계 결합',
      unit: '센터 × SKU × 주 row 단위 — A센터 데이터로 HPO(최종 재학습/2024 Holdout은 미실행 — H-LGBM에 밀려 프로덕션 미채택)',
      flow: ['입력 30개 feature(classifier/regressor 동일)', '① 판매 발생 여부 분류(0/1)', '② 발생 시 판매량 회귀(양수 행만)', 'P(발생) × 회귀예측 = 최종 예측'],
      purpose: '0수요가 많은 간헐적 수요 구조에서, 단일 회귀모델의 과소예측 문제를 분류+회귀 분리로 완화',
    },
    variableGroups: mlVariableGroups,
    preprocessing: [
      { target: '분류 목표 생성', columns: 'target_h1 · target_h2 · target_h4', method: '선택한 예측시점의 목표 판매수량이 0보다 큰지를 0/1 값으로 만들어 분류 모델을 학습', reason: '판매가 발생할 가능성을 먼저 예측' },
      { target: '회귀 학습행 선별', columns: 'target_h1 · target_h2 · target_h4', method: '선택한 예측시점의 목표 판매수량이 1 이상인 행만 회귀 모델 학습에 사용', reason: '판매가 발생한 경우의 수량만 별도로 예측' },
      { target: '결측값 보완', columns: 'adi_expanding_filled · cv2_expanding_filled · qty_lag1_filled_log1p · qty_rollmean_4_filled_log1p · qty_rollstd_4_filled_log1p', method: 'RF와 같은 상품분류 단계별 중앙값으로 분류기와 회귀기 입력을 보완', reason: '두 모델 모두 빈 값 없이 같은 입력 변수를 사용' },
      { target: '범주형 변환', columns: 'KAN_대분류 · KAN_중분류 · KAN_소분류', method: 'RF와 같은 범주별 숫자 열로 변환', reason: '상품분류를 분류기와 회귀기의 트리 분기에 사용' },
      { target: '회귀 목표 변환', columns: 'target_h1 · target_h2 · target_h4', method: '선택한 예측시점의 원래 판매량과 log1p 변환값을 비교하며, 실제 최종 설정은 원래 판매량을 사용', reason: '판매량 변환 방식까지 함께 비교' },
    ],
    paramRationale: {
      priorResearch: { source: 'RF와 동일 — Spiliotis et al. (2022) 범위를 재사용', reason: 'Hurdle-RF의 목적은 새 capacity 범위 도입이 아니라 동일 조건에서 분류/회귀 분리 효과만 보는 것이므로 RF와 같은 근거 사용', rows: [] },
      officialDocs: { source: 'scikit-learn RandomForestClassifier / RandomForestRegressor 공식 문서', reason: 'classifier·regressor 양쪽 구현 파라미터 정의·기본값 확인', rows: [] },
    },
    hpoNarrative: 'classifier: max_features{0.1,1/3}×min_samples_leaf{100,500}=4, regressor: 위 2개×target_transform{raw,log1p}=8 → 총 32 combination(48 fit). RF와 동일 그리드를 그대로 복제했으나 target_transform 축은 Hurdle-RF에만 추가됨.',
    fixedNarrative: 'n_estimators=100, max_depth=None, bootstrap=True는 RF와 동일. classifier만 criterion=gini, class_weight=None 추가.',
    lookbackNote: null,
    extraNotes: ['threshold 없음 — soft 곱셈 결합(hard threshold/calibration/class weighting 없음).', 'P13 HPO는 A센터 전용, 가드레일 통과 후 pooled WAPE가 H-LGBM보다 높아 최종 미채택 → 2024 Holdout 자체를 진행하지 않음.'],
  },

  'Hurdle-LightGBM': {
    label: 'H-LGBM', track: 'hurdle',
    parameterUi: PARAMETER_UI['Hurdle-LightGBM'],
    mainCard: {
      principle: '판매 발생 확률과 발생 시 판매량을 분리해 최종 수요를 산출',
      reason: '기존 LightGBM의 과소예측을 완화하면서 WAPE 성능을 유지하기 위해 검토',
    },
    hurdle: {
      classifierRole: '판매 발생 여부(y>0) 분류 — LGBMClassifier(objective=binary)',
      regressorRole: '양수 수요 행만으로 판매량 회귀 — LGBMRegressor(target=log1p, 예측 후 expm1)',
      combination: 'P(y>0) × expm1(regressor 예측) = 최종 예측 (soft 곱셈, threshold 없음)',
    },
    structure: {
      implementation: 'LightGBM 분류기 + 회귀기 Hurdle 모델',
      trainingUnit: '여러 상품을 통합해 분류·회귀 모델 학습',
      predictionMethod: '판매 발생 확률과 발생 시 판매량을 결합해 최종 수요 산출',
      library: 'lightgbm LGBMClassifier + LGBMRegressor 2단계 결합',
      unit: '센터 × SKU × 주 row 단위 — HP 선정은 A센터 전용, 최종 재학습은 A 전체 + B(체계변경 이후) pooled',
      flow: ['입력 30개 feature(classifier/regressor 동일)', '① 판매 발생 여부 분류(0/1)', '② 발생 시 판매량 회귀(양수 행만, log1p target)', 'P(발생) × expm1(회귀예측) = 최종 예측'],
      purpose: '0수요가 많은 간헐적 수요 구조에서 과소예측 문제 완화 — 실제 2024 Holdout까지 진행된 최종 프로덕션 모델',
    },
    variableGroups: mlVariableGroups,
    preprocessing: [
      { target: '분류 목표 생성', columns: 'target_h1 · target_h2 · target_h4', method: '선택한 예측시점의 목표 판매수량이 0보다 큰지를 0/1 값으로 만들어 분류 모델을 학습', reason: '판매가 발생할 가능성을 먼저 예측' },
      { target: '회귀 학습행 선별', columns: 'target_h1 · target_h2 · target_h4', method: '선택한 예측시점의 목표 판매수량이 1 이상인 행만 회귀 모델 학습에 사용', reason: '판매가 발생한 경우의 수량만 별도로 예측' },
      { target: '범주형 처리', columns: 'KAN_대분류 · KAN_중분류 · KAN_소분류', method: 'LightGBM이 범주형 정보를 직접 구분하도록 유지', reason: '분류기와 회귀기에서 같은 상품분류 정보를 사용' },
      { target: '회귀 목표 변환', columns: 'target_h1 · target_h2 · target_h4', method: '선택한 예측시점의 목표 판매수량에 1을 더해 로그 변환하고, 예측 후 expm1으로 원래 단위로 복원', reason: '큰 판매량의 영향을 완화하며 기존 LightGBM과 같은 변환 유지' },
    ],
    paramRationale: {
      priorResearch: { source: 'LightGBM과 동일 — Sprangers et al. (2024) 범위를 재사용', reason: '기존 LightGBM과 동일 complexity 범위를 사용해 성능 변화가 파라미터 범위가 아니라 Hurdle 구조 도입 자체에서 오도록 조건 통제', rows: [] },
      officialDocs: { source: 'LightGBM LGBMClassifier / LGBMRegressor 공식 문서', reason: 'classifier·regressor 각각의 실제 구현 기준값 확인', rows: [] },
    },
    hpoNarrative: 'classifier: num_leaves{8,31}×min_child_samples{100,1000}=4, regressor: 동일 그리드=4 → 4×4=16 combination. "기존 P13 LightGBM 그리드와 동일한 값"이라고 코드 docstring에 명시 — LightGBM과 동일 재사용.',
    fixedNarrative: 'learning_rate=0.1, n_estimators=100, feature_fraction/bagging_fraction=0.4 등 LightGBM과 동일 고정값. early_stopping/validation 미사용.',
    lookbackNote: null,
    extraNotes: [
      '⚠️ HP 선정(P13 HPO)은 A센터 데이터만 사용했지만, 실제 최종 모델은 A 전체 + B(2023-07-03 이후) pooled 데이터로 재학습함 — HP 선정 모집단과 최종 학습 모집단이 다름.',
      '⚠️ h4 pooled bias(-19.98%)가 가드레일(-20%) 경계에 근접, seed/fold 단위로는 경계를 넘는 경우도 존재.',
      'threshold 없음 — soft 곱셈 결합(hard threshold/calibration/class weighting 없음). archive 폴더의 threshold 탐색 스크립트는 더 이전 폐기 단계로 최종 트랙과 무관.',
    ],
  },
};
