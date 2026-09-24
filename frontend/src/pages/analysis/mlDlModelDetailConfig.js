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

function holidayCols(horizon) {
  const h = horizon || 'h1';
  return [
    { col: `target_${h}_공휴일_W0`, meaning: '예측 대상 주 자체에 공휴일(설·추석) 포함', purpose: '명절이 있는 주의 수요 변화 반영' },
    { col: `target_${h}_공휴일_W-1`, meaning: '예측 대상 주의 1주 전이 공휴일 주', purpose: '명절 다음 주의 수요 변화 반영' },
    { col: `target_${h}_공휴일_W+1`, meaning: '예측 대상 주의 1주 후가 공휴일 주', purpose: '명절 이전 주의 수요 변화 반영' },
  ];
}

// ML/Hurdle 공용 — get_model_feature_cols(horizon) 실제 반환 30개 컬럼을 5개 변수군으로 재분류.
function mlVariableGroups(horizon) {
  return [
    { group: '상품 · 분류', items: cols(['KAN_대분류', 'KAN_중분류', 'KAN_소분류', '입수']) },
    { group: '수요 이력', items: cols(['qty_log1p', 'qty_lag1_filled_log1p', 'qty_rollmean_4_filled_log1p', 'qty_rollstd_4_filled_log1p', 'adi_expanding_filled', 'cv2_expanding_filled', 'weeks_since_last_active_filled']) },
    { group: '시간 · 공휴일', items: [...cols(['ISO_주차', '월', '분기']), ...holidayCols(horizon)] },
    { group: '외부환경', items: cols(['평균온도', '총강수량', '강수량_호우_count', 'covid_flag', 'ccsi_lag_m1', 'cpi_y1_prev', 'cpi_y2_prev_yoy']) },
    { group: '운영 · 구조', items: cols(['existed_before_regime', 'is_warmup', 'coldstart_flag', 'center_is_B', 'temp_x_precip', 'center_temp_inter']) },
  ];
}

// DL 공용 — get_model_feature_roles(horizon) 실제 역할 구분(27개: lag/rolling 요약 대신 원시 시퀀스 사용).
function dlVariableGroups(horizon) {
  return [
    { group: '정적 범주형 (static categorical)', items: cols(['KAN_대분류', 'KAN_중분류', 'KAN_소분류']) },
    { group: '정적 연속형 (static continuous)', items: cols(['입수', 'existed_before_regime', 'center_is_B']) },
    { group: '미래에 미리 알 수 있는 변수 (time-varying known)', items: [...cols(['ISO_주차', '월', '분기', 'covid_flag']), ...holidayCols(horizon)] },
    { group: '과거 관측 변수 (time-varying observed)', items: cols(['평균온도', '총강수량', 'qty_log1p', 'is_warmup', 'coldstart_flag', 'adi_expanding_filled', 'cv2_expanding_filled', '강수량_호우_count', 'temp_x_precip', 'center_temp_inter', 'weeks_since_last_active_filled', 'ccsi_lag_m1', 'cpi_y1_prev', 'cpi_y2_prev_yoy']) },
  ];
}

const ML_TREE_PREPROCESSING_COMMON = [
  { target: 'target(qty)', method: 'log1p 변환 후 학습, 예측 후 expm1 + 0 클리핑으로 역변환', reason: '수요 분포 왜도 감소·음수 예측 방지 (RF/LightGBM 공통)' },
];

export const ML_DL_MODEL_DETAIL = {
  RF: {
    label: 'RF', track: 'ml',
    structure: {
      library: 'sklearn.ensemble.RandomForestRegressor',
      unit: '센터 × SKU × 주 row 단위 — A센터 데이터로 학습',
      flow: ['입력 30개 feature', '여러 결정 트리 학습(bagging)', '트리 예측 평균', 'log1p → expm1 역변환'],
      purpose: '판매 이력과 입력 변수의 비선형 관계를 트리 앙상블로 학습',
    },
    variableGroups: mlVariableGroups,
    preprocessing: [
      ...ML_TREE_PREPROCESSING_COMMON,
      { target: '구조적 결측 5종(adi/cv2/qty_lag1/rollmean4/rollstd4)', method: 'train 기준 KAN 소→중→대분류→전체 순 계층형 median 대체', reason: 'RF는 결측을 직접 처리하지 못해 대체가 필요 — LightGBM(native missing 허용)과 대조되는 지점' },
      { target: 'KAN 대/중/소분류', method: 'OneHotEncoder(handle_unknown="ignore")로 인코딩', reason: '트리 분기 기준으로 범주형을 사용하기 위함, train에서 학습한 카테고리만 사용' },
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
    structure: {
      library: 'lightgbm.LGBMRegressor',
      unit: '센터 × SKU × 주 row 단위 — A센터 데이터로 학습',
      flow: ['입력 30개 feature', 'gradient boosting 트리 순차 학습', 'log1p → expm1 역변환'],
      purpose: 'RF와 동일 입력으로 gradient boosting 구조의 성능 비교',
    },
    variableGroups: mlVariableGroups,
    preprocessing: [
      ...ML_TREE_PREPROCESSING_COMMON,
      { target: 'KAN 대/중/소분류', method: 'pandas Categorical dtype으로 유지 → LightGBM native categorical 처리', reason: 'OneHot 대신 LightGBM 고유 categorical 분기 활용 (RF와 반대 전략)' },
      { target: '구조적 결측 5종(adi/cv2/qty_lag1/rollmean4/rollstd4)', method: '대체하지 않고 LightGBM native missing으로 그대로 사용', reason: 'LightGBM은 결측을 분기 기준에 포함해 직접 처리 가능 (RF의 median 대체와 반대 전략)' },
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
    structure: {
      library: 'PyTorch nn.LSTM 직접 구현(라이브러리 래퍼 아님)',
      unit: 'A센터 전체 SKU를 풀링한 global 모델 1개 (SKU별 개별 모델 아님)',
      flow: ['13주 lookback 시퀀스', 'LSTM으로 시계열 패턴 학습', '마지막 hidden state + static feature 결합', 'Linear로 h값 직접 예측(Direct)'],
      purpose: '과거 수요 시퀀스 자체의 패턴을 학습해 미래 판매량을 예측',
    },
    variableGroups: dlVariableGroups,
    preprocessing: [
      { target: '구조적 결측(adi/cv2)', method: 'train 기준 KAN 소→중→대분류→전체 순 계층형 median 대체', reason: '시퀀스 생성 전 잔여 결측 제거' },
      { target: '13주 lookback 시퀀스', method: '(center,sku) 정렬 후 연속 이력 슬라이딩 윈도우 생성', reason: '연속성이 끊기거나 target이 NaN인 origin은 skip' },
      { target: '연속형 변수(static+time-varying)', method: 'train 기준 z-score 표준화', reason: '신경망 입력 스케일 정규화' },
      { target: 'KAN 대/중/소분류', method: 'train 기준 vocabulary로 정수 인덱스화(미확인 값은 <UNK>)', reason: 'embedding 입력을 위한 인덱싱' },
      { target: 'target', method: 'loss 계산 시 log1p 변환', reason: '수요 분포 왜도 감소' },
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
    structure: {
      library: 'pytorch_forecasting.TemporalFusionTransformer.from_dataset() (서드파티 라이브러리 그대로 사용)',
      unit: 'A센터 전체를 하나의 TimeSeriesDataSet으로 학습하는 global 모델',
      flow: ['13주 encoder + 1-step decoder 구성', 'static/known/observed 역할별 입력', 'Attention 기반 시계열 학습', 'h값 직접 예측(Direct)'],
      purpose: '변수 역할(static/known/observed)을 명시적으로 구분해 해석 가능한 attention 기반 구조로 예측',
    },
    variableGroups: dlVariableGroups,
    preprocessing: [
      { target: '구조적 결측(adi/cv2)', method: 'LSTM과 동일 계층형 median 대체', reason: '시퀀스 생성 전 잔여 결측 제거' },
      { target: 'encoder(13주) + decoder(1주)', method: 'long-format 변환 — decoder의 known(캘린더/공휴일)은 실측값, observed는 origin 마지막 시점 값 재사용', reason: '미래 실측이 없는 observed feature가 decoder로 새어 들어가지 않도록 방지(매 fold assert로 검증)' },
      { target: 'target', method: 'target_log1p 컬럼으로 직접 log1p 값 세팅, target_normalizer=identity(TFT 자체 재스케일 안 함)', reason: '이미 log1p된 값을 그대로 사용' },
      { target: 'KAN 대/중/소분류', method: 'NaNLabelEncoder(add_nan=True)로 train-fit, 미확인 카테고리는 자동 NaN 처리', reason: 'pytorch_forecasting 라이브러리 내장 encoder 사용(LSTM의 수동 <UNK> 방식과 다름)' },
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
    structure: {
      library: '커스텀 PyTorch 구현(ProbSparse self-attention encoder + encoder distilling + full-attention decoder, 원논문 구조 직접 재구현)',
      unit: 'A센터 전체를 풀링한 global encoder-decoder 모델',
      flow: ['13주 encoder 입력(known+observed)', 'label_len(=lookback/2) 과거 구간 + 1-step decoder', 'ProbSparse attention + distilling', 'h값 직접 예측(Direct)'],
      purpose: 'Long-sequence 효율화를 위한 ProbSparse attention 구조가 짧은 주간 시퀀스에서도 유효한지 확인',
    },
    variableGroups: dlVariableGroups,
    preprocessing: [
      { target: '구조적 결측(adi/cv2)', method: 'LSTM/TFT와 동일 계층형 median 대체', reason: '시퀀스 생성 전 잔여 결측 제거' },
      { target: 'encoder/decoder 텐서', method: 'decoder value는 encoder 마지막 label_len 구간의 qty_log1p(과거)+미래 1스텝 0패딩, decoder known은 캘린더/공휴일 실측값', reason: 'Informer 표준 decoder 입력 구성(label_len=lookback//2)' },
      { target: '미래 observed feature', method: 'decoder에 전달하지 않음(known 계열만 decoder에 포함)', reason: 'TFT와 동일하게 미래에 알 수 없는 observed feature의 leak 방지' },
      { target: '연속형 변수', method: 'z-score 표준화(LSTM과 SequencePreprocessor 클래스 공유)', reason: '입력 스케일 정규화' },
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
    hurdle: {
      classifierRole: '판매 발생 여부(y>0) 분류 — RandomForestClassifier',
      regressorRole: '양수 수요 행만으로 판매량 회귀 — RandomForestRegressor',
      combination: 'P(y>0) × 조건부 예측값(regressor) = 최종 예측 (soft 곱셈, threshold 없음)',
    },
    structure: {
      library: 'sklearn RandomForestClassifier + RandomForestRegressor 2단계 결합',
      unit: '센터 × SKU × 주 row 단위 — A센터 데이터로 HPO(최종 재학습/2024 Holdout은 미실행 — H-LGBM에 밀려 프로덕션 미채택)',
      flow: ['입력 30개 feature(classifier/regressor 동일)', '① 판매 발생 여부 분류(0/1)', '② 발생 시 판매량 회귀(양수 행만)', 'P(발생) × 회귀예측 = 최종 예측'],
      purpose: '0수요가 많은 간헐적 수요 구조에서, 단일 회귀모델의 과소예측 문제를 분류+회귀 분리로 완화',
    },
    variableGroups: mlVariableGroups,
    preprocessing: [
      { target: 'classification target', method: 'y_raw > 0 (원본 qty 기준 0/1)', reason: '판매 발생 여부 자체를 별도 분류' },
      { target: 'regressor 학습 데이터', method: 'y_raw > 0인 행만 사용(음수 없음, 0 제외)', reason: '조건부(발생 시) 판매량만 학습' },
      { target: '구조적 결측 5종', method: 'RF와 동일 — train 기준 KAN 계층형 median 대체', reason: 'classifier/regressor 공통' },
      { target: 'KAN 대/중/소분류', method: 'RF와 동일 — OneHotEncoder', reason: 'classifier/regressor 공통' },
      { target: 'regressor target', method: 'target_transform 하이퍼파라미터로 raw/log1p 중 HPO에서 선택(h1~h4 전부 raw로 선정)', reason: 'H-LGBM(log1p 고정)과 달리 HPO 탐색 대상 — non-hurdle baseline RF는 log1p를 쓰는데 Hurdle-RF는 raw가 더 나아 raw로 선정' },
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
    hurdle: {
      classifierRole: '판매 발생 여부(y>0) 분류 — LGBMClassifier(objective=binary)',
      regressorRole: '양수 수요 행만으로 판매량 회귀 — LGBMRegressor(target=log1p, 예측 후 expm1)',
      combination: 'P(y>0) × expm1(regressor 예측) = 최종 예측 (soft 곱셈, threshold 없음)',
    },
    structure: {
      library: 'lightgbm LGBMClassifier + LGBMRegressor 2단계 결합',
      unit: '센터 × SKU × 주 row 단위 — HP 선정은 A센터 전용, 최종 재학습은 A 전체 + B(체계변경 이후) pooled',
      flow: ['입력 30개 feature(classifier/regressor 동일)', '① 판매 발생 여부 분류(0/1)', '② 발생 시 판매량 회귀(양수 행만, log1p target)', 'P(발생) × expm1(회귀예측) = 최종 예측'],
      purpose: '0수요가 많은 간헐적 수요 구조에서 과소예측 문제 완화 — 실제 2024 Holdout까지 진행된 최종 프로덕션 모델',
    },
    variableGroups: mlVariableGroups,
    preprocessing: [
      { target: 'classification target', method: 'y_raw > 0 (원본 qty 기준 0/1)', reason: '판매 발생 여부 자체를 별도 분류' },
      { target: 'regressor 학습 데이터', method: 'y_raw > 0인 행만 사용', reason: '조건부(발생 시) 판매량만 학습 (h1 기준 전체 행의 약 26.6%만 사용)' },
      { target: 'KAN 대/중/소분류', method: 'LightGBM과 동일 — pandas Categorical dtype(native categorical)', reason: 'classifier/regressor 공통' },
      { target: '구조적 결측 5종', method: 'LightGBM과 동일 — native missing으로 그대로 사용(대체 없음)', reason: 'classifier/regressor 공통' },
      { target: 'regressor target', method: 'log1p 고정(HPO 대상 아님) — 예측 후 expm1 역변환', reason: 'target_transform을 탐색한 Hurdle-RF와 달리 log1p로 고정 (fixed_params_regressor)' },
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
