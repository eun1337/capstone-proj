import { useEffect, useState } from 'react';
import './analysis.css';
import './Overview.css';

function fmtNum(n) { return n.toLocaleString('ko-KR'); }

const PURCHASE_COLUMNS = ['작업유형', '일자', '매출처코드', '매출처 우편번호', '공급업체 코드', '공급업체 우편번호', '입고 형태', '상품코드', '바코드', '상품명', '규격', '옵션 코드', '옵션', '입수', '수량', 'EA', '판매금액', '부가세(과세)', '대분류', '중분류', '소분류'];
const SALES_COLUMNS_STD = ['판매일', '구분', '매출처 우편번호', '매출처코드', '판매수량', '옵션 코드', '규격', '입수', '바코드', '상품명', '대분류', '중분류', '소분류', '공급금액', '부가세(과세)'];
const SALES_COLUMNS_A2024 = ['판매일', '구분', '매출처코드', '우편번호', '판매수량', '옵션코드', '규격', '입수', '상품 바코드(대한상의)', '상품명', '대분류', '중분류', '소분류', '공급가액', '부가세'];

const SOURCE_FILES = [
  { id: 'a-purchase', center: 'A', type: '매입', period: '2021~2024', rows: 227534, cols: 21, columns: PURCHASE_COLUMNS },
  { id: 'a-sales-2123', center: 'A', type: '매출', period: '2021~2023', rows: 1461359, cols: 15, columns: SALES_COLUMNS_STD },
  { id: 'a-sales-2024', center: 'A', type: '매출', period: '2024', rows: 382476, cols: 15, columns: SALES_COLUMNS_A2024 },
  { id: 'b-purchase', center: 'B', type: '매입', period: '2021~2024', rows: 345021, cols: 21, columns: PURCHASE_COLUMNS },
  { id: 'b-sales-2123', center: 'B', type: '매출', period: '2021~2023', rows: 1236403, cols: 15, columns: SALES_COLUMNS_STD },
  { id: 'b-sales-2024', center: 'B', type: '매출', period: '2024', rows: 493327, cols: 15, columns: SALES_COLUMNS_STD },
];

const SOURCE_TYPES = {
  "a-purchase": {
    "작업유형": "문자열",
    "일자": "문자열",
    "매출처코드": "빈값만 관찰",
    "매출처 우편번호": "빈값만 관찰",
    "공급업체 코드": "문자열",
    "공급업체 우편번호": "문자열",
    "입고 형태": "문자열",
    "상품코드": "문자열",
    "바코드": "문자열",
    "상품명": "문자열",
    "규격": "문자열",
    "옵션 코드": "문자열",
    "옵션": "문자열",
    "입수": "숫자",
    "수량": "숫자",
    "EA": "숫자",
    "판매금액": "숫자",
    "부가세(과세)": "숫자",
    "대분류": "문자열",
    "중분류": "문자열",
    "소분류": "문자열"
  },
  "a-sales-2123": {
    "판매일": "문자열",
    "구분": "문자열",
    "매출처 우편번호": "문자열",
    "매출처코드": "문자열",
    "판매수량": "숫자",
    "옵션 코드": "문자열",
    "규격": "문자열",
    "입수": "숫자",
    "바코드": "문자열",
    "상품명": "문자열",
    "대분류": "문자열",
    "중분류": "문자열",
    "소분류": "문자열",
    "공급금액": "숫자",
    "부가세(과세)": "숫자"
  },
  "a-sales-2024": {
    "판매일": "숫자",
    "구분": "문자열",
    "매출처코드": "문자열",
    "우편번호": "문자열",
    "판매수량": "숫자",
    "옵션코드": "문자열",
    "규격": "문자열",
    "입수": "숫자",
    "상품 바코드(대한상의)": "문자열",
    "상품명": "문자열",
    "대분류": "문자열",
    "중분류": "문자열",
    "소분류": "문자열",
    "공급가액": "숫자",
    "부가세": "숫자"
  },
  "b-purchase": {
    "작업유형": "문자열",
    "일자": "문자열",
    "매출처코드": "빈값만 관찰",
    "매출처 우편번호": "빈값만 관찰",
    "공급업체 코드": "문자열",
    "공급업체 우편번호": "문자열",
    "입고 형태": "문자열",
    "상품코드": "문자열",
    "바코드": "문자열",
    "상품명": "문자열",
    "규격": "문자열",
    "옵션 코드": "문자열",
    "옵션": "문자열",
    "입수": "숫자",
    "수량": "숫자",
    "EA": "숫자",
    "판매금액": "숫자",
    "부가세(과세)": "숫자",
    "대분류": "문자열",
    "중분류": "문자열",
    "소분류": "문자열"
  },
  "b-sales-2123": {
    "판매일": "문자열",
    "구분": "문자열",
    "매출처 우편번호": "문자열",
    "매출처코드": "문자열",
    "판매수량": "숫자",
    "옵션 코드": "문자열",
    "규격": "문자열",
    "입수": "숫자",
    "바코드": "문자열",
    "상품명": "문자열",
    "대분류": "문자열",
    "중분류": "문자열",
    "소분류": "문자열",
    "공급금액": "숫자",
    "부가세(과세)": "숫자"
  },
  "b-sales-2024": {
    "판매일": "문자열",
    "구분": "문자열",
    "매출처 우편번호": "문자열",
    "매출처코드": "문자열",
    "판매수량": "숫자",
    "옵션 코드": "문자열",
    "규격": "문자열",
    "입수": "숫자",
    "바코드": "문자열",
    "상품명": "문자열",
    "대분류": "문자열",
    "중분류": "문자열",
    "소분류": "문자열",
    "공급금액": "숫자",
    "부가세(과세)": "숫자"
  }
};
const COLUMN_ROLES = {
  '작업유형': '매입 작업 구분', '일자': '매입 거래일', '판매일': '매출 거래일',
  '매출처코드': '매출처 식별', '매출처 우편번호': '매출처 지역 식별', '우편번호': '매출처 지역 식별',
  '공급업체 코드': '공급업체 식별', '공급업체 우편번호': '공급업체 지역 식별', '입고 형태': '입고 방식 구분',
  '상품코드': '원천 상품 식별 코드', '바코드': '상품 식별', '상품 바코드(대한상의)': '상품 식별',
  '상품명': '상품 이름', '규격': '상품 규격', '옵션 코드': '상품 옵션 식별', '옵션코드': '상품 옵션 식별', '옵션': '상품 옵션 표기',
  '입수': '포장당 수량', '수량': '매입 수량', 'EA': '낱개 단위 수량', '판매수량': '매출 수량',
  '판매금액': '거래 금액', '공급금액': '공급 금액', '공급가액': '공급 금액', '부가세(과세)': '부가세 금액', '부가세': '부가세 금액',
  '대분류': '상품 대분류', '중분류': '상품 중분류', '소분류': '상품 소분류', '구분': '매출 거래 구분',
};
const SOURCE_RENAME = { '공급금액': '공급가액', '우편번호': '매출처 우편번호', '상품 바코드(대한상의)': '바코드', '부가세(과세)': '부가세', '옵션 코드': '옵션코드' };
function columnNormalization(file, column) {
  const notes = [];
  if (SOURCE_RENAME[column]) notes.push(`${column} → ${SOURCE_RENAME[column]}`);
  if (column === '일자' || column === '판매일') notes.push('쉼표·공백 정리 후 날짜형 통일' + (file.id === 'a-sales-2024' ? ', 시간 제거' : ''));
  if ((column.includes('우편번호') && !(file.type === '매입' && column === '매출처 우편번호'))) notes.push('문자열 5자리 복원, 유효성 검사');
  if (column.includes('바코드')) notes.push('8·12·13·14자리 외 행 제거');
  if (file.type === '매입' && ['매출처코드', '매출처 우편번호'].includes(column)) notes.push('표본에서 빈값만 관찰');
  return notes.join(' · ') || '—';
}

const PREP_STEPS = [
  { title: '컬럼명 통일', problem: '연도·센터별 매출/매입 파일의 컬럼명이 조금씩 다름 (예: 공급금액/공급가액, 우편번호 표기 차이)', process: '컬럼명 매핑 규칙으로 표준 스키마에 맞춰 통일', result: 'A/B, 2021~2023/2024 파일을 동일한 구조로 병합' },
  { title: '데이터 타입 정리', problem: '일자 컬럼에 불필요한 문자가 섞이고 우편번호 앞자리 0이 소실됨', process: '날짜를 datetime으로 통일 변환, 우편번호는 문자열로 5자리 복원', result: '날짜·지역 조인에 사용할 수 있는 정제된 타입 확보' },
  { title: '바코드·상품 식별', problem: '표기 차이로 같은 상품이 서로 다른 상품처럼 인식되는 경우 존재', process: '옵션코드 정규화 + 바코드·옵션코드 조합에 상품명이 여러 개인 케이스를 검수해 병합', result: '바코드 + 옵션코드 + 상품클러스터로 정의된 안정적인 상품 식별자' },
  { title: '매출·반품 처리', problem: '반품을 매출과 상쇄하면 실제 판매·반품 규모를 알 수 없음', process: '수량 부호로 판매수량과 반품수량을 분리 보존 (상쇄하지 않음)', result: '판매와 반품을 구분해서 볼 수 있는 집계 구조' },
  { title: 'SKU 기준 구성', problem: '예측은 상품 단위가 아니라 실제 판매관리 단위로 이뤄져야 함', process: '바코드 × 옵션코드 × 상품클러스터를 SKU로 정의하고 센터를 더해 (센터, SKU, 주) 기준 확정', result: '센터 × SKU × 주 단위의 일관된 데이터 grain' },
  { title: '주간 집계', problem: '원천 데이터는 거래(일) 단위라 주간 예측에 바로 사용할 수 없음', process: '월요일 시작 기준으로 일→주 리샘플, 컬럼 성격에 맞게 합계·평균·최댓값 등으로 집계', result: '센터 × SKU × 주 단위 판매수량 시계열 완성' },
  { title: '결측·이상치 처리', problem: '바코드/우편번호 형식 오류, 수량 부호가 반전된 것으로 의심되는 상품 등 이상 데이터 존재', process: '형식 오류 행 제거, 부호반전 의심 상품 사전 제외, 결측 파생값은 상위 분류 단계로 대체', result: '신뢰할 수 있는 판매수량 시계열 확보' },
  { title: '외부·파생변수 결합', problem: '판매량에는 날씨·물가·공휴일 등 외부 요인의 영향이 섞여 있음', process: '날씨·물가지수·공휴일·COVID 지표를 거래 단위로 결합 후 주간 집계, 공휴일 전후·경제지표 파생변수는 최종 테이블에 추가 병합', result: '통계/ML/DL 모델이 활용하는 외생변수·파생변수 확보' },
];

const FLOW_STEPS = [
  { n: 1, icon: <IconDb />, title: '데이터 준비', desc: '원천 데이터 정합화 → SKU · 주간 예측 데이터 구성', modal: 'prep' },
  {
    n: 2, icon: <IconBranch />, title: '트랙별 모델 개발 · 검증', modal: 'dev',
    lines: [
      '공통: 시간순 데이터만 사용, 1·2·4주 후 예측을 각각 별도로 학습',
      '통계: 2021~2023 전체 데이터로 시계열 구조 결정',
      'ML: 변수 기반 학습 + 하이퍼파라미터 탐색 + 시간순 반복검증',
      'DL: 최근 시계열 입력 + HPO·Early Stopping + 동일 시간순 반복검증',
    ],
  },
  {
    n: 3, icon: <IconCheck />, title: '대표모델 선정', modal: 'select',
    lines: [
      '통계: ① 정상 수렴 확인 → ② AICc 최소 구조 선택',
      'ML·DL: ① |Bias| ≤ 20% → ② WAPE 최소 → ③ 근접 시 최악 구간 확인',
    ],
  },
  { n: 4, icon: <IconFlag />, title: '최종 성능 평가', desc: '대표모델·설정 고정 → 2021~2023 전체로 최종 학습 → 2024 Holdout 평가', modal: 'eval' },
];

const HURDLE_STEPS = [
  { title: '기준 모델 평가', desc: 'RF, LGBM, LSTM, TFT, Informer', icon: <IconGauge /> },
  { title: '지속적인 과소예측 경향 확인', desc: '대부분의 시도가 |Bias| > 20%로 기준 미충족', icon: <IconWarnTri /> },
  { title: 'Hurdle 구조 적용', desc: '판매 발생 여부와 발생 시 판매량을 분리하여 예측', icon: <IconSplit /> },
  { title: '재검증 및 후보 비교', desc: 'H-RF, H-LGBM에서 Bias 기준 통과 후보 생성', icon: <IconCompare /> },
  { title: '최종 선정', desc: '세 예측 시점 기준으로 Hurdle-LightGBM 선택', icon: <IconCheckCircle /> },
];

const MODAL_TITLES = {
  source: '원천 데이터 구조',
  prep: '데이터 준비 — 전처리 과정',
  dev: '트랙별 모델 개발 · 검증 원칙',
  select: '대표모델 선정 규칙',
  eval: '최종 성능 평가 원칙',
  hurdle: 'Hurdle 구조란',
};

export default function Overview() {
  const [modal, setModal] = useState(null);
  const openModal = (type) => setModal(type);
  const closeModal = () => setModal(null);

  return (
    <div className="ov-page">
      <div className="az-page-hd">
        <div>
          <h2>01 모델링 개요</h2>
          <p>데이터, 방법, 기준, 절차를 한눈에 확인할 수 있는 전체 요약입니다.</p>
        </div>
      </div>

      <div className="ov-main">
        <div className="ov-row ov-row-top">
          <ProblemCard />
          <SourceDataCard onOpen={openModal} />
        </div>

        <FlowCard onOpen={openModal} />

        <div className="ov-row ov-row-bottom">
          <SplitCard />
          <HurdleCard onOpen={openModal} />
          <div className="ov-representatives" aria-label="대표모델">
          <FinalCard label="통계 대표모델" model="SARIMA" desc="계절 패턴을 반영한 시계열 모델" />
          <FinalCard label="ML/DL 대표모델" model="Hurdle-LightGBM" desc="판매 발생 여부와 발생량을 분리한 2단계 모델" />
          </div>
        </div>
      </div>

      {modal && (
        <Modal kind={modal} title={MODAL_TITLES[modal]} onClose={closeModal}>
          {modal === 'source' && <SourceModalBody />}
          {modal === 'prep' && <PrepModalBody />}
          {modal === 'dev' && <DevModalBody />}
          {modal === 'select' && <SelectModalBody />}
          {modal === 'eval' && <EvalModalBody />}
          {modal === 'hurdle' && <HurdleModalBody />}
        </Modal>
      )}
    </div>
  );
}

function Modal({ kind, title, onClose, children }) {
  useEffect(() => {
    function onKey(e) { if (e.key === 'Escape') onClose(); }
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onClose]);

  return (
    <div className="ov-modal-backdrop" onClick={onClose}>
      <div className={`ov-modal ov-modal-${kind}`} role="dialog" aria-modal="true" aria-label={title} onClick={(e) => e.stopPropagation()}>
        <div className="ov-modal-hd">
          <h3>{title}</h3>
          <button className="ov-modal-close" onClick={onClose}>✕</button>
        </div>
        <div className="ov-modal-body">{children}</div>
      </div>
    </div>
  );
}

function ProblemCard() {
  return (
    <div className="az-card ov-problem-card">
      <div className="ov-icon-badge ov-badge-blue"><IconTarget /></div>
      <div className="ov-problem-text">
        <div className="ov-card-title">예측 문제</div>
        <div className="ov-problem-sub">센터 × SKU별 주간 판매수량 예측</div>
      </div>
      <div className="ov-pill-row">
        <span className="ov-pill">1주 후</span>
        <span className="ov-pill">2주 후</span>
        <span className="ov-pill">4주 후</span>
      </div>
    </div>
  );
}

function SourceDataCard({ onOpen }) {
  return (
    <button type="button" className="az-card ov-source-card" onClick={() => onOpen('source')}>
      <div className="az-card-hd">
        <h3>원천 데이터</h3>
        <p>A / B 물류센터 매출 · 매입 원본 파일</p>
      </div>
      <div className="ov-source-grid">
        {['A', 'B'].map((center) => (
          <div className="ov-source-center" key={center}>
            <div className="ov-source-center-title">{center}센터</div>
            <table className="ov-source-table">
              <thead><tr><th>구분</th><th>기간</th><th>행 수</th><th>컬럼 수</th></tr></thead>
              <tbody>{SOURCE_FILES.filter((f) => f.center === center).map((f) => (
                <tr key={f.id}><td>{f.type}</td><td>{f.period}</td><td>{fmtNum(f.rows)}</td><td>{f.cols}</td></tr>
              ))}</tbody>
            </table>
          </div>
        ))}
      </div>
      <span className="ov-more-link">전체 구조 보기 →</span>
    </button>
  );
}

function FlowCard({ onOpen }) {
  return (
    <div className="az-card ov-flow-card">
      <div className="az-card-hd">
        <h3>전체 분석 흐름</h3>
        <p>데이터 준비부터 최종 평가까지의 진행 절차</p>
      </div>
      <div className="ov-flow-steps">
        {FLOW_STEPS.map((s, i) => [
          <button type="button" className="ov-flow-step" key={`step-${s.n}`} onClick={() => onOpen(s.modal)}>
            <div className="ov-flow-step-hd">
              <span className="ov-flow-icon">{s.icon}</span>
              <span className="ov-flow-num">{s.n}</span>
            </div>
            <div className="ov-flow-step-title">{s.title}</div>
            {s.desc && <div className="ov-flow-step-desc">{s.desc}</div>}
            {s.lines && (
              <ul className="ov-flow-step-lines">
                {s.lines.map((l) => <li key={l}>{l}</li>)}
              </ul>
            )}
            <span className="ov-more-link">자세히 →</span>
          </button>,
          i < FLOW_STEPS.length - 1 && <span className="ov-flow-connector" key={`conn-${s.n}`}>→</span>,
        ])}
      </div>
    </div>
  );
}

function SplitCard() {
  return (
    <div className="az-card ov-split-card">
      <div className="az-card-hd">
        <h3>데이터 분할 방법</h3>
        <p>통계모델과 ML/DL의 데이터 분할 방식이 다릅니다.</p>
      </div>
      <div className="ov-split-section">
        <div className="ov-split-sub-hd"><IconClock /> 통계 모델</div>
        <div className="ov-split-track">
          <div className="ov-split-box ov-split-dev ov-split-wide">2021 ~ 2023 전체 Development (별도 Fold 없음)</div>
          <span className="ov-flow-arrow">›</span>
          <div className="ov-split-box ov-split-holdout">2024<br />Holdout</div>
        </div>
      </div>
      <div className="ov-split-section">
        <div className="ov-split-sub-hd"><IconLayers /> ML / DL 모델 · A센터 · 2021년부터 누적 학습</div>
        <div className="ov-split-track">
          <div className="ov-fold-list">
            {[
              ['Q1', '2022.12', '2023.01~03'],
              ['Q2', '2023.03', '2023.04~06'],
              ['Q3', '2023.06', '2023.07~09'],
              ['Q4', '2023.09', '2023.10~12'],
            ].map(([fold, trainEnd, validation]) => (
              <div className="ov-split-box ov-split-q" key={fold}>
                <strong>{fold}</strong><span>학습 누적 ~{trainEnd}</span><span>→ 검증 {validation}</span>
              </div>
            ))}
          </div>
          <span className="ov-flow-arrow">›</span>
          <div className="ov-split-box ov-split-holdout">2024<br />Holdout</div>
        </div>
      </div>
      <div className="ov-split-note"><IconInfo /> 분기 경계는 예측 대상일 기준이며, 검증 시작일 이전만 학습합니다.</div>
      <div className="ov-split-note"><IconInfo /> 1주 · 2주 · 4주 후 예측은 각각 별도로 학습 · 평가합니다.</div>
    </div>
  );
}

function HurdleCard({ onOpen }) {
  return (
    <button type="button" className="az-card ov-hurdle-card" onClick={() => onOpen('hurdle')}>
      <div className="az-card-hd">
        <h3><IconRefresh /> Hurdle 구조 도입 배경</h3>
        <p>기준 ML/DL 모델에서 발견된 문제와 개선 과정</p>
      </div>
      <div className="ov-hurdle-steps">
        {HURDLE_STEPS.map((s, i) => (
          <div className="ov-hurdle-step" key={s.title}>
            <div className="ov-hurdle-row">
              <div className="ov-hurdle-icon">{s.icon}</div>
              <div>
                <div className="ov-hurdle-title">{s.title}</div>
                <div className="ov-hurdle-desc">{s.desc}</div>
              </div>
            </div>
            {i < HURDLE_STEPS.length - 1 && <span className="ov-hurdle-arrow">↓</span>}
          </div>
        ))}
      </div>
      <span className="ov-more-link">구조 자세히 보기 →</span>
    </button>
  );
}

function FinalCard({ label, model, desc }) {
  return (
    <div className="az-card ov-final-card">
      <div className="ov-icon-badge ov-badge-blue"><IconCheck /></div>
      <div>
        <div className="ov-final-card-label">{label}</div>
        <div className="ov-final-card-model">{model}</div>
        <div className="ov-final-card-desc">{desc}</div>
      </div>
    </div>
  );
}

function SourceModalBody() {
  return (
    <>
      <table className="ov-modal-table">
        <thead><tr><th>센터</th><th>구분</th><th>기간</th><th>행 수</th><th>컬럼 수</th></tr></thead>
        <tbody>
          {SOURCE_FILES.map((f) => (
            <tr key={f.id}><td>{f.center}</td><td>{f.type}</td><td>{f.period}</td><td>{fmtNum(f.rows)}</td><td>{f.cols}</td></tr>
          ))}
        </tbody>
      </table>
      <p className="ov-modal-note">원본 파일별 컬럼명 유지 · 데이터 타입은 각 시트 첫 100행의 Excel 저장 타입 관찰값입니다. 전체 행의 타입을 보장하지 않습니다.</p>
      {SOURCE_FILES.map((file) => (
        <section className="ov-modal-section" key={file.id}>
          <h4>{file.center}센터 · {file.type} · {file.period}</h4>
          <table className="ov-modal-table ov-schema-table">
            <thead><tr><th>컬럼명</th><th>데이터 타입</th><th>의미/역할</th><th>확인된 이슈·정규화</th></tr></thead>
            <tbody>{file.columns.map((column) => (
              <tr key={column}><td>{column}</td><td>{SOURCE_TYPES[file.id][column]}</td><td>{COLUMN_ROLES[column]}</td><td>{columnNormalization(file, column)}</td></tr>
            ))}</tbody>
          </table>
        </section>
      ))}
    </>
  );
}

function PrepModalBody() {
  return (
    <div className="ov-prep-grid">
      {PREP_STEPS.map((s, i) => (
        <div className="ov-prep-item" key={s.title}>
          <div className="ov-prep-n">{i + 1}</div>
          <div className="ov-prep-body">
            <div className="ov-prep-title">{s.title}</div>
            <div className="ov-prep-row"><b>문제</b> {s.problem}</div>
            <div className="ov-prep-row"><b>처리</b> {s.process}</div>
            <div className="ov-prep-row"><b>결과</b> {s.result}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

function DevModalBody() {
  return (
    <>
      <section className="ov-modal-section">
        <h4>공통 원칙</h4>
        <ul className="ov-modal-ul">
          <li>시간순으로 정렬된 데이터만 사용 (미래 정보가 과거 시점 학습에 섞이지 않도록 함)</li>
          <li>1주 · 2주 · 4주 후 예측을 각각 독립적인 예측 대상으로 다룸</li>
        </ul>
      </section>
      <section className="ov-modal-section">
        <h4>통계 트랙</h4>
        <div className="ov-modal-kv"><span>후보 모델</span><span>ARIMA, ARIMAX(S1~S4), SARIMA, SARIMAX</span></div>
        <div className="ov-modal-kv"><span>입력</span><span>센터 × SKU 주간 판매수량</span></div>
        <div className="ov-modal-kv"><span>검증 방식</span><span>2021~2023 전체 데이터로 구조를 결정 (별도 검증 구간 없음)</span></div>
        <div className="ov-modal-kv"><span>파라미터 결정</span><span>후보 구조를 탐색해 정상 수렴 여부를 확인하고, 수렴한 후보 중 AICc가 가장 낮은 구조를 선택</span></div>
      </section>
      <section className="ov-modal-section">
        <h4>ML 트랙</h4>
        <div className="ov-modal-kv"><span>후보 모델</span><span>Random Forest, LightGBM</span></div>
        <div className="ov-modal-kv"><span>입력</span><span>30개 Feature (가격, 프로모션, 캘린더 등)</span></div>
        <div className="ov-modal-kv"><span>검증 방식</span><span>2023년을 3개월씩 4개 구간으로 나눠, 각 검증 시점 이전 데이터만 누적 학습(시간순 반복검증)</span></div>
        <div className="ov-modal-kv"><span>파라미터 결정</span><span>하이퍼파라미터를 탐색한 뒤 각 구간의 검증 결과를 종합해 후보를 비교</span></div>
      </section>
      <section className="ov-modal-section">
        <h4>DL 트랙</h4>
        <div className="ov-modal-kv"><span>후보 모델</span><span>LSTM, TFT, Informer</span></div>
        <div className="ov-modal-kv"><span>입력</span><span>27개 Feature + 최근 13주 Sequence</span></div>
        <div className="ov-modal-kv"><span>검증 방식</span><span>ML과 동일한 2023년 4개 구간 시간순 반복검증</span></div>
        <div className="ov-modal-kv"><span>파라미터 결정</span><span>HPO와 Early Stopping을 적용해 각 구간의 검증 결과를 종합해 후보를 비교</span></div>
      </section>
    </>
  );
}

function SelectModalBody() {
  return (
    <>
      <section className="ov-modal-section">
        <h4>통계 트랙</h4>
        <ol className="ov-modal-ol">
          <li>후보 구조의 정상 수렴 여부 확인</li>
          <li>정상 수렴한 후보 중 AICc가 가장 낮은 구조 선택</li>
        </ol>
      </section>
      <section className="ov-modal-section">
        <h4>ML / DL 트랙</h4>
        <ol className="ov-modal-ol">
          <li>2023년 4개 검증구간의 예측 결과를 모두 합쳐 Bias · WAPE 계산</li>
          <li>|Bias| ≤ 20%를 통과한 후보만 남김</li>
          <li>통과한 후보 중 WAPE가 가장 낮은 모델 선택</li>
          <li>성능 차이가 매우 작으면, 검증구간 중 성능이 가장 나빴던 구간까지 함께 확인해 최종 결정</li>
        </ol>
        <p className="ov-modal-note">Hurdle 구조로 만든 H-RF · H-LGBM 후보도 동일한 절차(Bias 기준 → WAPE 최소 → 필요 시 최악 구간 확인)를 거쳐 비교합니다.</p>
      </section>
      <p className="ov-modal-emphasis">정확도만 가장 높은 모델이 아니라, 예측 편향 기준을 먼저 충족한 후보 중 오차와 검증구간별 안정성을 함께 확인합니다.</p>
    </>
  );
}

function EvalModalBody() {
  return (
    <>
      <div className="ov-modal-kv"><span>평가 대상</span><span>센터 × SKU별 주간 판매수량</span></div>
      <div className="ov-modal-kv"><span>센터</span><span>A · B 센터 및 전체(ALL)</span></div>
      <div className="ov-modal-kv"><span>예측 시점</span><span>1주 후 · 2주 후 · 4주 후를 각각 독립적으로 평가</span></div>
      <div className="ov-modal-kv"><span>비교 원칙</span><span>트랙별로 확정된 대표모델과 설정을 그대로 고정한 뒤 2021~2023 전체 데이터로 최종 재학습하고, 2024 데이터에 처음 적용해 평가합니다. 2024 데이터는 학습·선택 과정에 사용하지 않습니다.</span></div>
    </>
  );
}

function HurdleModalBody() {
  return (
    <>
      <section className="ov-modal-section">
        <h4>Hurdle 구조란</h4>
        <p>판매량 예측을 두 단계로 나눕니다. ① 해당 주에 판매가 발생하는지(0 / 비0)를 분류하고, ② 판매가 발생한 경우에만 판매량을 회귀로 예측합니다. 두 단계의 결과를 결합해 최종 수요를 산출합니다.</p>
      </section>
      <section className="ov-modal-section">
        <h4>도입 목적</h4>
        <p>일반 회귀 모델은 판매가 없는 주(0)가 많은 간헐적 수요 패턴에서 지속적으로 과소예측하는 경향을 보였습니다. 발생 여부와 발생량을 분리해서 예측하면 이런 편향을 줄일 수 있습니다.</p>
      </section>
    </>
  );
}

const ICON_BASE = { viewBox: '0 0 24 24', fill: 'none', stroke: 'currentColor', strokeWidth: 1.8, strokeLinecap: 'round', strokeLinejoin: 'round' };
function IconTarget() { return <svg {...ICON_BASE}><circle cx="12" cy="12" r="8" /><circle cx="12" cy="12" r="3" /></svg>; }
function IconDb() { return <svg {...ICON_BASE}><ellipse cx="12" cy="6" rx="7" ry="3" /><path d="M5 6v12c0 1.66 3.13 3 7 3s7-1.34 7-3V6" /><path d="M5 12c0 1.66 3.13 3 7 3s7-1.34 7-3" /></svg>; }
function IconWarnTri() { return <svg {...ICON_BASE}><path d="M12 4 2 20h20L12 4z" /><line x1="12" y1="10" x2="12" y2="14.5" /><circle cx="12" cy="17.2" r="0.6" fill="currentColor" stroke="none" /></svg>; }
function IconBranch() { return <svg {...ICON_BASE}><circle cx="6" cy="6" r="2.4" /><circle cx="6" cy="18" r="2.4" /><circle cx="18" cy="12" r="2.4" /><path d="M8 6h4a4 4 0 0 1 4 4M8 18h4a4 4 0 0 0 4-4" /></svg>; }
function IconCheck() { return <svg {...ICON_BASE}><circle cx="12" cy="12" r="9" /><path d="M8 12.5l2.7 2.7L16.5 9" /></svg>; }
function IconFlag() { return <svg {...ICON_BASE}><path d="M5 21V4" /><path d="M5 5h11l-2.5 3.5L16 12H5" /></svg>; }
function IconClock() { return <svg {...ICON_BASE}><circle cx="12" cy="12" r="9" /><path d="M12 7v5l3.2 2" /></svg>; }
function IconLayers() { return <svg {...ICON_BASE}><polygon points="12 3 21 8 12 13 3 8 12 3" /><polyline points="3 16 12 21 21 16" /><polyline points="3 12 12 17 21 12" /></svg>; }
function IconInfo() { return <svg {...ICON_BASE}><circle cx="12" cy="12" r="9" /><line x1="12" y1="11" x2="12" y2="16.5" /><circle cx="12" cy="7.6" r="0.6" fill="currentColor" stroke="none" /></svg>; }
function IconRefresh() { return <svg {...ICON_BASE}><path d="M4 12a8 8 0 0 1 14-5.3L21 9" /><path d="M21 4v5h-5" /><path d="M20 12a8 8 0 0 1-14 5.3L3 15" /><path d="M3 20v-5h5" /></svg>; }
function IconGauge() { return <svg {...ICON_BASE}><path d="M4 15a8 8 0 1 1 16 0" /><path d="M12 15l3.5-4.5" /><circle cx="12" cy="15" r="1" fill="currentColor" stroke="none" /></svg>; }
function IconSplit() { return <svg {...ICON_BASE}><path d="M6 4v6a3 3 0 0 0 3 3h6a3 3 0 0 0 3-3V4" /><path d="M12 13v7" /></svg>; }
function IconCompare() { return <svg {...ICON_BASE}><path d="M4 6h6M4 12h4M4 18h6" /><path d="M14 6h6M16 12h4M14 18h6" /></svg>; }
function IconCheckCircle() { return <svg {...ICON_BASE}><circle cx="12" cy="12" r="9" /><path d="M8 12.5l2.7 2.7L16.5 9" /></svg>; }
