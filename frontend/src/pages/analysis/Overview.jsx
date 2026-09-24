import { useEffect, useState } from 'react';
import './analysis.css';
import './Overview.css';

function fmtNum(n) { return n.toLocaleString('ko-KR'); }

const SOURCE_ROWS = [
  { label: 'A센터 매입', period: '2021~2024', rows: 227534, cols: 21 },
  { label: 'A센터 매출', period: '2021~2024', rows: 1843836, cols: 15 },
  { label: 'B센터 매입', period: '2021~2024', rows: 345021, cols: 21 },
  { label: 'B센터 매출', period: '2021~2024', rows: 1541902, cols: 15 },
];

const SOURCE_INFO = [
  { k: '목표', v: '센터별 상품의 1·2·4주 후 주간 수요를 예측하여 발주·재고관리 의사결정을 지원' },
  { k: '예측 단위', v: '센터 × 상품 × 주' },
  { k: '상품 식별', v: '바코드 + 옵션코드(EA/BX/CS) + 상품클러스터' },
  { k: '예측 대상', v: '1주 후(h1) · 2주 후(h2) · 4주 후(h4) 주간 판매수량' },
];

const SOURCE_FLOWS = [
  { from: '매출 데이터', to: '수요예측의 판매수량 생성' },
  { from: '매입 데이터', to: '재고·입고 등 운영 분석' },
];

const EXTERNAL_SOURCES = [
  { key: 'econ', label: '경제', org: '한국은행·국가데이터처/KOSIS', period: '2017/18~2024 · 월별', purpose: '소비심리·물가', detail: 'CCSI · CPI' },
  { key: 'weather', label: '기상', org: '기상청', period: '2021~2024 · 일별', purpose: '기상 영향', detail: '온도 · 강수' },
  { key: 'holiday', label: '명절', org: '한국천문연구원', period: '2021~2024 · 일별', purpose: '명절 전후 수요', detail: '설·추석 전후' },
  { key: 'product', label: 'KAN 상품분류', org: '대한상공회의소', period: '정적 표준', purpose: '대·중·소분류 표준화', detail: 'KAN 대·중·소' },
  { key: 'covid', label: '코로나', org: '질병관리청', period: '2019~2024 · 일별', purpose: '팬데믹 영향', detail: 'COVID flag' },
];

const PREP_STEPS = [
  { n: 1, title: '상품 식별 정리', bullets: ['날짜·식별자 형식 통일', '바코드 결측 제거', '포장단위·상품 식별 기준 통일'], result: '상품 단위 확정' },
  { n: 2, title: '거래 정제', bullets: ['이상·0수량 거래 제거', '판매와 반품 분리', '비정상 반품 패턴 제외'], result: '실제 판매만 추출' },
  { n: 3, title: 'KAN 표준화', bullets: ['센터별 상이한 분류체계 통일', 'LLM 기반 KAN 분류·검증', '분류 결측 보완'], result: 'kan' },
  { n: 4, title: '주간 시계열화', bullets: ['지역별 거래 합산', '일별 → 주간 수요 집계', '거래 없는 주 0수요 생성'], result: '센터 × 상품 × 주' },
  { n: 5, title: 'Feature 생성', bullets: ['직전 수요·최근 평균/변동 생성', '간헐성·불규칙성 지표 생성', '신규·이력부족 상태 생성'], result: '모델 입력 생성' },
];

const STAT_CANDIDATES = ['ARIMA', 'ARIMAX', 'SARIMA', 'SARIMAX'];
const ML_CANDIDATES = ['RF', 'LightGBM', 'LSTM', 'TFT', 'Informer'];

const ML_DL_A_STEPS = ['2023.01~03 검증', '2023.04~06 검증', '2023.07~09 검증', '2023.10~12 검증', '2024 Holdout'];

export default function Overview() {
  const [detail, setDetail] = useState(null);
  return (
    <div className="ov-page">
      <div className="az-page-hd">
        <div>
          <h2>01 데이터·모델링 개요</h2>
          <p>실제 물류센터 데이터부터 모델 개발·검증·최종평가 구조까지 한 화면에서 확인</p>
        </div>
      </div>

      <div className="ov-main">
        <DataIntroCard onOpen={() => setDetail('source')} />
        <CenterBasisCard onOpen={() => setDetail('center')} />
        <ExternalDataCard onOpen={() => setDetail('external')} />
        <PreprocessingCard onOpen={() => setDetail('prep')} />
        <SplitCard onOpen={() => setDetail('split')} />
        <TrackCard onOpen={() => setDetail('track')} />
      </div>
      {detail && <OverviewDetailModal kind={detail} onClose={() => setDetail(null)} />}
    </div>
  );
}


function OverviewIcon({ kind }) {
  const paths = {
    econ: 'M4 20V12M10 20V7M16 20V4M3 20H21',
    weather: 'M7 17H18A4 4 0 0 0 18 9A6 6 0 0 0 6 10A3.5 3.5 0 0 0 7 17M9 20V21M15 20V21',
    holiday: 'M4 6H20V21H4ZM8 3V8M16 3V8M4 11H20M8 15H10M14 15H16',
    product: 'M3 7L12 3L21 7V17L12 21L3 17ZM3 7L12 11L21 7M12 11V21',
    covid: 'M12 3L20 6V12Q20 18 12 22Q4 18 4 12V6ZM8 12H16M12 8V16',
    database: 'M4 6C4 3 20 3 20 6V18C20 21 4 21 4 18ZM4 6C4 9 20 9 20 6M4 12C4 15 20 15 20 12',
    trophy: 'M8 4H16V8C16 11 14.2 13 12 13C9.8 13 8 11 8 8ZM8 6H4V8C4 10 5.5 11 8 11M16 6H20V8C20 10 18.5 11 16 11M12 13V17M8 21H16M9 17H15V21H9Z',
    tag: 'M3 4H12L21 13L13 21L3 11ZM7 8H8',
    filter: 'M3 4H21L14 12V20L10 18V12Z',
    layers: 'M3 7L12 3L21 7L12 11ZM3 12L12 16L21 12M3 17L12 21L21 17',
    sliders: 'M4 6H20M4 12H20M4 18H20M8 3V9M16 9V15M10 15V21',
  };
  return <svg className="ov6-small-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[kind]} /></svg>;
}

function Card({ title, subtitle, className = '', tooltip, onOpen, children }) {
  return <section className={`az-card ov6-card ov6-clickable ${className}`} data-tooltip={tooltip} onClick={onOpen} role="button" tabIndex={0} onKeyDown={(event) => { if (event.key === 'Enter' || event.key === ' ') onOpen?.(); }}><header className="ov6-card-hd"><h3>{title}</h3>{subtitle && <p>{subtitle}</p>}</header>{children}</section>;
}
function DataIntroCard({ onOpen }) {
  return <Card title="1. 실제 원천데이터" className="ov6-intro" onOpen={onOpen}>
    <div className="ov6-six-grid">
      {SOURCE_ROWS.map((r) => <article className="ov6-source-tile" key={r.label}><span className="ov6-tile-icon"><OverviewIcon kind={r.label.includes('매입') ? 'product' : 'econ'} /></span><div><b>{r.label}</b><strong>{fmtNum(r.rows)} <small>행</small></strong><span>{r.cols}개 컬럼</span></div></article>)}
      <article className="ov6-source-setting ov6-hover" data-tooltip="상품 식별 기준: 바코드 + 옵션코드(EA/BX/CS) + 상품클러스터"><span className="ov6-tile-icon"><OverviewIcon kind="database" /></span><div><b>예측 단위</b><strong>센터 × 상품 × 주</strong></div></article>
      <article className="ov6-source-setting"><span className="ov6-tile-icon"><OverviewIcon kind="econ" /></span><div><b>예측 대상</b><strong>1·2·4주 후 판매수량</strong></div></article>
    </div>
  </Card>;
}
function CenterBasisCard({ onOpen }) {
  return <Card title="2. 센터별 데이터 활용" subtitle="센터별 데이터 사용 기간과 활용 목적을 구분합니다." className="ov6-basis" onOpen={onOpen}>
    <div className="ov6-center-usage-visual">
      {['A', 'B'].map((center) => <section key={center}>
        <div className="ov6-center-time-panel"><h4>{center}센터</h4><div className={`ov6-year-line is-${center.toLowerCase()}`}><div className="ov6-year-labels"><span>2021</span><span>2022</span><span>2023</span><span>2024</span></div><div className="ov6-year-track"><i /><i /><i /><i /></div><div className="ov6-period-boxes">{center === 'A' ? <><span className="is-train">21~23 개발 활용 데이터<small>안정 이력 156주</small></span><span className="is-eval">24 평가 데이터</span></> : <><span className="is-excluded">23.07 이전 제외<small>구조 변화</small></span><span className="is-train">23.07~12 개발 활용 구간<small>안정 이력 26주</small></span><span className="is-eval">24 평가 데이터</span></>}</div></div></div>
        <div className="ov6-center-role-panel"><div><b>통계모델</b><span>{center === 'A' ? <>21~23 데이터로<br />상품별 모델 구조 결정·적합</> : <>23.07~12 데이터로<br />상품별 모델 구조 결정·적합</>}</span></div><div><b>ML/DL</b><span>{center === 'A' ? <>23년 4개 구간 검증으로<br />모델·설정 탐색</> : <>A센터에서 확정한 구조의<br />적용성 검증</>}</span></div></div>
      </section>)}
    </div>
  </Card>;
}
function ExternalDataCard({ onOpen }) {
  return <Card title="3. 외부데이터" subtitle="모델 입력에 사용한 외부정보" onOpen={onOpen}><div className="ov6-external-grid">{EXTERNAL_SOURCES.map((source) => <article className="ov6-external-item ov6-hover" key={source.key} data-tooltip={`기간·주기: ${source.period} · 목적: ${source.purpose}`}><div className="ov6-external-icon"><OverviewIcon kind={source.key} /></div><h4>{source.label}</h4><dl><div><dd>{source.org}</dd></div><div><dd>{source.detail}</dd></div></dl></article>)}</div></Card>;
}
function PreprocessingCard({ onOpen }) {
  return <Card title="4. 원천데이터 → 모델 입력 데이터" subtitle="원천 거래 데이터를 정제·표준화하여 센터 × 상품 × 주 단위의 연속 수요 시계열로 구성합니다." onOpen={onOpen}><div className="ov6-pipeline">{PREP_STEPS.map((step, i) => { return <div className="ov6-step ov6-hover" data-tooltip={step.bullets.join(' · ')} key={step.n}><div className="ov6-step-symbol"><OverviewIcon kind={["tag", "filter", "layers", "holiday", "sliders"][i]} /></div><h4>{step.title}</h4><div className="ov6-step-result">{step.result === 'kan' ? <span className="ov6-kan-result">분류 결측률 <strong className="ov6-red">44.73%</strong> → <strong className="ov6-blue">0%</strong></span> : step.result}</div></div>; })}</div></Card>;
}
function SplitCard({ onOpen }) {
  return <Card title="5. 데이터 분할 방식" subtitle="모델별 데이터 활용 순서와 예측 진행 단계를 보여줍니다." className="ov6-split-diagram" onOpen={onOpen}>
    <section className="ov6-split-diagram-group"><h4>통계모델</h4>{[['A센터', '21~23', '모델 개발'], ['B센터', '23.07~12', '모델 개발 · 26주']].map(([center, period, label]) => <div className="ov6-stat-split-row" key={center}><b>{center}</b><span><strong>{period}</strong><small>{label}</small></span><i>→</i><span className="is-eval"><strong>24년</strong><small>주차별 예측·평가<br />1·2·4주 후</small></span></div>)}</section>
    <section className="ov6-split-diagram-group is-ml"><h4>ML/DL</h4><div className="ov6-ml-split-flow"><div className="ov6-fold-stage"><strong>A센터 모델·설정 탐색</strong><small>23년 4개 구간 순차 검증</small><div>{['21~22\n학습', '23.01~03\n검증 1', '23.04~06\n검증 2', '23.07~09\n검증 3', '23.10~12\n검증 4'].map((label) => <span key={label}>{label.split('\n').map((line) => <small key={line}>{line}</small>)}</span>)}</div></div><i>→</i><span><strong>B센터 적용성<br />검증</strong><small>23.07~12 활용</small></span><i>→</i><span><strong>A+B 최종<br />재학습</strong><small>A 21~23 + B 23.07~12</small></span><i>→</i><span className="is-eval"><strong>24년 최종평가</strong><small>A/B센터 · 1·2·4주 후</small></span></div></section>
  </Card>;
}
function TrackCard({ onOpen }) {
  return (
    <Card title="6. 모델 선정 기준 및 대표모델" subtitle="다양한 접근법을 비교하여 최적의 예측 모델을 선정합니다." className="ov6-track-readable" onOpen={onOpen}>
      <div className="ov6-track-layout">
        {[['통계 트랙', STAT_CANDIDATES, 'SARIMA', '통계 대표모델']].map(([label, models, finalist, finalistLabel]) => (
          <div className={`ov6-track-lane ${label === '통계 트랙' ? 'is-stat' : 'is-ml'}`} key={label}>
            <h4>{label}</h4>
            <div className="ov6-track-lane-body">
              <div className="ov6-parallel-candidates" aria-label={`${label} 후보 모델`}>
                {models.map((model) => <span key={model}>{model}</span>)}
              </div>
              <span className="ov6-selection-arrow" aria-hidden="true">→</span>
              <div className="ov6-selected-candidate"><strong>{finalist}</strong><span>{finalistLabel}</span></div>
            </div>
          </div>
        ))}
        <div className="ov6-track-lane is-ml is-two-stage">
          <h4>ML/DL 트랙</h4>
          <div className="ov6-ml-stage-flow">
            <div><div className="ov6-parallel-candidates">{ML_CANDIDATES.map((model) => <span key={model}>{model}</span>)}</div></div>
            <span className="ov6-stage-result">→</span>
            <div><small>Hurdle 구조</small><div className="ov6-parallel-candidates is-hurdle"><span>Hurdle-RF</span><span>Hurdle-LightGBM</span></div></div>
            <span className="ov6-selection-arrow" aria-hidden="true">→</span>
            <div className="ov6-selected-candidate"><strong>Hurdle-LightGBM</strong><span>ML/DL 대표모델</span></div>
          </div>
        </div>
        <div className="ov6-track-compare">
          <p><OverviewIcon kind="trophy" />최종 비교 <small>(2024 Holdout)</small></p>
          <div><strong>SARIMA</strong><span>vs</span><strong>Hurdle-LightGBM</strong></div>
        </div>
      </div>
    </Card>
  );
}

const DETAIL_META = {
  source: { title: '원 데이터 구성', tabs: ['A센터 매입', 'A센터 매출', 'B센터 매입', 'B센터 매출'] },
  center: { title: 'A/B센터 데이터 특성 및 활용 기준', tabs: ['수요 특성 비교', '구조 변화', '적용 기준', '센터별 역할'] },
  external: { title: '외부정보 선정·가공 기준', tabs: ['경제', '기상', '명절', '상품분류', '코로나'] },
  prep: { title: '전처리 및 Feature 생성', tabs: ['원 데이터 정제', '주간 시계열화', 'Feature 생성', '모델별 실제 사용'] },
  split: { title: '검증 및 최종평가 설계', tabs: ['통계모델', 'ML/DL'] },
  track: { title: '모델 구성 및 활용 이유', tabs: ['통계 트랙', 'ML/DL 트랙'] },
};

function OverviewDetailModal({ kind, onClose }) {
  const meta = DETAIL_META[kind];
  const [tab, setTab] = useState(meta.tabs[0]);
  useEffect(() => {
    const previous = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const closeOnEscape = (event) => { if (event.key === 'Escape') onClose(); };
    window.addEventListener('keydown', closeOnEscape);
    return () => { document.body.style.overflow = previous; window.removeEventListener('keydown', closeOnEscape); };
  }, [onClose]);
  return <div className="ovd-overlay" onMouseDown={onClose} role="presentation">
    <section className="ovd-modal" role="dialog" aria-modal="true" aria-label={meta.title} onMouseDown={(event) => event.stopPropagation()}>
      <header className="ovd-head"><div><span>01 데이터·모델링 개요</span><h3>{meta.title}</h3></div><button type="button" onClick={onClose} aria-label="닫기">×</button></header>
      <nav className="ovd-tabs" aria-label="상세 항목">{meta.tabs.map((item) => <button type="button" className={tab === item ? 'active' : ''} onClick={() => setTab(item)} key={item}>{item}</button>)}</nav>
      <div className="ovd-body"><OverviewDetailContent kind={kind} tab={tab} /></div>
    </section>
  </div>;
}

function DetailTable({ headers, rows }) {
  return <div className="ovd-table-wrap"><table className="ovd-table"><thead><tr>{headers.map((h) => <th key={h}>{h}</th>)}</tr></thead><tbody>{rows.map((row, i) => <tr key={i}>{row.map((cell, j) => <td key={j}>{cell}</td>)}</tr>)}</tbody></table></div>;
}
function DetailList({ items }) { return <ul className="ovd-list">{items.map((item) => <li key={item}>{item}</li>)}</ul>; }
function DetailNote({ children, tone = '' }) { return <p className={`ovd-note ${tone}`}>{children}</p>; }

const SOURCE_SCHEMA = {
  purchase: [
    ['작업유형', '입고유형', 'object', 'string', '운영분석'], ['일자', '입고일', 'object', 'date', '사용'], ['매출처코드', '거래처 ID', 'float', 'string', '미사용'], ['매출처 우편번호', '거래처 지역', 'float', 'string', '미사용'], ['공급업체 코드', '공급업체 ID', 'int', 'string', '운영분석'], ['공급업체 우편번호', '공급업체 지역', 'int', 'string', '운영분석'], ['입고형태', '입고방식', 'object', 'string', '운영분석'], ['상품코드', '내부 상품 ID', 'int', 'string', '미사용'], ['바코드', '상품 식별', 'float', 'string', 'SKU 식별'], ['상품명', '상품명', 'object', 'string', '상품 식별'], ['규격', '상품규격', 'object', 'string', 'SKU 식별'], ['옵션코드', '포장단위', 'object', 'string', 'SKU 식별'], ['옵션', '포장명', 'object', '-', '미사용'], ['입수', '포장당 수량', 'int', 'float', '활용'], ['수량', '입고수량', 'int', 'int', '운영분석'], ['EA', '낱개환산', 'int', 'int', '운영분석'], ['판매금액', '금액', 'int', 'int', '운영분석'], ['부가세', '세액', 'int', 'int', '미사용'], ['대분류', '기존분류', 'object', 'KAN', 'KAN 대체'], ['중분류', '기존분류', 'object', 'KAN', 'KAN 대체'], ['소분류', '기존분류', 'object', 'KAN', 'KAN 대체'],
  ],
  sales: [
    ['판매일', '판매일', 'object/date', 'date', '주차 생성'], ['구분', '매출/반품', 'object', 'string', '판매/반품 분리'], ['우편번호', '배송지역', 'int', 'string', '지역결합 후 미사용'], ['매출처코드', '거래처 ID', 'int', 'string', '모델 미사용'], ['판매수량', '거래수량', 'int', 'int', '수요 생성'], ['옵션코드', '포장단위', 'object', 'string', 'SKU 식별'], ['규격', '상품규격', 'object', 'string', 'SKU 식별'], ['입수', '포장당 수량', 'int', 'float', '모델 Feature'], ['바코드', '상품 식별', 'float/int', 'string', 'SKU 식별'], ['상품명', '상품명', 'object', 'string', '조회용'], ['대분류', '기존분류', 'object', 'KAN', 'KAN 대체'], ['중분류', '기존분류', 'object', 'KAN', 'KAN 대체'], ['소분류', '기존분류', 'object', 'KAN', 'KAN 대체'], ['공급금액', '거래금액', 'int', 'int', '모델 미사용'], ['부가세', '세액', 'int', 'int', '모델 미사용'],
  ],
};

function OverviewDetailContent({ kind, tab }) {
  if (kind === 'source') {
    const rows = tab.includes('매입') ? SOURCE_SCHEMA.purchase : SOURCE_SCHEMA.sales;
    const source = SOURCE_ROWS.find((item) => tab.startsWith(item.label));
    return <><div className="ovd-summary-grid"><span>기간 <b>{source?.period}</b></span><span>원본 행 수 <b>{fmtNum(source?.rows || 0)}</b></span><span>원본 컬럼 수 <b>{source?.cols}</b></span></div><DetailTable headers={['컬럼', '설명', '원본 타입', '전처리 후', '활용 상태']} rows={rows} /><DetailNote>연도별로 다른 컬럼명은 동일 의미 기준으로 통일합니다. 예: 상품 바코드(대한상의) → 바코드 · 옵션 코드 → 옵션코드 · 공급가액 → 공급금액 · 부가세 계열 → 부가세.</DetailNote></>;
  }
  if (kind === 'center') {
    if (tab === '수요 특성 비교') return <><DetailTable headers={['지표', 'A센터', 'B센터', '의미']} rows={[["일별 Zero Ratio", '95.50%', '97.01%', '두 센터 모두 일별 상품 수요가 매우 희소하며 B가 더 희소'], ['변동계수', '0.338', '0.871', 'B의 상대적 변동성이 A보다 약 2.6배 큼'], ['장기 패턴', '비교적 안정적', '월말 집중 현상 반복', 'B에서 별도 구조 확인 필요']]} /><DetailNote>Zero Ratio: 상품별 일 단위 관측 중 판매수량이 0인 비율 · 변동계수: 평균 대비 수요 변동 크기이며 값이 높을수록 상대적인 변동성이 큼</DetailNote></>;
    if (tab === '구조 변화') return <><DetailTable headers={['확인 내용', '결과', '의미']} rows={[["월말 집중일 수요 / 평상시", '20~26배', '정상적인 일별 변동보다 매우 큼'], ['집중 현상 소멸 이후', '1.6~2.0배', '과거 패턴이 크게 감소'], ['집중 현상이 확인된 날짜', '29일', '일회성 이상치가 아님'], ['집중일 평균 참여 상품', '약 4,239개', '소수 상품 문제가 아님'], ['상위 10개 상품 수요 비중', '12.9%', '특정 상품 대량주문만으로 설명 어려움']]} /><DetailNote tone="warning">한두 상품의 대량주문이 아니라 수천 개 상품 거래가 특정 월말 날짜에 함께 기록되는 기록·집계 구조로 판단합니다.</DetailNote></>;
    if (tab === '적용 기준') return <><div className="ovd-summary-grid"><span>B센터 모델 사용 시작 <b>2023-07-03</b></span><span>구조변화 이전 제외 <b>479,291행</b></span></div><DetailNote>B센터의 2023.07 이전 판매수량은 학습에서 제외하지만 해당 상품이 과거에도 판매된 상품인지 여부는 보존합니다.</DetailNote><DetailNote>기존 상품을 7월 이후 처음 등장했다는 이유만으로 신규상품으로 잘못 판단하지 않습니다.</DetailNote></>;
    return <DetailTable headers={['구분', 'A센터', 'B센터']} rows={[["모델 개발 기간", '2021.01~2023.12', '2023.07~2023.12'], ['개발 데이터', '1,509,000행 / 12,709 SKU', '173,410행 / 7,192 SKU'], ['역할', '모델 개발·파라미터 선정의 주 기준', 'A에서 개발한 방식이 다른 센터에서도 유지되는지 확인'], ['2024', '최종평가', '최종평가']]} />;
  }
  if (kind === 'external') return <ExternalDetail tab={tab} />;
  if (kind === 'prep') return <PrepDetail tab={tab} />;
  if (kind === 'split') return <SplitDetail tab={tab} />;
  return <TrackDetail tab={tab} />;
}

function ExternalDetail({ tab }) {
  if (tab === '경제') return <><h4>기본 정보</h4><DetailTable headers={['구분', '내용']} rows={[["사용 정보", '소비자심리지수 CCSI, 소비자물가지수 CPI'], ['선정 이유', '소비심리와 물가 변화가 수요와 관련되는지 모델 투입 전에 통계적으로 확인'], ['선정 방법', '예측 시점별 후보 생성 후 Spearman 상관계수 + Mutual Information으로 비교'], ['최종 활용', 'ccsi_lag_m1 · cpi_y1_prev · cpi_y2_prev_yoy']]} /><h4>원본과 후보 생성</h4><DetailTable headers={['정보', '출처·기간·범위', '최종 변수']} rows={[["CCSI", '한국은행 소비자동향조사/KOSIS · 2018.01~2024.12 · 전국 월별', 'ccsi_lag_m1'], ['CPI', '국가데이터처 소비자물가조사/KOSIS · 2017.12~2024.12 · 전국 월별', 'cpi_y1_prev · cpi_y2_prev_yoy']]} /><DetailTable headers={['시점', '후보 수']} rows={[["올해", '직전월 1개'], ['1년 전', '전월·당월·익월 3개'], ['2년 전', '전월·당월·익월 3개'], ['3년 전', '전월·당월·익월 3개']]} /><DetailNote>CCSI 10개 + CPI 10개 = 총 20개 후보. 현재 연도의 당월값은 발표 전 정보가 될 수 있어 직전월만 사용합니다.</DetailNote><h4>선정 지표와 CPI 추가 검증</h4><DetailTable headers={['지표', '의미']} rows={[["Spearman", '+1은 함께 증가, -1은 반대 방향, 0은 뚜렷한 관계가 적음'], ['Mutual Information', '경제지표가 수요에 주는 비선형 정보량 확인']]} /><DetailTable headers={['형태', '의미', '확인 목적']} rows={[["원본", '실제 CPI 값', '기본 관계 확인'], ['추세 제거(detrend)', '장기 상승·하락 흐름 제거', '시간 경과로 함께 움직인 것인지 확인'], ['전년동월대비(YoY)', '전년 같은 달 대비 변화', '물가 상승 속도와 수요 관계 확인']]} /><h4>경제 변수 최종 선정</h4><DetailTable headers={['변수', '의미', '선정 이유']} rows={[["ccsi_lag_m1", '직전월 소비자심리지수', '원본·추세 제거 후 모두 약 +0.29'], ['cpi_y1_prev', '1년 전 직전월 CPI', '추세 제거 후에도 관계 유지 · MI 0.215'], ['cpi_y2_prev_yoy', '2년 전 직전월 CPI의 YoY', 'YoY 기준 상관 -0.286']]} /><DetailNote>소비심리 · 물가 수준 · 물가 변화속도를 대표하는 3개 변수만 사용합니다.</DetailNote></>;
  if (tab === '기상') return <><h4>기본 정보</h4><DetailTable headers={['출처', '기간·주기', '규모', '범위']} rows={[["기상청 ASOS/AWS", '2021.01~2024.12 · 일별', '61,358행 / 7컬럼', '15개 시도 / 42개 지역']]} /><DetailNote>원본 컬럼: 시도 · 시군구 · 년 · 월 · 일 · 평균온도 · 총강수량</DetailNote><h4>선정 및 연결 기준</h4><DetailTable headers={['구분', '내용']} rows={[["사용 정보", '온도·강수'], ['선정 이유', '지역별 기상 변화에 따른 상품 수요 차이 반영'], ['연결 기준', '판매지역과 대응되는 기상 관측지역'], ['최종 활용', '평균온도 · 총강수량 및 기상 파생변수']]} /><details><summary>사용 지역 42개 보기</summary><DetailTable headers={['시도', '관측지역']} rows={[["강원특별자치도", '삼척시'], ['경기도', '김포시, 성남시 분당구, 이천시, 평택시'], ['경상남도', '거제시, 고성군, 김해시, 남해군, 밀양시, 사천시, 산청군, 양산시, 의령군, 진주시, 창녕군, 창원시, 통영시, 하동군, 함안군, 합천군'], ['경상북도', '경산시, 경주시, 구미시, 문경시, 영덕군, 영천시, 울진군, 포항시'], ['광주광역시', '동구'], ['대구광역시', '시도 단위'], ['대전광역시', '동구'], ['부산광역시', '시도 단위'], ['서울특별시', '시도 단위'], ['울산광역시', '시도 단위'], ['전라남도', '순천시'], ['전북특별자치도', '익산시'], ['제주특별자치도', '제주시'], ['충청남도', '논산시, 천안시'], ['충청북도', '음성군, 제천시']]} /></details><h4>최종 활용 변수</h4><DetailTable headers={['변수', '의미']} rows={[["평균온도", '지역 평균기온'], ['총강수량', '지역 누적 강수량'], ['강수량_호우_count', '일 강수량 80mm 이상 발생 횟수'], ['temp_x_precip', '온도와 강수의 복합 영향'], ['center_temp_inter', '센터별 기온 영향 차이']]} /></>;
  if (tab === '명절') return <><DetailTable headers={['출처', '전체 공휴일', '최종 사용']} rows={[["한국천문연구원 특일정보", '74건', '설·추석 27건']]} /><DetailTable headers={['구분', '내용']} rows={[["사용 정보", '설날·추석'], ['선정 이유', '명절 전 구매 집중과 연휴 중·후 수요 변화 반영'], ['가공 방식', '명절 전주·당주·다음주 구분'], ['최종 활용', '공휴일_W-1 · 공휴일_W0 · 공휴일_W+1']]} /><DetailTable headers={['변수', '의미']} rows={[["공휴일_W-1", '명절 1주 전'], ['공휴일_W0', '명절 포함 주'], ['공휴일_W+1', '명절 1주 후']]} /><DetailNote>1주·2주·4주 후 예측 대상 주를 기준으로 생성해 명절 전후 수요 이동을 반영합니다.</DetailNote></>;
  if (tab === '상품분류') return <><DetailTable headers={['출처', '규모', '최종 활용']} rows={[["대한상공회의소", '3,088개 표준 분류코드', 'KAN_CODE · KAN_대분류 · KAN_중분류 · KAN_소분류']]} /><DetailTable headers={['구분', '내용']} rows={[["문제", '센터별 상품분류 체계가 다르고 분류 결측 존재'], ['선정 기준', '대한상공회의소 KAN을 공통 상품분류 기준으로 적용'], ['처리 방식', 'LLM으로 KAN 대·중·소분류 매핑 후 검증·보정'], ['결과', '상품분류 결측률 44.73% → 0%']]} /><div className="ovd-flow"><span>KAN 공통기준</span><b>→</b><span>LLM 자동분류</span><b>→</b><span>검증·보정</span></div><DetailNote tone="success">A·B센터 상품을 동일 분류체계로 표준화해 분석과 모델에 활용합니다.</DetailNote></>;
  return <><DetailTable headers={['구분', '내용']} rows={[["사용 정보", '코로나 영향기간 여부'], ['선정 이유', '코로나·사회적 거리두기 기간과 평상시 소비·물류 환경 구분'], ['가공 방식', '영향기간 1, 그 외 0'], ['최종 활용', 'covid_flag']]} /><DetailTable headers={['값', '기준']} rows={[["1", '2020-01-20~2022-04-17'], ['0', '그 외']]} /><DetailNote>주간 데이터는 2022-04-18주부터 0입니다.</DetailNote></>;
}

function PrepDetail({ tab }) {
  if (tab === '원 데이터 정제') return <DetailTable headers={['문제', '구체적인 문제', '규모', '처리', '결과']} rows={[
    ['날짜·식별자 타입', '날짜 문자열, 바코드·우편번호 숫자 저장', '전체', 'date/string 변환', '정렬·식별 오류 방지'], ['파일별 컬럼명', '판매일/일자, 옵션코드 표기 등 상이', '연도·센터별', '컬럼명 통일', '병합 가능'], ['바코드 결측', 'SKU 식별 불가', 'A매입 746행, B매입 106행', '제거', '식별 가능한 거래만 유지'], ['매입 이상거래', '정상 입고 흐름과 다른 거래', 'A 4,999행, B 373행', '제거', '운영데이터 유효성 확보'], ['포장단위 차이', '동일 포장단위 표기가 다름', '전체', 'EA/BX/CS 통일', '단위 표준화'], ['동일 바코드 상품 충돌', '동일 바코드라도 규격·상품이 다름', '충돌 상품', '상품클러스터 분리', 'SKU 식별 확정'], ['수량 0 거래', '판매·반품 어느 것도 아님', '177행', '제거', '정상 거래만 유지'], ['상품분류 결측·상이', '센터별 체계 차이·결측', '44.73%', 'KAN+LLM 분류', '0%'], ['판매·반품 혼재', '양수 판매·음수 반품 혼재', '전체', '판매/반품 분리', '실제 판매수요 구분'], ['비정상 반품 패턴', '대부분 반품인 거래조합', '316개 조합', '거래≥10 & 반품비율≥90% 제외', '수요 왜곡 방지'], ['지역·일별 분산', '같은 SKU가 여러 지역·날짜 행으로 존재', '전체', '지역합산 + 주 집계', '센터×SKU×주'], ['거래 없는 주 누락', '판매 0이면 행 자체가 없음', '전체 SKU', '주간 캘린더 생성 후 0수요 생성', '연속 시계열'], ['B센터 기록구조 변화', '전·후 데이터 특성이 다름', '479,291행', '2023.07 이후 사용', '동일구조 학습'], ['신규·이력부족', '과거 수요 Feature 부족', '해당 SKU', 'cold-start / warm-up', '별도 상태 관리'],
  ]} />;
  if (tab === '주간 시계열화') return <><h4>왜 일별이 아니라 주별인가?</h4><DetailTable headers={['센터', 'Lag1', 'Lag7']} rows={[["A센터", '0.0474', '0.6540'], ['B센터', '0.5482', '0.7826']]} /><DetailNote>하루 전보다 7일 전 수요와의 관계가 강해 월요일 기준 주간 수요로 집계합니다. 판매와 반품을 분리해 실제 판매수량 target을 만들고, 거래가 없는 center×SKU×week는 0으로 채워 연속 패널을 구성합니다.</DetailNote><div className="ovd-flow"><span>지역별 판매 합산</span><b>→</b><span>센터×SKU×월요일 기준 주</span><b>→</b><span>무거래 주 0 생성</span><b>→</b><span>1주·2주·4주 후 target shift</span></div><DetailTable headers={['센터', '시계열 사용 기간']} rows={[["A센터", '2021-01-04~2024-12-30'], ['B센터 모델링 안정구간', '2023-07-03~2024-12-30']]} /></>;
  if (tab === '모델별 실제 사용') return <><h4>통계모델</h4><DetailTable headers={['모델', '사용 정보']} rows={[["ARIMA", '과거 주간 수요'], ['ARIMAX', '수요 + 경제/COVID/명절 조합'], ['SARIMA', '수요 + 계절구조'], ['SARIMAX', '수요 + 계절구조 + 외생변수']]} /><h4>ML</h4><DetailTable headers={['모델', 'Feature 구성']} rows={[["RF / LightGBM / Hurdle-LightGBM", 'horizon별 30개 Feature · 상품정보, 수요이력, 수요특성, 기상, 경제, 명절, COVID, 센터정보']]} /><h4>DL</h4><DetailTable headers={['모델', 'Feature 구성']} rows={[["LSTM / TFT / Informer", '24개 Feature · 상품 고정정보, 과거 관측정보, 미래에 미리 알 수 있는 정보를 sequence로 구성']]} /></>;
  const featureRows = [
    ['현재 수요', 'qty_log1p', '현재 주 수요 log 변환', '극단값 영향 완화'], ['최근 수요', 'lag 계열', '이전 주 수요', '최근 수요정보'], ['최근 평균', 'rolling mean', '최근 기간 평균', '단기 수요수준'], ['최근 변동성', 'rolling std', '최근 기간 표준편차', '변동성'], ['간헐성', 'ADI', '판매 발생 간 평균간격', '간헐수요 특성'], ['수요 변동', 'CV²', '판매량 변동계수²', '불규칙성'], ['최근 활동', 'weeks since last active', '마지막 판매 후 경과주', '휴면상태'], ['초기 이력', 'is_warmup', '초기 이력기간 여부', '이력 부족 구분'], ['신규상품', 'coldstart_flag', '충분한 과거이력 여부', '신규상품 구분'], ['예측 목표', '1주·2주·4주 후 target', '1·2·4주 후 수요', '예측시점별 직접예측'],
  ];
  return <><DetailNote>외부정보는 별도 설명하고 수요 데이터 기반 파생변수만 표시합니다.</DetailNote><DetailTable headers={['유형', '생성 컬럼', '설명', '생성 이유']} rows={featureRows} /><h4>KAN 기반 Feature 결측 보완</h4><div className="ovd-flow"><span>KAN 소분류</span><b>→</b><span>중분류</span><b>→</b><span>대분류</span><b>→</b><span>센터 전체</span></div><DetailTable headers={['Feature', '소분류', '중분류', '대분류', '센터 전체']} rows={[["qty_lag1", '15,059', '159', '38', '3'], ['qty_rollmean_4', '42,209', '465', '120', '9']]} /><DetailNote>전체 평균보다 같은 상품군의 수요특성을 최대한 유지하기 위한 방식입니다.</DetailNote></>;
}

function SplitDetail({ tab }) {
  if (tab === '통계모델') return <DetailTable headers={['단계', 'A센터', 'B센터', '활용']} rows={[["모델 개발", '2021-01-04~2023-12-25', '2023-07-03~2023-12-25', '구조·파라미터 결정'], ['최종평가', '2024-01-01~2024-12-30', '동일', '확정 모델 성능 평가']]} />;
  return <><h4>ML/DL · A센터</h4><DetailTable headers={['검증', '학습기간', '검증기간']} rows={[["1", '2021-01-04~2022-12-26', '2023-01~03'], ['2', '2021-01-04~2023-03', '2023-04~06'], ['3', '2021-01-04~2023-06', '2023-07~09'], ['4', '2021-01-04~2023-09', '2023-10~12']]} /><DetailNote>학습기간을 누적 확대하면서 미래 기간을 검증하여 실제 운영환경을 재현합니다.</DetailNote><h4>ML/DL · B센터</h4><DetailList items={['2023.07 이후를 1주씩 앞으로 이동하며 검증', 'h1 25회 · h2 24회 · h4 22회', 'A센터에서 개발한 방식이 다른 센터에서도 동일하게 적용되는지 확인']} /><h4>최종 재학습과 평가</h4><DetailNote tone="success">최종 Hurdle-LightGBM은 A센터 2021~2023 + B센터 2023.07~2023.12 전체로 재학습한 뒤 2024를 평가합니다.</DetailNote><DetailTable headers={['구간', 'A센터', 'B센터']} rows={[["2024 이전 개발 데이터", '1,509,000행', '173,410행'], ['2024 최종평가 데이터', '739,073행', '426,622행']]} /></>;
}

function TrackDetail({ tab }) {
  if (tab === '통계 트랙') return <><DetailTable headers={['모델', '활용 이유']} rows={[["ARIMA", '과거 수요만으로 예측 가능한 수준을 확인하는 기준 모델'], ['ARIMAX', '경제·명절·COVID 등 외부정보의 추가 효과 확인'], ['SARIMA', '주간 수요의 계절적 반복 패턴 반영'], ['SARIMAX', '계절성 + 외부정보를 함께 반영했을 때의 효과 확인']]} /><DetailNote tone="success">최종 후보: SARIMA</DetailNote></>;
  return <><DetailTable headers={['모델', '활용 이유']} rows={[["Random Forest", '비선형 관계를 확인하는 기본 ML 모델'], ['LightGBM', '대규모 표형 데이터와 다양한 Feature를 효율적으로 학습'], ['LSTM', '과거 여러 주의 순차적 수요 패턴 학습'], ['TFT', '상품정보·과거 관측값·미래 달력정보를 역할별로 함께 학습'], ['Informer', 'Attention 기반으로 복합 시계열 패턴 학습 가능성 비교'], ['Hurdle-LightGBM', '0수요가 많은 데이터에서 수요 발생 여부와 발생량을 분리 예측']]} /><DetailNote tone="success">최종 후보: Hurdle-LightGBM</DetailNote></>;
}
