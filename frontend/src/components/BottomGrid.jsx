import './BottomGrid.css';

// ── Mock Data ──────────────────────────────────────────────────
const CAT_TOP5 = [
  { cat: '라면',    sales: '₩128M', pct: 22 },
  { cat: '과자류',  sales: '₩95M',  pct: 16 },
  { cat: '음료',    sales: '₩87M',  pct: 15 },
  { cat: '쌀',      sales: '₩76M',  pct: 13 },
  { cat: '채소류',  sales: '₩64M',  pct: 11 },
];

const REGION_TOP5 = [
  { region: '서울', sales: '₩214M', change: +8.2  },
  { region: '경기', sales: '₩187M', change: +5.1  },
  { region: '인천', sales: '₩98M',  change: +12.3 },
  { region: '부산', sales: '₩76M',  change: -2.1  },
  { region: '대구', sales: '₩54M',  change: +3.4  },
];

const RETURNS = {
  returnRate:    '2.3%',
  defective:      142,
  custRequest:     89,
  lowStock:         3,
  overStock:        7,
  belowSafety:      5,
};

const SURGE_TOP5 = [
  { name: '제주 감귤 2kg',   surge: +45, qty: 320 },
  { name: '경북 사과 5kg',   surge: +38, qty: 420 },
  { name: '경기 배추 3kg',   surge: +32, qty: 450 },
  { name: '전남 고구마 5kg', surge: +28, qty: 380 },
  { name: '강원 감자 3kg',   surge: +21, qty: 210 },
];

// ── Shared Card shell ──────────────────────────────────────────
function GridCard({ title, subtitle, children }) {
  return (
    <div className="bg-card">
      <div className="bg-card-hd">
        <div>
          <h4 className="bg-card-title">{title}</h4>
          {subtitle && <p className="bg-card-sub">{subtitle}</p>}
        </div>
      </div>
      <div className="bg-card-body">{children}</div>
    </div>
  );
}

// ── Card 1: Category TOP5 ──────────────────────────────────────
function CategoryTop5() {
  const maxPct = Math.max(...CAT_TOP5.map((r) => r.pct));
  return (
    <GridCard title="카테고리별 매출 TOP 5" subtitle="이번 주 기준">
      <table className="bg-table">
        <thead>
          <tr>
            <th>카테고리</th>
            <th>매출</th>
            <th>비중</th>
          </tr>
        </thead>
        <tbody>
          {CAT_TOP5.map((row, i) => (
            <tr key={row.cat}>
              <td>
                <span className="rank-badge">{i + 1}</span>
                {row.cat}
              </td>
              <td className="num">{row.sales}</td>
              <td>
                <div className="mini-bar-wrap">
                  <div className="mini-bar-track">
                    <div
                      className="mini-bar-fill"
                      style={{ width: `${(row.pct / maxPct) * 100}%`, background: '#3b82f6' }}
                    />
                  </div>
                  <span className="mini-pct">{row.pct}%</span>
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </GridCard>
  );
}

// ── Card 2: Region TOP5 ────────────────────────────────────────
function RegionTop5() {
  return (
    <GridCard title="지역별 매출 TOP 5" subtitle="이번 주 기준">
      <table className="bg-table">
        <thead>
          <tr>
            <th>지역</th>
            <th>매출</th>
            <th>전주 대비</th>
          </tr>
        </thead>
        <tbody>
          {REGION_TOP5.map((row, i) => {
            const up = row.change >= 0;
            return (
              <tr key={row.region}>
                <td>
                  <span className="rank-badge">{i + 1}</span>
                  {row.region}
                </td>
                <td className="num">{row.sales}</td>
                <td>
                  <span className={`change-badge ${up ? 'up' : 'dn'}`}>
                    {up ? '▲' : '▼'} {Math.abs(row.change)}%
                  </span>
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </GridCard>
  );
}

// ── Card 3: Returns & Inventory Summary ───────────────────────
function ReturnsSummary() {
  return (
    <GridCard title="반품 / 재고 현황" subtitle="실시간">
      <div className="summary-grid">
        <div className="summary-block">
          <span className="summary-label">반품률</span>
          <span className="summary-val neutral">{RETURNS.returnRate}</span>
        </div>
        <div className="summary-block">
          <span className="summary-label">불량 반품</span>
          <span className="summary-val danger">{RETURNS.defective}건</span>
        </div>
        <div className="summary-block">
          <span className="summary-label">고객 요청 반품</span>
          <span className="summary-val warning">{RETURNS.custRequest}건</span>
        </div>
        <div className="summary-block">
          <span className="summary-label">재고 부족 품목</span>
          <span className="summary-val danger">{RETURNS.lowStock}개</span>
        </div>
        <div className="summary-block">
          <span className="summary-label">재고 과잉 품목</span>
          <span className="summary-val warning">{RETURNS.overStock}개</span>
        </div>
        <div className="summary-block">
          <span className="summary-label">안전재고 미달</span>
          <span className="summary-val danger">{RETURNS.belowSafety}개</span>
        </div>
      </div>
    </GridCard>
  );
}

// ── Card 4: Surge TOP5 ─────────────────────────────────────────
function SurgeTop5() {
  return (
    <GridCard title="판매량 급증 TOP 5" subtitle="최근 7일">
      <div className="surge-list">
        {SURGE_TOP5.map((row, i) => (
          <div key={row.name} className="surge-row">
            <span className="rank-badge">{i + 1}</span>
            <div className="surge-info">
              <span className="surge-name">{row.name}</span>
              <span className="surge-qty">{row.qty.toLocaleString()}개</span>
            </div>
            <span className="surge-badge">
              ▲ {row.surge}%
            </span>
          </div>
        ))}
      </div>
    </GridCard>
  );
}

// ── Placeholder Card ───────────────────────────────────────────
function PlaceholderCard() {
  return (
    <div className="bg-card bg-placeholder">
      <div className="bg-placeholder-inner">
        <span className="bg-placeholder-icon">📋</span>
        <p className="bg-placeholder-text">이슈 / 날짜 요약</p>
        <p className="bg-placeholder-sub">추후 데이터 연동 예정</p>
      </div>
    </div>
  );
}

// ── Main Export ────────────────────────────────────────────────
export default function BottomGrid() {
  return (
    <div className="bottom-grid">
      <CategoryTop5 />
      <RegionTop5 />
      <ReturnsSummary />
      <SurgeTop5 />
      <PlaceholderCard />
    </div>
  );
}
